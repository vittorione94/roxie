"""The trainer's counters and accumulators, shared by every backend.

These drive the helpers directly: plain arithmetic over the trainer's own state,
needing no env, no agent and no replay buffer.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from gymnasium import spaces

from roxie.agents.hyperparams import AgentHyperparams
from roxie.utils.rollout import (
    ChunkSums,
    EnvPoolRollout,
    JaxRollout,
    finished_episodes,
)
from roxie.utils.trainer import Trainer, _new_epoch_acc


def _trainer(**kwargs):
    """A Trainer with no env and no agent — enough for the bookkeeping helpers."""
    return Trainer(output_dir="/nonexistent", **kwargs)


def test_fresh_run_starts_every_counter_at_zero():
    t = _trainer(epoch_steps=1000, save_steps=500)
    epochs, episodes, grads = t._init_counters()
    assert (t.steps, epochs, episodes, grads) == (0, 0, 0, 0)


def test_resume_continues_the_predecessors_counters():
    t = _trainer(
        epoch_steps=1000, save_steps=500,
        resume=dict(steps=2500, epochs=2, episodes=71, gradient_steps=900),
    )
    epochs, episodes, grads = t._init_counters()
    assert (t.steps, epochs, episodes, grads) == (2500, 2, 71, 900)
    # Cadences are phased against the TOTAL, so a resumed run's next epoch
    # boundary lands where the original run's would have: at 3000, not 3500.
    assert Trainer._crossed(2500, 3000, 1000)
    assert not Trainer._crossed(2500, 2999, 1000)


def test_a_step_jump_needs_no_rephasing():
    """A warmup fill moves `steps` without incrementing through the grid.

    Nothing has to be re-phased afterwards, because a boundary is a property of
    `steps` rather than of a counter: the jump lands where it lands.
    """
    # 0 -> 1700 crosses 1000 once, so the NEXT boundary is 2000.
    assert Trainer._crossed(0, 1700, 1000)
    assert not Trainer._crossed(1700, 1999, 1000)
    assert Trainer._crossed(1700, 2000, 1000)


def test_the_epoch_cadence_cannot_drift_off_the_grid():
    """What counting up from the last firing got wrong.

    A chunk rarely divides `epoch_steps`, so a counter overshoots by most of a
    chunk every epoch and the overshoot accumulates. Locked to the grid, the
    Nth epoch lands within one chunk of N * epoch_steps however the chunks
    slice it.
    """
    epoch, chunk = 300_000, 2048
    steps, fires = 0, []
    while steps < 3_000_000:
        prev, steps = steps, steps + chunk
        if Trainer._crossed(prev, steps, epoch):
            fires.append(steps)

    assert len(fires) == 10, "an epoch went missing"
    for n, at in enumerate(fires, start=1):
        assert 0 <= at - n * epoch < chunk, (
            f"epoch {n} fired at {at}, off the {n * epoch} grid point"
        )


class _FakeAgent:
    """Records what each checkpoint was asked for.

    Returning no payload keeps these cadence tests off the filesystem — the
    trainer skips the write, as it does for a baseline with nothing to save.
    What actually reaches disk is covered in `test_resume.py`.
    """

    def __init__(self):
        self.saves = []

    def checkpoint_payload(self, *, include_buffer=False, extra_metadata=None):
        self.saves.append(extra_metadata)
        return None


def _checkpoint(t, agent, prev_steps):
    return t._checkpoint_if_due(
        agent, epochs=1, episodes=2,
        gradient_steps=3, prev_steps=prev_steps,
    )


def test_no_save_before_the_cadence():
    t, agent = _trainer(steps=10_000, save_steps=500), _FakeAgent()
    t.steps = 400
    assert (agent.saves, _checkpoint(t, agent, 0)) == ([], False)


def test_save_on_cadence_does_not_stop():
    t, agent = _trainer(steps=10_000, save_steps=500), _FakeAgent()
    t.steps = 512
    assert not _checkpoint(t, agent, 0)
    assert len(agent.saves) == 1
    # Locked to the grid, so the stride's overshoot doesn't accumulate into an
    # ever-earlier save: the next one is at 1000, not at 1012.
    t.steps = 999
    assert not _checkpoint(t, agent, 512) and len(agent.saves) == 1
    t.steps = 1000
    assert not _checkpoint(t, agent, 999) and len(agent.saves) == 2


def test_step_budget_forces_a_final_save_and_stops():
    t, agent = _trainer(steps=1000, save_steps=10_000), _FakeAgent()
    t.steps = 1000
    assert _checkpoint(t, agent, 900) and len(agent.saves) == 1
    assert agent.saves[0]["steps"] == 1000


def _accumulate(scores, lengths, done):
    """Mask one step's finished episodes and reduce it, as a chunk of length 1.

    The stacked time axis a real chunk carries is the leading 1 here, which is
    also what `noise` and the (absent) metrics are meaned over.
    """
    ep_ret, ep_len, done_f = finished_episodes(
        jnp.asarray(scores), jnp.asarray(lengths, dtype=jnp.int32),
        jnp.asarray(done),
    )
    sums = ChunkSums.from_chunk(
        ep_ret[None], ep_len[None], done_f[None],
        jnp.zeros_like(ep_ret)[None], {}, n_steps=1,
    )
    return float(sums.count), sums


def test_only_finished_episodes_are_counted():
    # Two envs done (returns 3.0 and 7.0), two still running.
    n, sums = _accumulate([3.0, 1.5, 7.0, -2.0], [30, 15, 70, 20],
                          [True, False, True, False])
    assert n == 2.0
    assert float(sums.count) == 2.0
    assert float(sums.ret) == pytest.approx(10.0)
    assert float(sums.length) == pytest.approx(100.0)


def test_a_step_with_no_terminations_contributes_nothing():
    n, sums = _accumulate([3.0, 1.5], [30, 15], [False, False])
    assert n == 0.0
    assert all(float(getattr(sums, name)) == 0.0 for name in
               ("ret", "ret_sq", "length", "length_sq", "count"))


def test_squares_recover_the_population_std_at_the_boundary():
    # Returns 2.0 and 8.0 -> mean 5.0, population std 3.0. `_epoch_stats` gets
    # that from the `_sq` companions without keeping a host-side list.
    _, sums = _accumulate([2.0, 8.0], [10, 10], [True, True])
    scores, lengths = np.zeros(2), np.zeros(2, dtype=np.int32)
    n, mean_ret, std_ret, mean_len, std_len = Trainer._epoch_stats(
        sums, scores, lengths,
    )
    assert n == 2
    assert mean_ret == pytest.approx(5.0)
    assert std_ret == pytest.approx(3.0)
    assert (mean_len, std_len) == (pytest.approx(10.0), pytest.approx(0.0))


def test_epoch_stats_falls_back_to_in_flight_counters_when_nothing_finished():
    # Episodes longer than an epoch: the line must not go blank.
    acc = _new_epoch_acc(())
    n, mean_ret, _, mean_len, _ = Trainer._epoch_stats(
        acc["sums"], np.array([4.0, 6.0]), np.array([10, 30], dtype=np.int32),
    )
    assert n == 0
    assert mean_ret == pytest.approx(5.0)
    assert mean_len == pytest.approx(20.0)


def test_losses_are_absent_not_zero_when_an_epoch_ran_no_passes():
    """A logged 0.0 is indistinguishable from a converged loss — the ambiguity
    that hid the v1 grid's zero-gradient-step bug for a full overnight sweep."""
    acc = _new_epoch_acc(())
    assert acc["actor_losses"] == [] and acc["critic_losses"] == []


class _FakeEnv:
    """Enough env to construct a rollout, which never steps it here."""

    single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,))
    func_env = object()


class _CadencedAgent:
    def __init__(self, steps_between_updates):
        self.hp = AgentHyperparams(steps_between_updates=steps_between_updates)


def test_a_chunk_is_exactly_one_update_window():
    """What makes the fused acting chunk a throughput change and not an
    algorithm change: the actor cannot move inside a chunk, because the chunk
    ends where the learning pass fires."""
    assert Trainer._collect_steps(_CadencedAgent(2_048), 256) == 8
    assert Trainer._collect_steps(_CadencedAgent(2_048), 1) == 2_048


def test_an_agent_without_a_declared_cadence_stays_per_step():
    """A baseline has no hyperparameters at all, and a learning agent may leave
    the window at 0. Both keep the old one-step loop."""
    assert Trainer._collect_steps(_NoHyperparamsAgent(), 256) == 1
    assert Trainer._collect_steps(_CadencedAgent(0), 256) == 1


class _NoHyperparamsAgent:
    """A non-learning baseline: `hp` is None, as `Agent` declares it."""

    hp = None


def test_a_window_narrower_than_one_iteration_still_collects_a_step():
    assert Trainer._collect_steps(_CadencedAgent(64), 256) == 1


def test_the_two_rollouts_declare_matching_surfaces():
    """One loop drives both, so anything it calls must exist on each."""
    surface = ("_probe", "warmup", "step", "collect", "roll_random",
               "reset_tally", "epoch_refresh", "reset_test", "evaluate")
    for name in surface:
        assert hasattr(JaxRollout, name), f"JaxRollout is missing {name}"
        assert hasattr(EnvPoolRollout, name), f"EnvPoolRollout is missing {name}"


def test_each_backend_supplies_only_its_own_hooks():
    """The loop is shared; what a backend may differ in is not.

    `advance` (traced physics or a host callback), `observe_params`, `prepare`,
    `reset_test` and `epoch_refresh` are the seams — `collect`, `roll_random`
    and `evaluate` must be the SAME functions for both, or the two paths can
    drift apart again.
    """
    for name in ("collect", "roll_random", "evaluate", "step", "reset_tally"):
        assert JaxRollout.__dict__.get(name) is None, (
            f"JaxRollout overrides the shared {name}"
        )
        assert EnvPoolRollout.__dict__.get(name) is None, (
            f"EnvPoolRollout overrides the shared {name}"
        )


def test_a_rollout_is_frozen_once_built():
    """What a stale static argument would cost.

    `num_envs` and `metric_keys` are passed to `collect_chunk` as STATIC
    arguments. A write after the first trace would not retrace — the chunk
    would keep running against the geometry it was compiled for, silently. So
    `_probe` settles both before `__init__` and nothing may move them after.
    """
    rollout = JaxRollout(_FakeEnv(), _FakeEnv(), 4, ("height",))
    assert rollout.num_envs == 4 and rollout.metric_keys == ("height",)

    for name, value in (("metric_keys", ()), ("num_envs", 8),
                        ("environment", None)):
        with pytest.raises(AttributeError, match="frozen"):
            setattr(rollout, name, value)
    # ...and the values actually survived the attempts.
    assert rollout.num_envs == 4 and rollout.metric_keys == ("height",)


def test_the_backend_step_functions_carry_no_instance():
    """What lets the compiled programs take `advance` as a static argument.

    A bound method would hold its rollout in `__self__`, putting the instance
    back in the jit cache key and retracing the whole chunk once per rollout —
    which is exactly what a static `self` used to do.
    """
    for cls in (JaxRollout, EnvPoolRollout):
        for name in ("advance", "observe_params"):
            fn = getattr(cls, name)
            assert fn is not None, f"{cls.__name__} declares no {name}"
            assert not hasattr(fn, "__self__"), (
                f"{cls.__name__}.{name} is bound and would key the cache on "
                f"its instance"
            )
            assert hash(fn) is not None
