"""The trainer's counters and accumulators, shared by every backend.

`_run_jax` and `_run_envpool` used to be two loops keeping these books
separately, in duplicated blocks that had already drifted (warmup episodes
counted on one path only, a random-action gate on one path only). There is now
one `Trainer._run`, with the backend behind `roxie.utils.rollout` and the
learning strategy behind `roxie.utils.learner`.

These drive the helpers directly: they are plain arithmetic over the trainer's
own state and need no env, no agent and no replay buffer. The parity test is the
load-bearing one — it pins that the JAX rollout's on-device accumulation and the
EnvPool rollout's host-side accumulation compute the same numbers, which is what
lets a single loop serve both.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.utils.learner import SyncLearner, build_learner
from roxie.utils.rollout import EnvPoolRollout, JaxRollout
from roxie.utils.trainer import Trainer, _new_epoch_acc


def _trainer(**kwargs):
    """A Trainer with no env and no agent — enough for the bookkeeping helpers."""
    return Trainer(output_dir="/nonexistent", **kwargs)


# --- counters ---------------------------------------------------------------


def test_fresh_run_starts_every_counter_at_zero():
    t = _trainer(epoch_steps=1000, save_steps=500)
    epochs, episodes, grads, epoch_steps, since_save = t._init_counters()
    assert (t.steps, epochs, episodes, grads) == (0, 0, 0, 0)
    assert (epoch_steps, since_save) == (0, 0)


def test_resume_continues_the_predecessors_counters():
    t = _trainer(
        epoch_steps=1000, save_steps=500,
        resume=dict(steps=2500, epochs=2, episodes=71, gradient_steps=900),
    )
    epochs, episodes, grads, epoch_steps, since_save = t._init_counters()
    assert (t.steps, epochs, episodes, grads) == (2500, 2, 71, 900)
    # Cadences are phased against the TOTAL, so a resumed run's next epoch
    # boundary lands where the original run's would have.
    assert (epoch_steps, since_save) == (500, 0)


def test_cadence_rephases_after_warmup_jumps_steps():
    t = _trainer(epoch_steps=1000, save_steps=500)
    t._init_counters()
    t.steps = 1700                      # warmup fill, not an increment
    assert t._resync_cadence() == (700, 200)


# --- save cadence -----------------------------------------------------------


class _FakeAgent:
    def __init__(self):
        self.saves = []

    def save(self, path, include_buffer=False, extra_metadata=None):
        self.saves.append(extra_metadata)


class _RecordingLearner:
    """Records that the learner was quiesced around the checkpoint."""

    def __init__(self):
        self.events = []

    def pause(self):
        self.events.append("pause")

    def resume(self):
        self.events.append("resume")


def _checkpoint(t, agent, since_save, learner=None):
    return t._checkpoint_if_due(
        agent, learner or _RecordingLearner(), epochs=1, episodes=2,
        gradient_steps=3, steps_since_save=since_save,
    )


def test_no_save_before_the_cadence():
    t, agent = _trainer(steps=10_000, save_steps=500), _FakeAgent()
    t.steps = 400
    since_save, stop = _checkpoint(t, agent, 400)
    assert (agent.saves, since_save, stop) == ([], 400, False)


def test_save_on_cadence_rephases_and_does_not_stop():
    t, agent = _trainer(steps=10_000, save_steps=500), _FakeAgent()
    t.steps = 512
    since_save, stop = _checkpoint(t, agent, 512)
    assert len(agent.saves) == 1 and not stop
    # Re-phased against the total, so the stride's overshoot doesn't accumulate
    # into an ever-earlier save.
    assert since_save == 12


def test_step_budget_forces_a_final_save_and_stops():
    t, agent = _trainer(steps=1000, save_steps=10_000), _FakeAgent()
    t.steps = 1000
    _, stop = _checkpoint(t, agent, 0)
    assert stop and len(agent.saves) == 1
    assert agent.saves[0]["steps"] == 1000


def test_learner_is_quiesced_around_the_write():
    t, agent, learner = _trainer(steps=10_000, save_steps=500), _FakeAgent(), _RecordingLearner()
    t.steps = 512
    _checkpoint(t, agent, 512, learner)
    assert learner.events == ["pause", "resume"]


def test_no_quiesce_when_nothing_is_due():
    t, agent, learner = _trainer(steps=10_000, save_steps=500), _FakeAgent(), _RecordingLearner()
    t.steps = 400
    _checkpoint(t, agent, 400, learner)
    assert learner.events == []


# --- episode accumulation ---------------------------------------------------


def _accumulate(xp, scores, lengths, done):
    """Run one step of accumulation in the given array namespace."""
    acc = _new_epoch_acc()
    n = Trainer._accumulate_episodes(
        acc, xp.asarray(scores), xp.asarray(lengths, dtype=xp.int32),
        xp.asarray(done), xp,
    )
    return float(n), acc


def test_only_finished_episodes_are_counted():
    # Two envs done (returns 3.0 and 7.0), two still running.
    n, acc = _accumulate(np, [3.0, 1.5, 7.0, -2.0], [30, 15, 70, 20],
                         [True, False, True, False])
    assert n == 2.0
    assert float(acc["count"]) == 2.0
    assert float(acc["ret"]) == pytest.approx(10.0)
    assert float(acc["len"]) == pytest.approx(100.0)


def test_a_step_with_no_terminations_contributes_nothing():
    n, acc = _accumulate(np, [3.0, 1.5], [30, 15], [False, False])
    assert n == 0.0
    assert all(float(acc[k]) == 0.0 for k in ("ret", "ret_sq", "len", "len_sq", "count"))


def test_squares_recover_the_population_std_at_the_boundary():
    # Returns 2.0 and 8.0 -> mean 5.0, population std 3.0. `_epoch_stats` gets
    # that from the `_sq` companions without keeping a host-side list.
    _, acc = _accumulate(np, [2.0, 8.0], [10, 10], [True, True])
    scores, lengths = np.zeros(2), np.zeros(2, dtype=np.int32)
    n, mean_ret, std_ret, mean_len, std_len = Trainer._epoch_stats(acc, scores, lengths)
    assert n == 2
    assert mean_ret == pytest.approx(5.0)
    assert std_ret == pytest.approx(3.0)
    assert (mean_len, std_len) == (pytest.approx(10.0), pytest.approx(0.0))


def test_epoch_stats_falls_back_to_in_flight_counters_when_nothing_finished():
    # Episodes longer than an epoch: the line must not go blank.
    acc = _new_epoch_acc()
    n, mean_ret, _, mean_len, _ = Trainer._epoch_stats(
        acc, np.array([4.0, 6.0]), np.array([10, 30], dtype=np.int32),
    )
    assert n == 0
    assert mean_ret == pytest.approx(5.0)
    assert mean_len == pytest.approx(20.0)


def test_jax_and_numpy_accumulation_agree():
    """The parity that lets one loop serve both backends.

    The JAX rollout accumulates on device (a host reduction per step would sync
    the pipeline); the EnvPool rollout accumulates host-side. Same numbers.
    """
    rng = np.random.default_rng(0)
    scores = rng.normal(size=64) * 10.0
    lengths = rng.integers(1, 1000, size=64)
    done = rng.random(size=64) < 0.25

    n_np, acc_np = _accumulate(np, scores, lengths, done)
    n_jnp, acc_jnp = _accumulate(jnp, scores, lengths, done)

    assert n_np == pytest.approx(n_jnp)
    for key in ("ret", "ret_sq", "len", "len_sq", "count"):
        assert float(acc_np[key]) == pytest.approx(float(acc_jnp[key]), rel=1e-5)


def test_losses_are_absent_not_zero_when_an_epoch_ran_no_bursts():
    """A logged 0.0 is indistinguishable from a converged loss — the ambiguity
    that hid the v1 grid's zero-gradient-step bug for a full overnight sweep."""
    acc = _new_epoch_acc()
    assert acc["actor_losses"] == [] and acc["critic_losses"] == []


# --- learner selection ------------------------------------------------------


class _StubAgent:
    """Enough of the agent contract for the sync learner's bookkeeping."""

    normalize_observations = False

    def __init__(self, per_update=0):
        self.per_update = per_update
        self.added = 0
        self.last_action = None

    def step(self, obs, evaluate, key):
        self.last_action = obs
        return obs

    def add(self, prev_env_state, next_env_state):
        self.added += 1

    def update(self, steps, agent_rng):
        return self.per_update, 1.0, 2.0


class _AsyncCapableAgent(_StubAgent):
    def learn(self, key, n_steps=None):
        return 0.0, 0.0


class _StubRollout:
    def __init__(self, supports_async):
        self.supports_async = supports_async


def test_sync_learner_accumulates_gradient_steps_and_clears_losses():
    learner = SyncLearner(_StubAgent(per_update=3), jax.random.PRNGKey(0))
    for _ in range(2):
        learner.observe(None, None, actions=None, steps=0)
    grads, losses = learner.drain()
    assert grads == 6 and losses == [(1.0, 2.0), (1.0, 2.0)]
    # Drained losses are not re-reported; the cumulative count persists.
    assert learner.drain() == (6, [])


def test_sync_learner_records_no_loss_when_no_burst_fired():
    learner = SyncLearner(_StubAgent(per_update=0), jax.random.PRNGKey(0))
    learner.observe(None, None, actions=None, steps=0)
    assert learner.drain() == (0, [])


def test_sync_learner_pause_and_resume_are_inert():
    learner = SyncLearner(_StubAgent(), jax.random.PRNGKey(0))
    learner.pause()
    learner.resume()
    learner.stop()


def test_async_is_declined_when_not_requested():
    t = _trainer()
    t.steps = 0
    learner = build_learner(t, _StubRollout(True), _AsyncCapableAgent(),
                            jax.random.PRNGKey(0), state=None)
    assert isinstance(learner, SyncLearner)


def test_async_is_declined_for_a_backend_that_acts_on_device(capsys):
    """A background learner only overlaps when acting is off its device; on the
    JAX path it would just contend for the GPU the env step already saturates."""
    t = _trainer(async_learner=True)
    t.steps = 0
    learner = build_learner(t, _StubRollout(False), _AsyncCapableAgent(),
                            jax.random.PRNGKey(0), state=None)
    assert isinstance(learner, SyncLearner)
    assert "falling back to the synchronous learner" in capsys.readouterr().out


def test_async_is_declined_for_an_agent_without_a_learn_burst():
    t = _trainer(async_learner=True)
    t.steps = 0
    learner = build_learner(t, _StubRollout(True), _StubAgent(),
                            jax.random.PRNGKey(0), state=None)
    assert isinstance(learner, SyncLearner)


# --- rollout selection ------------------------------------------------------


def test_the_two_rollouts_declare_matching_surfaces():
    """One loop drives both, so anything it calls must exist on each."""
    surface = ("xp", "supports_async", "prepare", "warmup", "step",
               "epoch_refresh", "mining_stats", "evaluate")
    for name in surface:
        assert hasattr(JaxRollout, name), f"JaxRollout is missing {name}"
        assert hasattr(EnvPoolRollout, name), f"EnvPoolRollout is missing {name}"


def test_only_the_cpu_backend_offers_async_learning():
    assert EnvPoolRollout.supports_async and not JaxRollout.supports_async


def test_each_rollout_accumulates_in_its_own_namespace():
    assert JaxRollout.xp is jnp and EnvPoolRollout.xp is np
