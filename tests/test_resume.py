"""Resuming a run must CONTINUE it, not restart it.

A checkpoint that only carries the networks is enough to watch a policy
(`play.py`) and useless to resume training: dropping Adam's moments, the
exploration schedule's step counter, SAC's learned temperature, MPO's Lagrange
duals or the trainer's step count all produce a run that keeps going but is no
longer the run that was interrupted — and every one of those failures is silent,
visible only as a kink in a curve hours later.

So the central test here is `test_resume_continues_the_run_it_interrupted`: an
agent saved, rebuilt from its config and restored must take its NEXT gradient
step to exactly the same parameters as an agent that was never interrupted.
That is a bit-for-bit statement over every piece of state the update reads, so
it subsumes a per-field checklist and fails for anything the checkpoint forgot
— including state added to an agent later, which a hand-written list would miss.
Every learning agent goes through it.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest
from flax import nnx, struct
from omegaconf import OmegaConf

from roxie.agents.utils import build_agent
from roxie.environment import functional
from roxie.environment.vector import JaxVectorEnv
from roxie.utils.checkpoint import checkpoint_steps, find_checkpoint
from roxie.utils.trainer import Trainer
from tests import harness
from tests.harness import ACT, AGENTS, ENVS, OBS


# A minimal env, driven through the same `step` / `add` calls as the trainer


@struct.dataclass
class _Inner:
    """The fake env's state: a position and an episode clock."""

    obs: jnp.ndarray
    t: jnp.ndarray


_HORIZON = 5


class _FakeEnv(functional.FuncEnv):
    """Deterministic env with the surface the Trainer uses, and nothing else.

    A plain `FuncEnv`: single-env, pure, vmappable. Every state leaf carries a
    per-env leading axis once the driver batches it, so the auto-reset gather
    applies to all of them.
    """

    observation_space = functional.unbounded_box(OBS)
    action_space = functional.box(-1.0, 1.0, shape=(ACT,))

    def initial(self, rng, params=None):
        return _Inner(obs=jax.random.uniform(
            rng, (OBS,), dtype=jnp.float32
        ) * 0.1, t=jnp.int32(0))

    def transition(self, state, action, rng, params=None):
        return _Inner(
            obs=jnp.tanh(state.obs + 0.01 * jnp.sum(action)), t=state.t + 1,
        )

    def observation(self, state, rng, params=None):
        return state.obs

    def reward(self, state, action, next_state, rng, params=None):
        return jnp.sum(next_state.obs, axis=-1)

    def terminal(self, state, rng, params=None):
        return state.t >= _HORIZON

    def transition_info(self, state, action, next_state, params=None):
        return {"metrics": {"reward/alive": jnp.ones((), dtype=jnp.float32)}}


def _fake_vector_env(num_envs=ENVS):
    return JaxVectorEnv(_FakeEnv(), num_envs, max_episode_steps=2 * _HORIZON)


# State comparison


def _as_array(leaf):
    # Typed PRNG keys (a dropout module's rng state) refuse `np.asarray`; the
    # raw key data is what has to survive the round trip anyway.
    if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key):
        leaf = jax.random.key_data(leaf)
    return np.asarray(leaf)


def _leaves(module):
    return [_as_array(x) for x in jax.tree.leaves(nnx.state(module))]


def _modules(agent):
    """Every checkpointed module, keyed by a readable name."""
    out = {
        name: getattr(agent.state, name)
        for name in (
            "actor",
            "critic",
            "target_actor",
            "target_critic",
            "actor_optimizer",
            "critic_optimizer",
        )
        if getattr(agent.state, name, None) is not None
    }
    out.update(agent._checkpoint_modules())
    return out


def _assert_same_state(a, b):
    """Every module of `a` and `b` holds identical numbers."""
    mods_a, mods_b = _modules(a), _modules(b)
    assert set(mods_a) == set(mods_b)
    for name in mods_a:
        left, right = _leaves(mods_a[name]), _leaves(mods_b[name])
        assert len(left) == len(right), f"{name}: leaf count differs"
        for i, (x, y) in enumerate(zip(left, right)):
            np.testing.assert_array_equal(x, y, err_msg=f"{name} leaf {i} differs")
    np.testing.assert_array_equal(
        np.asarray(a.state.obs_stats.count), np.asarray(b.state.obs_stats.count)
    )
    np.testing.assert_allclose(
        np.asarray(a.state.obs_stats.sum), np.asarray(b.state.obs_stats.sum)
    )


def _differs(a, b) -> bool:
    """True if any checkpointed leaf differs — the guard against a vacuous
    equality test on two agents that were never actually trained apart."""
    for name, module in _modules(a).items():
        for x, y in zip(_leaves(module), _leaves(_modules(b)[name])):
            if x.shape != y.shape or not np.array_equal(x, y):
                return True
    return False


def _trained(name, seed_key):
    """A built agent driven for a while and updated at least once."""
    agent, _steps = harness.warmed(name, key=seed_key)
    gradient_steps, _, _ = agent.learn(jax.random.fold_in(seed_key, 99))
    assert gradient_steps > 0, f"{name}: no gradient step ran — the test is vacuous"
    return agent


def _save(agent, path, **kwargs):
    """Write one self-contained checkpoint, the way `Trainer._save` does.

    A single directory rather than a manager's step/item pair — `Agent.restore`
    reads both, and these tests want a path they chose themselves.
    """
    payload = agent.checkpoint_payload(**kwargs)
    assert payload is not None, f"{type(agent).__name__} produced no payload"
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(Path(path).resolve(), payload)


# Locating a checkpoint


class TestFindCheckpoint:
    def test_parses_step_count_from_the_directory_name(self):
        assert checkpoint_steps("/runs/x/checkpoints/500000") == 500_000
        assert checkpoint_steps("/runs/x/checkpoints") is None

    def test_accepts_run_dir_checkpoints_dir_and_step_dir(self, tmp_path):
        step = tmp_path / "checkpoints" / "1000"
        step.mkdir(parents=True)
        for candidate in (tmp_path, tmp_path / "checkpoints", step):
            assert find_checkpoint(candidate) == step.resolve()

    def test_picks_the_highest_step_not_the_newest_file(self, tmp_path):
        checkpoints = tmp_path / "checkpoints"
        for steps in (500, 9_000, 1_000):
            (checkpoints / str(steps)).mkdir(parents=True)
        # 1_000 was created last; 9_000 is the one further into training.
        assert find_checkpoint(tmp_path).name == "9000"

    def test_a_wrong_path_fails_instead_of_starting_over(self, tmp_path):
        """Silently training from scratch is the one outcome a resume must never
        produce — hours later it looks exactly like a run that diverged."""
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path / "nope")
        (tmp_path / "checkpoints").mkdir()
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path)


# The per-agent round trip


@pytest.mark.parametrize("name", AGENTS)
def test_resume_continues_the_run_it_interrupted(name, tmp_path):
    """The whole feature in one assertion, for every agent that learns.

    Three claims in sequence, each the guard of the next: a fresh agent must
    DIFFER from the trained one (or the equality below is vacuous), a restore
    must reproduce it exactly (networks, targets, optimizer slots, the buffer
    and each agent's own extra modules — SAC's temperature, MPO's duals, the
    noise module's counter), and the next gradient step must then land where it
    would have landed had the run never stopped.
    """
    key = jax.random.PRNGKey(1)
    uninterrupted = _trained(name, key)

    path = tmp_path / "1024"
    _save(uninterrupted, path, include_buffer=True)

    resumed = harness.build(name)
    assert _differs(uninterrupted, resumed), (
        "the trained agent must differ from a fresh one"
    )

    metadata = resumed.restore(path)
    assert metadata["buffer_restored"] is True
    _assert_same_state(uninterrupted, resumed)

    # Same continuation for both: the same fresh transitions, then the same
    # update with the same key.
    # Four iterations rather than a second warmup: the buffers are already
    # full, and PPO's queue is exactly `sample_sequence_length` rows wide.
    continue_key = jax.random.PRNGKey(7)
    for agent in (uninterrupted, resumed):
        harness.drive(agent, 4, continue_key)
        agent.learn(jax.random.fold_in(continue_key, 5))

    _assert_same_state(uninterrupted, resumed)


# What a checkpoint carries, driven through one agent
#
# These five are claims about the SAVE FORMAT rather than about any one
# algorithm — the per-agent round trip above is what covers the algorithms — so
# they share one trained TD3 and the two checkpoints it writes. Training an
# agent is a compile; writing a checkpoint is not.


@pytest.fixture(scope="module")
def saved(tmp_path_factory):
    """One trained agent, saved both ways: with the replay buffer and without."""
    agent = _trained("td3", jax.random.PRNGKey(3))
    root = tmp_path_factory.mktemp("saved")
    _save(agent, root / "with_buffer", include_buffer=True)
    _save(agent, root / "1024", extra_metadata={"steps": 1024})
    return agent, root


def test_the_buffer_is_opt_in_and_its_absence_is_reported(saved):
    """Default saves omit the replay buffer (it dominates the checkpoint), so
    restore must say so — that flag is what makes the trainer refill it through
    warmup instead of sampling zeros. Everything else still comes back, and the
    agent trains on."""
    _agent, root = saved
    restored = harness.build("td3")
    assert restored.restore(root / "1024")["buffer_restored"] is False

    harness.drive(restored, harness.WARMUP_ITERS, jax.random.PRNGKey(4))
    gradient_steps, _, _ = restored.learn(jax.random.PRNGKey(5))
    assert gradient_steps > 0


def test_the_restored_buffer_is_still_a_flashbax_state(saved):
    """Not the plain dict orbax hands back — a dict would fail at the next
    `add`, which is the first thing a resumed run does."""
    agent, root = saved
    restored = harness.build("td3")
    restored.restore(root / "with_buffer")

    assert type(restored.state.buffer_state) is type(agent.state.buffer_state)
    assert dataclasses.is_dataclass(restored.state.buffer_state.experience)
    # And it still accepts new data.
    harness.drive(restored, 2, jax.random.PRNGKey(13))


def test_a_buffer_from_a_different_env_count_is_refused(saved):
    """Resuming with a changed `parallel_envs` must not splice a mis-shaped
    buffer into the agent; it falls back to refilling."""
    _agent, root = saved
    cfg = harness.agent_config("td3")
    cfg.memory_config.add_batch_size = ENVS * 2
    wider = build_agent(
        cfg,
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=-jnp.ones(ACT, dtype=jnp.float32),
        action_high=jnp.ones(ACT, dtype=jnp.float32),
        noise_config=OmegaConf.load(harness.CONFIGS / "noise" / "gaussian.yaml"),
    )
    before = _leaves(wider.state.actor)
    metadata = wider.restore(root / "with_buffer")

    assert metadata["buffer_restored"] is False
    assert wider.state.buffer_state.experience.observation.shape[0] == ENVS * 2
    # The refusal is scoped to the buffer: the networks still loaded.
    assert any(
        not np.array_equal(x, y) for x, y in zip(before, _leaves(wider.state.actor))
    )


def test_exploration_schedule_does_not_restart(saved):
    """The noise module's step counter drives the decay schedule. Restarting it
    puts a deep-into-training policy back under initial-scale exploration."""
    agent, root = saved
    scale = float(agent.noise_module.get_current_scale())
    count = int(agent.noise_module.step_count.get_value())
    assert count > 0

    restored = harness.build("td3")
    assert int(restored.noise_module.step_count.get_value()) == 0

    restored.restore(root / "1024")
    assert int(restored.noise_module.step_count.get_value()) == count
    assert float(restored.noise_module.get_current_scale()) == pytest.approx(scale)


def test_stateless_agent_restores_metadata_only(saved):
    """A non-learning baseline has nothing numeric to restore, but resuming one
    must still recover the step count rather than raising.

    The base `save` declines (no `state`) rather than raising, so the checkpoint
    read here is one an agent that HAS a state wrote.
    """
    from roxie.agents.basic import NormalRandom

    _agent, root = saved
    agent = NormalRandom(
        env_obs_size=OBS, env_action_size=ACT,
        action_low=-jnp.ones(ACT, dtype=jnp.float32),
        action_high=jnp.ones(ACT, dtype=jnp.float32),
    )

    metadata = agent.restore(root / "1024")
    assert int(metadata["steps"]) == 1024
    assert metadata["buffer_restored"] is False


# Trainer wiring


class _RecordingAgent:
    """Captures what the trainer asks the checkpoint payload for."""

    def __init__(self):
        self.calls = []

    def checkpoint_payload(self, *, include_buffer=False, extra_metadata=None):
        self.calls.append((include_buffer, extra_metadata))
        return {"metadata": extra_metadata or {}}


class TestTrainerResumeWiring:
    def test_fresh_run_starts_from_zero(self):
        trainer = Trainer(output_dir="/tmp", epoch_steps=100, save_steps=100)
        assert trainer.initial_steps == 0
        assert trainer.skip_warmup is False
        assert trainer._warmup_iters(1000, 10) == 100

    def test_resume_metadata_seeds_every_counter(self):
        trainer = Trainer(
            output_dir="/tmp",
            epoch_steps=100,
            save_steps=100,
            resume={
                "steps": 5_000,
                "epochs": 12,
                "episodes": 340,
                "gradient_steps": 900,
                "buffer_restored": True,
            },
        )
        assert trainer.initial_steps == 5_000
        assert trainer.initial_epochs == 12
        assert trainer.initial_episodes == 340
        assert trainer.initial_gradient_steps == 900
        # A restored buffer is already warm; refilling would prepend a block of
        # random-action transitions to a trained policy's data.
        assert trainer.skip_warmup is True
        assert trainer._warmup_iters(1000, 10) == 0

    def test_warmup_still_runs_when_the_buffer_did_not_come_back(self):
        trainer = Trainer(
            output_dir="/tmp", epoch_steps=100, save_steps=100,
            resume={"steps": 5_000, "buffer_restored": False},
        )
        assert trainer._warmup_iters(1000, 10) == 100

    def test_checkpoints_record_the_progress_needed_to_resume_them(self, tmp_path):
        trainer = Trainer(output_dir=str(tmp_path), save_buffer=True)
        trainer.steps = 4_096
        agent = _RecordingAgent()
        trainer._save(agent, epochs=3, episodes=42, gradient_steps=77)
        trainer._checkpoint_manager.close()

        (include_buffer, metadata), = agent.calls
        assert include_buffer is True
        assert metadata == {
            "steps": 4_096, "epochs": 3, "episodes": 42, "gradient_steps": 77,
        }
        # The manager names the step directory after the env-step count, which
        # is what `find_checkpoint` reads back.
        assert (tmp_path / "checkpoints" / "4096").is_dir()

    @pytest.mark.parametrize(
        "replace_checkpoint, expected", [(True, ["300"]), (False, ["100", "200", "300"])]
    )
    def test_replace_checkpoint_is_the_retention_policy(
        self, tmp_path, replace_checkpoint, expected
    ):
        """Superseded checkpoints are pruned by orbax's `max_to_keep`.

        `replace_checkpoint: false` must keep every save — that is what makes
        `find_checkpoint`'s "highest step wins" rule meaningful, and what a run
        wanting a checkpoint series relies on.
        """
        trainer = Trainer(
            output_dir=str(tmp_path), replace_checkpoint=replace_checkpoint
        )
        agent = _RecordingAgent()
        for step in (100, 200, 300):
            trainer.steps = step
            trainer._save(agent, epochs=0, episodes=0, gradient_steps=0)
        trainer._checkpoint_manager.close()

        kept = sorted(
            path.name
            for path in (tmp_path / "checkpoints").iterdir()
            if path.is_dir() and checkpoint_steps(path) is not None
        )
        assert kept == expected


@pytest.fixture
def logging_to(tmp_path):
    """Point the process-wide logger at the test's tmp dir.

    The training loops log through the module-level logger, which creates a
    timestamped directory in the CWD when nothing initialized it — i.e. drops
    stray `log.csv` trees into the repo on every test run.
    """
    from roxie.utils import logger

    logger.initialize(path=str(tmp_path / "logs"), backends=[])
    yield
    logger.close()


def test_trainer_resumes_the_run_end_to_end(tmp_path, logging_to):
    """The integration: train, checkpoint, rebuild, resume — and land on the
    total step budget rather than running it a second time from zero."""
    first, second = 16 * ENVS, 32 * ENVS
    env, test_env = _fake_vector_env(), _fake_vector_env(num_envs=2)

    def run(steps, agent, resume=None):
        trainer = Trainer(
            output_dir=str(tmp_path), steps=steps, epoch_steps=8 * ENVS,
            save_steps=first, test_episodes=2, show_progress=False,
            save_buffer=True, resume=resume,
        )
        trainer.initialize(agent=agent, environment=env, test_environment=test_env)
        trainer.run(ENVS, nnx.Rngs(envs=0, agent=3))
        return trainer

    run(first, harness.build("td3"))

    checkpoint = find_checkpoint(tmp_path)
    assert checkpoint.name == str(first)

    resumed_agent = harness.build("td3")
    metadata = resumed_agent.restore(checkpoint)
    assert int(metadata["steps"]) == first
    assert metadata["buffer_restored"] is True

    trainer = run(second, resumed_agent, resume=metadata)

    # Continued from `first` instead of replaying it: the second leg spent
    # exactly `second - first` env steps, and skipped the warmup refill.
    assert trainer.initial_steps == first
    assert trainer.steps == second
    assert (Path(tmp_path) / "checkpoints" / str(second)).is_dir()


def test_a_resume_without_a_saved_buffer_refills_it(tmp_path, logging_to):
    """Without `save_buffer`, a resumed off-policy run must re-run warmup —
    otherwise its first gradient steps sample an all-zero buffer.

    Stops at `initialize`: running the second leg would only re-time the loop
    the test above already drives, and what the refill costs is the arithmetic
    `_warmup_iters` reports.
    """
    steps = 16 * ENVS
    env = _fake_vector_env()

    trainer = Trainer(
        output_dir=str(tmp_path), steps=steps, epoch_steps=8 * ENVS,
        save_steps=steps, test_episodes=2, show_progress=False,
    )
    trainer.initialize(
        agent=harness.build("td3"), environment=env, test_environment=env
    )
    trainer.run(ENVS, nnx.Rngs(envs=0, agent=3))

    resumed_agent = harness.build("td3")
    metadata = resumed_agent.restore(find_checkpoint(tmp_path))
    assert metadata["buffer_restored"] is False

    resumed = Trainer(
        output_dir=str(tmp_path), steps=steps + 8 * ENVS, epoch_steps=8 * ENVS,
        save_steps=steps, test_episodes=2, show_progress=False, resume=metadata,
    )
    resumed.initialize(
        agent=resumed_agent, environment=env, test_environment=env
    )
    assert resumed._warmup_iters(resumed_agent.hp.memory_warmup, ENVS) == (
        resumed_agent.hp.memory_warmup // ENVS
    )
