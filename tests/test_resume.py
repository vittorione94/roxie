"""Resuming a run must CONTINUE it, not restart it.

A checkpoint that only carries the networks is enough to watch a policy
(`play.py`) and useless to resume training: dropping Adam's moments, the
exploration schedule's step counter, SAC's learned temperature, MPO's Lagrange
duals or the trainer's step count all produce a run that keeps going but is no
longer the run that was interrupted — and every one of those failures is silent,
visible only as a kink in a curve hours later.

So the central test here is not "the weights came back". It is
`test_resume_matches_an_uninterrupted_run`: an agent saved, rebuilt from its
config and restored must take its NEXT gradient step to exactly the same
parameters as an agent that was never interrupted. That is a bit-for-bit
statement over every piece of state the update reads, so it fails for anything
the checkpoint forgot — including state added to an agent later, which is the
case a hand-written list of fields would miss.

Every agent config in `roxie/configs/agent/` that learns is covered.
"""

import dataclasses
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx, struct
from hydra.utils import instantiate
from omegaconf import OmegaConf

from roxie.utils.checkpoint import checkpoint_steps, find_checkpoint
from roxie.utils.trainer import Trainer

REPO = Path(__file__).resolve().parent.parent

# Every learning agent. The baselines in `basic.py` have no `state` to restore;
# `test_stateless_agent_restores_metadata_only` covers what resume means there.
AGENTS = ["ddpg", "td3", "d4pg", "td4", "sac", "mpo", "ppo"]

OBS, ACT, NUM_ENVS = 6, 3, 8


# --------------------------------------------------------------------------
# Agents, shrunk to test size
# --------------------------------------------------------------------------


def _config(name: str):
    """The shipped agent config, shrunk to something that runs on a CPU runner.

    Only sizes and schedules are touched — the algorithm, its module layout and
    its optimizer blocks stay exactly as shipped, because those are what the
    checkpoint format has to keep up with.
    """
    cfg = OmegaConf.create(
        {
            "env": {"parallel_envs": NUM_ENVS},
            "agent": OmegaConf.load(REPO / "roxie" / "configs" / "agent" / f"{name}.yaml"),
        }
    ).agent

    cfg.actor_config.features = [16, 16]
    cfg.critic_config.features = [16, 16]
    # Learn early and often: the point is to have optimizer moments, a moved
    # policy and a non-trivial schedule state to lose.
    for key, value in (
        ("memory_warmup", 4 * NUM_ENVS),
        ("steps_before_learning", 4 * NUM_ENVS),
        ("steps_between_updates", NUM_ENVS),
        ("learning_steps", 2),
        ("num_minibatches", 2),
        ("num_action_samples", 4),
    ):
        if key in cfg:
            cfg[key] = value

    memory = cfg.memory_config
    for key, value in (
        ("max_length", 64 * NUM_ENVS),
        ("min_length", 4 * NUM_ENVS),
        ("sample_batch_size", 16),
        ("max_length_time_axis", 16),
        ("add_batch_size", NUM_ENVS),
        ("sample_sequence_length", 4),
    ):
        if key in memory:
            memory[key] = value
    return cfg


def _build(name: str):
    """Construct the agent the way `train.py` does."""
    cfg = _config(name)
    kwargs = dict(
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=-jnp.ones(ACT),
        action_high=jnp.ones(ACT),
    )
    if name in ("ddpg", "td3", "d4pg", "td4"):
        kwargs["noise_config"] = OmegaConf.load(
            REPO / "roxie" / "configs" / "noise" / "gaussian.yaml"
        )
    return instantiate(cfg, _recursive_=False, **kwargs)


# --------------------------------------------------------------------------
# A minimal env, driven through the same `step` / `add` calls as the trainer
# --------------------------------------------------------------------------


@struct.dataclass
class _Inner:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    t: jnp.ndarray
    info: dict
    metrics: dict


@struct.dataclass
class _State:
    env_state: _Inner


_HORIZON = 5


def _env_state(t, obs):
    done = t >= _HORIZON
    return _State(
        env_state=_Inner(
            obs=obs,
            reward=jnp.sum(obs, axis=-1),
            done=done,
            t=t,
            info={"termination": done, "truncation": jnp.zeros_like(done)},
            metrics={"reward/alive": jnp.ones(obs.shape[:-1])},
        )
    )


class _FakeEnv:
    """Deterministic batched env with the surface the Trainer uses.

    Everything the training loops touch and nothing else: `reset`/`step` are
    pure and vmappable, the state exposes `obs` / `reward` / `done` /
    `info["termination"]` / `info["truncation"]` / `metrics`, and every leaf
    carries a per-env leading axis so the trainer's auto-reset gather applies.
    """

    observation_size = OBS
    action_size = ACT
    max_episode_steps = 2 * _HORIZON

    def reset(self, key):
        obs = jax.random.uniform(key, (OBS,)) * 0.1
        return _env_state(jnp.int32(0), obs)

    def step(self, state, action):
        t = state.env_state.t + 1
        obs = jnp.tanh(state.env_state.obs + 0.01 * jnp.sum(action))
        return _env_state(t, obs)


def _drive(agent, iterations, key, num_envs=NUM_ENVS):
    """Push `iterations` env steps through the agent's own `step` + `add`.

    Deliberately the agent's public path rather than a direct buffer poke: it is
    what fills the replay buffer, advances the exploration noise counter and
    accumulates the observation statistics, i.e. most of the state a resume has
    to carry.
    """
    obs = jnp.zeros((num_envs, OBS))
    prev = _env_state(jnp.zeros(num_envs, jnp.int32), obs)
    for i in range(iterations):
        step_key = jax.random.fold_in(key, i)
        agent.step(prev.env_state.obs, evaluate=False, key=step_key)
        nxt = _env_state(
            prev.env_state.t + 1,
            jnp.tanh(prev.env_state.obs + 0.05 * jax.random.normal(step_key, (num_envs, OBS))),
        )
        agent.add(prev.env_state, nxt.env_state)
        prev = nxt
    return prev


# --------------------------------------------------------------------------
# State comparison
# --------------------------------------------------------------------------


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


def _trained(name, seed_key, iterations=8):
    """A built agent driven for a while and updated at least once."""
    agent = _build(name)
    _drive(agent, iterations, seed_key)
    gradient_steps, _, _ = agent.update(
        steps=100 * NUM_ENVS, agent_rng=jax.random.fold_in(seed_key, 99)
    )
    assert gradient_steps > 0, f"{name}: no gradient step ran — the test is vacuous"
    return agent


# --------------------------------------------------------------------------
# Locating a checkpoint
# --------------------------------------------------------------------------


class TestFindCheckpoint:
    def test_parses_step_count_from_the_directory_name(self):
        assert checkpoint_steps("/runs/x/checkpoints/step_500000") == 500_000
        assert checkpoint_steps("/runs/x/checkpoints") is None

    def test_accepts_run_dir_checkpoints_dir_and_step_dir(self, tmp_path):
        step = tmp_path / "checkpoints" / "step_1000"
        step.mkdir(parents=True)
        for candidate in (tmp_path, tmp_path / "checkpoints", step):
            assert find_checkpoint(candidate) == step.resolve()

    def test_picks_the_highest_step_not_the_newest_file(self, tmp_path):
        checkpoints = tmp_path / "checkpoints"
        for steps in (500, 9_000, 1_000):
            (checkpoints / f"step_{steps}").mkdir(parents=True)
        # 1_000 was created last; 9_000 is the one further into training.
        assert find_checkpoint(tmp_path).name == "step_9000"

    def test_a_wrong_path_fails_instead_of_starting_over(self, tmp_path):
        """Silently training from scratch is the one outcome a resume must never
        produce — hours later it looks exactly like a run that diverged."""
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path / "nope")
        (tmp_path / "checkpoints").mkdir()
        with pytest.raises(FileNotFoundError):
            find_checkpoint(tmp_path)


# --------------------------------------------------------------------------
# Per-agent round trip
# --------------------------------------------------------------------------


@pytest.mark.parametrize("name", AGENTS)
def test_restore_recovers_every_checkpointed_module(name, tmp_path):
    """Networks, targets, optimizer slots, the buffer and each agent's own
    extra modules all come back."""
    agent = _trained(name, jax.random.PRNGKey(0))
    path = tmp_path / "step_1024"
    agent.save(path, include_buffer=True)

    fresh = _build(name)
    assert _differs(agent, fresh), "the trained agent must differ from a fresh one"

    metadata = fresh.restore(path)
    assert metadata["buffer_restored"] is True
    _assert_same_state(agent, fresh)


@pytest.mark.parametrize("name", AGENTS)
def test_resume_matches_an_uninterrupted_run(name, tmp_path):
    """The next gradient step after a resume must land where it would have
    landed had the run never stopped.

    This is the whole feature in one assertion: it reads every piece of state
    the update touches, so it fails for any of them the checkpoint drops —
    Adam's moments, SAC's temperature, MPO's duals, the update-schedule
    boundary — without naming them.
    """
    key = jax.random.PRNGKey(1)
    uninterrupted = _trained(name, key)

    path = tmp_path / "step_1024"
    uninterrupted.save(path, include_buffer=True)
    resumed = _build(name)
    resumed.restore(path)

    # Same continuation for both: the same fresh transitions, then the same
    # update with the same key.
    continue_key = jax.random.PRNGKey(7)
    for agent in (uninterrupted, resumed):
        _drive(agent, 6, continue_key)
        agent.update(steps=200 * NUM_ENVS, agent_rng=jax.random.fold_in(continue_key, 5))

    _assert_same_state(uninterrupted, resumed)


@pytest.mark.parametrize("name", AGENTS)
def test_optimizer_moments_are_restored_not_reinitialized(name, tmp_path):
    """A resume that reinitialized the optimizers would be a fine-tune: Adam's
    first steps after a cold start are effectively unscaled, so the policy jumps
    exactly where the checkpoint says it had converged."""
    agent = _trained(name, jax.random.PRNGKey(2))
    path = tmp_path / "step_1024"
    agent.save(path)

    restored = _build(name)
    restored.restore(path)
    cold = _build(name)
    cold.restore(path, restore_optimizers=False)

    def moments(a):
        return _leaves(a.state.actor_optimizer)

    saved = moments(agent)
    assert any(np.any(x != 0) for x in saved), "no optimizer moments to restore"
    for x, y in zip(saved, moments(restored)):
        np.testing.assert_array_equal(x, y)
    assert any(
        not np.array_equal(x, y) for x, y in zip(saved, moments(cold))
    ), "restore_optimizers=False still restored the optimizer"


@pytest.mark.parametrize("name", AGENTS)
def test_buffer_is_opt_in_and_its_absence_is_reported(name, tmp_path):
    """Default saves omit the replay buffer (it dominates the checkpoint), so
    restore must say so — that flag is what makes the trainer refill it through
    warmup instead of sampling zeros."""
    agent = _trained(name, jax.random.PRNGKey(3))
    path = tmp_path / "step_1024"
    agent.save(path)

    restored = _build(name)
    metadata = restored.restore(path)
    assert metadata["buffer_restored"] is False

    # Everything except the buffer still came back, and the agent trains on.
    _drive(restored, 8, jax.random.PRNGKey(4))
    gradient_steps, _, _ = restored.update(
        steps=300 * NUM_ENVS, agent_rng=jax.random.PRNGKey(5)
    )
    assert gradient_steps > 0


def test_a_buffer_from_a_different_env_count_is_refused(tmp_path):
    """Resuming with a changed `parallel_envs` must not splice a mis-shaped
    buffer into the agent; it falls back to refilling."""
    agent = _trained("td3", jax.random.PRNGKey(6))
    path = tmp_path / "step_1024"
    agent.save(path, include_buffer=True)

    cfg = _config("td3")
    cfg.memory_config.add_batch_size = NUM_ENVS * 2
    wider = instantiate(
        cfg,
        _recursive_=False,
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=-jnp.ones(ACT),
        action_high=jnp.ones(ACT),
        noise_config=OmegaConf.load(
            REPO / "roxie" / "configs" / "noise" / "gaussian.yaml"
        ),
    )
    before = _leaves(wider.state.actor)
    metadata = wider.restore(path)

    assert metadata["buffer_restored"] is False
    assert wider.state.buffer_state.experience.observation.shape[0] == NUM_ENVS * 2
    # The refusal is scoped to the buffer: the networks still loaded.
    assert any(
        not np.array_equal(x, y) for x, y in zip(before, _leaves(wider.state.actor))
    )


def test_exploration_schedule_does_not_restart(tmp_path):
    """The noise module's step counter drives the decay schedule. Restarting it
    puts a deep-into-training policy back under initial-scale exploration."""
    agent = _trained("td3", jax.random.PRNGKey(8))
    scale = float(agent.noise_module.get_current_scale())
    count = int(agent.noise_module.step_count.value)
    assert count > 0

    path = tmp_path / "step_1024"
    agent.save(path)
    restored = _build("td3")
    assert int(restored.noise_module.step_count.value) == 0

    restored.restore(path)
    assert int(restored.noise_module.step_count.value) == count
    assert float(restored.noise_module.get_current_scale()) == pytest.approx(scale)


def test_sac_temperature_and_mpo_duals_survive(tmp_path):
    """Both agents learn scalars that live outside `self.state`; losing them
    re-runs an annealing that already converged."""
    sac = _trained("sac", jax.random.PRNGKey(9))
    mpo = _trained("mpo", jax.random.PRNGKey(10))

    for agent, read in (
        (sac, lambda a: float(a.log_alpha_module.log_alpha.value)),
        (mpo, lambda a: float(np.ravel(np.asarray(a.dual_params.log_temperature.value))[0])),
    ):
        path = tmp_path / f"step_{id(agent)}"
        agent.save(path)
        fresh = _build("sac" if agent is sac else "mpo")
        assert read(fresh) != pytest.approx(read(agent)), "value never moved"
        fresh.restore(path)
        assert read(fresh) == pytest.approx(read(agent))


def test_stateless_agent_restores_metadata_only(tmp_path):
    """A non-learning baseline has nothing numeric to restore, but resuming one
    must still recover the step count rather than raising."""
    from roxie.agents.basic import NormalRandom

    agent = NormalRandom(
        env_obs_size=OBS, env_action_size=ACT,
        action_low=-jnp.ones(ACT), action_high=jnp.ones(ACT),
    )
    path = tmp_path / "step_64"
    # The base `save` declines (no `state`) rather than raising, so write a
    # checkpoint through an agent that has one and restore it into the baseline.
    _trained("td3", jax.random.PRNGKey(11)).save(path, extra_metadata={"steps": 64})

    metadata = agent.restore(path)
    assert int(metadata["steps"]) == 64
    assert metadata["buffer_restored"] is False


# --------------------------------------------------------------------------
# Trainer wiring
# --------------------------------------------------------------------------


class _RecordingAgent:
    """Captures what the trainer asks `save` for."""

    def __init__(self):
        self.calls = []

    def save(self, path, *, include_buffer=False, extra_metadata=None):
        self.calls.append((path, include_buffer, extra_metadata))


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

        (path, include_buffer, metadata), = agent.calls
        assert Path(path).name == "step_4096"
        assert include_buffer is True
        assert metadata == {
            "steps": 4_096, "epochs": 3, "episodes": 42, "gradient_steps": 77,
        }


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
    first, second = 16 * NUM_ENVS, 32 * NUM_ENVS
    env, test_env = _FakeEnv(), _FakeEnv()

    def run(output_dir, steps, agent, resume=None):
        trainer = Trainer(
            output_dir=str(output_dir), steps=steps, epoch_steps=8 * NUM_ENVS,
            save_steps=first, test_episodes=2, show_progress=False,
            save_buffer=True, resume=resume,
        )
        trainer.initialize(agent=agent, environment=env, test_environment=test_env)
        trainer.run(NUM_ENVS, nnx.Rngs(envs=0, agent=3))
        return trainer

    agent = _build("td3")
    run(tmp_path, first, agent)

    checkpoint = find_checkpoint(tmp_path)
    assert checkpoint.name == f"step_{first}"

    resumed_agent = _build("td3")
    metadata = resumed_agent.restore(checkpoint)
    assert int(metadata["steps"]) == first
    assert metadata["buffer_restored"] is True

    trainer = run(tmp_path, second, resumed_agent, resume=metadata)

    # Continued from `first` instead of replaying it: the second leg spent
    # exactly `second - first` env steps, and skipped the warmup refill.
    assert trainer.initial_steps == first
    assert trainer.steps == second
    assert (Path(tmp_path) / "checkpoints" / f"step_{second}").is_dir()


def test_trainer_refills_the_buffer_when_the_checkpoint_has_none(tmp_path, logging_to):
    """Without `save_buffer`, a resumed off-policy run must re-run warmup —
    otherwise its first gradient steps sample an all-zero buffer."""
    steps = 16 * NUM_ENVS
    env = _FakeEnv()

    agent = _build("td3")
    trainer = Trainer(
        output_dir=str(tmp_path), steps=steps, epoch_steps=8 * NUM_ENVS,
        save_steps=steps, test_episodes=2, show_progress=False,
    )
    trainer.initialize(agent=agent, environment=env, test_environment=env)
    trainer.run(NUM_ENVS, nnx.Rngs(envs=0, agent=3))

    resumed_agent = _build("td3")
    metadata = resumed_agent.restore(find_checkpoint(tmp_path))
    assert metadata["buffer_restored"] is False

    warmup_iters = resumed_agent.memory_warmup // NUM_ENVS
    resumed = Trainer(
        output_dir=str(tmp_path), steps=steps + 8 * NUM_ENVS,
        epoch_steps=8 * NUM_ENVS, save_steps=steps, test_episodes=2,
        show_progress=False, resume=metadata,
    )
    resumed.initialize(agent=resumed_agent, environment=env, test_environment=env)
    assert resumed._warmup_iters(resumed_agent.memory_warmup, NUM_ENVS) == warmup_iters
    resumed.run(NUM_ENVS, nnx.Rngs(envs=0, agent=3))

    # Warmup steps are real env steps and count on top of the restored total.
    assert resumed.steps >= steps + warmup_iters * NUM_ENVS


def test_transition_prototype_is_unchanged_by_a_round_trip(tmp_path):
    """The restored buffer must be flashbax's own state class, not the plain
    dict orbax hands back — a dict would fail at the next `add`."""
    agent = _trained("sac", jax.random.PRNGKey(12))
    path = tmp_path / "step_1024"
    agent.save(path, include_buffer=True)

    restored = _build("sac")
    restored.restore(path)
    assert type(restored.state.buffer_state) is type(agent.state.buffer_state)
    assert dataclasses.is_dataclass(restored.state.buffer_state.experience)
    # And it still accepts new data.
    _drive(restored, 2, jax.random.PRNGKey(13))
