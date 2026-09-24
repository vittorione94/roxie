"""Shared scaffolding for the agent-level tests.

`train.py` builds an agent as one `build_agent(cfg.agent, ...)` call over the
shipped yaml, so that is what these build too — a hand-written agent config
would stop tracking the one that ships. Only sizes and schedules are shrunk
here; the algorithm, the module layout and the optimizer blocks stay exactly as
shipped, because those are what the learning pass, the checkpoint and the diagnostics
are written against.

Building an agent and running one learning pass is a compile, and the suite has seven
agents, so anything built through here is meant to be built ONCE per module and
shared by every assertion that reads it.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from omegaconf import OmegaConf

import roxie.agents  # noqa: F401  (avoid circular import)
from roxie.agents.utils import Transition, build_agent
from roxie.environment.vector import Timestep

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "roxie" / "configs"

# Every agent that learns, in the order the docs list them. The baselines in
# `basic.py` have no `state` and are covered by `test_basic_agents.py`.
AGENTS = ("ddpg", "d4pg", "td3", "td4", "sac", "mpo", "ppo")
# The deterministic arms: they explore through a noise module rather than
# through their own policy, and carry the pre-activation penalty.
DETERMINISTIC = ("ddpg", "d4pg", "td3", "td4")

OBS, ACT, ENVS = 8, 3, 8

# Enough iterations to fill every shipped buffer geometry: the flat pair
# buffer's `min_length`, a trajectory buffer's `n_step + 1` window (D4PG and
# TD4 ship `n_step: 5`) and PPO's queue width alike.
WARMUP_ITERS = 8


def agent_config(name: str, **hyperparams):
    """The shipped config for `name`, shrunk to what a CPU test can drive."""
    # The env shapes `loader.publish_env_shapes` would have written back; the
    # network blocks interpolate them.
    cfg = OmegaConf.create(
        {
            "env": {
                "parallel_envs": ENVS,
                "obs_size": OBS,
                "action_size": ACT,
                "obs_action_size": OBS + ACT,
            },
            "agent": OmegaConf.load(CONFIGS / "agent" / f"{name}.yaml"),
        }
    ).agent

    cfg.actor_config.features = [16, 16]
    for critic in _critic_blocks(cfg.critic_config):
        critic.features = [16, 16]

    for key, value in (
        ("max_length", 64 * ENVS),
        ("min_length", 4 * ENVS),
        ("sample_batch_size", 16),
        ("max_length_time_axis", 16),
        ("sample_sequence_length", 4),
        # PPO's queue hardcodes the env count rather than reading
        # `${env.parallel_envs}`.
        ("add_batch_size", ENVS),
    ):
        if key in cfg.memory_config:
            cfg.memory_config[key] = value

    # Learn as soon as the buffer is legitimately full, then every iteration:
    # what these exercise is the learning pass, not the schedule
    # (`test_update_schedule.py` covers that).
    defaults = {
        "memory_warmup": 4 * ENVS,
        "steps_between_updates": ENVS,
        "learning_steps": 2,
        "num_minibatches": 2,
        "num_action_samples": 4,
    }
    for key, value in {**defaults, **hyperparams}.items():
        if key in cfg.hyperparams:
            cfg.hyperparams[key] = value
    return cfg


def build(name: str, **hyperparams):
    """Construct the agent `name` the way `train.py` does."""
    cfg = agent_config(name, **hyperparams)
    kwargs = dict(
        env_obs_size=OBS,
        env_action_size=ACT,
        action_low=-jnp.ones(ACT, dtype=jnp.float32),
        action_high=jnp.ones(ACT, dtype=jnp.float32),
    )
    if name in DETERMINISTIC:
        kwargs["noise_config"] = OmegaConf.load(CONFIGS / "noise" / "gaussian.yaml")
    return build_agent(cfg, **kwargs)


def _critic_blocks(critic_config):
    """The blocks that actually name a network: a `TwinCritic` nests one per
    head, everything else is the network itself."""
    if critic_config._target_.endswith("TwinCritic"):
        return [critic_config.critic1, critic_config.critic2]
    return [critic_config]


def drive(agent, iterations: int, key, obs=None, drift: float = 0.0) -> int:
    """Act and buffer for `iterations` env steps; returns the env-step count.

    Deliberately the agent's public path rather than a direct buffer poke: it is
    what fills the replay buffer, advances the exploration noise counter and
    accumulates the observation statistics — most of the state a pass reads and
    a checkpoint has to carry.

    `drift` grows the observation scale by that much per step, which is what
    moves the running statistics far enough during one rollout for PPO's
    normalizer freeze to be observable (`test_ppo.py`). The default, 0, keeps
    the distribution stationary.
    """
    if obs is None:
        obs = jax.random.normal(
            jax.random.PRNGKey(1), (ENVS, OBS), dtype=jnp.float32
        )
    false = jnp.zeros((ENVS,), jnp.bool_)

    for i in range(iterations):
        step_key = jax.random.fold_in(key, i)
        act_key, obs_key, reward_key = jax.random.split(step_key, 3)
        action, _noise, extras = agent.select_action(obs, act_key)
        # `scale - 1` shifts the mean along with the spread, so both running
        # moments move; at drift 0 both terms vanish and this is a standard
        # normal, which is what every other caller gets.
        scale = 1.0 + drift * i
        timestep = Timestep(
            obs=jax.random.normal(obs_key, (ENVS, OBS), dtype=jnp.float32) * scale
            + (scale - 1.0),
            reward=jax.random.normal(reward_key, (ENVS,), dtype=jnp.float32),
            terminated=false,
            truncated=false,
            info={},
        )
        # What `Learner.buffer` does: the action and the behaviour extras come
        # from the `select_action` that chose them, since the agent keeps no
        # acting state of its own.
        agent.buffer_transitions(
            Transition(
                observation=obs,
                action=action,
                reward=timestep.reward,
                terminal=timestep.terminated,
                truncation=timestep.truncated,
                **(extras or {}),
            ),
            timestep.obs,
        )
        obs = timestep.obs

    return iterations * ENVS


def warmed(name: str, key=None, **hyperparams):
    """A built agent with a full buffer, ready for its first learning pass.

    Returns `(agent, env_steps)`; the step count is what the trainer would hand
    `pop_diagnostics`.
    """
    agent = build(name, **hyperparams)
    if key is None:
        key = jax.random.PRNGKey(0)
    steps = drive(agent, WARMUP_ITERS, key)
    return agent, steps


def snapshot(module) -> list:
    """Copies of every `nnx.Param` leaf, so a later read shows the movement.

    `nnx.Param` only: the raw module tree also carries rng keys, which are not
    comparable as arrays.
    """
    return [np.asarray(x) for x in jax.tree.leaves(nnx.state(module, nnx.Param))]


def moved(before: list, module) -> bool:
    """True if any parameter of `module` differs from its snapshot."""
    return any(
        not np.allclose(old, new)
        for old, new in zip(before, snapshot(module))
    )
