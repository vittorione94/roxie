"""The fused acting burst must be the per-step loop, only faster.

`JaxRollout.collect` scans a whole update window — select, step, buffer, score —
into one dispatch, because a per-step Python loop spends 3.6 ms of host dispatch
on an env step whose device work is 0.15 ms. That is only a throughput change if
it lands byte-for-byte where the old loop landed, so these run the same seed
through both paths and compare everything a run carries forward: the replay
buffer, the observation statistics, the noise module's decay counter, the env
state, and the episode sums the epoch metrics are built from.

Driven through a real (tiny) env rather than stubs: what the fused path has to
get right is the nnx split/merge of a mutated train state across `lax.scan`, and
a stub cannot exercise that.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.agents.utils import build_agent
from roxie.environment.functional import space_size
from roxie.environment.loader import build_env
from roxie.utils.learner import SyncLearner
from roxie.utils.rollout import JaxRollout, stepwise_collect

CHUNK = 4
NUM_ENVS = 4

# Off-policy: a warmed buffer, and an update window exactly one chunk long.
_OFF_POLICY_OVERRIDES = (
    "agent.memory_warmup=512",
    f"agent.steps_between_updates={CHUNK * NUM_ENVS}",
    "agent.memory_config.max_length=256",
    "agent.memory_config.min_length=8",
    "agent.memory_config.sample_batch_size=8",
)

# PPO has no warmup and no replay: its queue IS the update window, so the chunk
# comes from `sample_sequence_length`, which `steps_between_updates` derives
# from. `num_minibatches` must divide `add_batch_size`.
_OVERRIDES = {
    "ppo": (
        f"agent.memory_config.sample_sequence_length={CHUNK}",
        "agent.memory_config.max_length_time_axis=16",
        "agent.num_minibatches=1",
    ),
}


def _build(agent_name):
    from hydra import compose, initialize_config_dir
    from pathlib import Path

    from roxie.environment import suites
    from roxie.utils import hydra_searchpath

    hydra_searchpath.register()
    suites.register_resolvers()
    configs = Path(__file__).resolve().parents[1] / "roxie" / "configs"
    with initialize_config_dir(config_dir=str(configs), version_base=None):
        cfg = compose(
            config_name=f"dmc/bench_{agent_name}",
            overrides=[
                "release.task=CartpoleBalance",
                f"env.parallel_envs={NUM_ENVS}",
                # Enough to condition the observation statistics first:
                # `obs_mean_std` takes the variance as E[x^2] - E[x]^2, which
                # cancels catastrophically while the samples are few, so float
                # reassociation between the two paths would move the action.
                "agent.learning_steps=1",
                "logging.wandb.enabled=false",
                *_OVERRIDES.get(agent_name, _OFF_POLICY_OVERRIDES),
            ],
        )
    env, test_env, _ = build_env(
        cfg.env, mode="train", num_envs=NUM_ENVS, test_episodes=2,
    )
    act_space = env.single_action_space
    kwargs = dict(
        env_obs_size=space_size(env.single_observation_space),
        env_action_size=space_size(act_space),
        action_low=jnp.asarray(act_space.low, jnp.float32),
        action_high=jnp.asarray(act_space.high, jnp.float32),
    )
    if "noise" in cfg:
        kwargs["noise_config"] = cfg.noise
    agent = build_agent(cfg.agent, **kwargs)
    rollout = JaxRollout(env, test_env, agent, NUM_ENVS, nnx.Rngs(envs=0), 2)
    state = rollout.prepare()
    # On-policy agents have no warmup to run (and no replay to fill).
    warmup_steps = getattr(agent, "memory_warmup", 0) // NUM_ENVS
    if warmup_steps > 0:
        state, _ = rollout.warmup(agent, warmup_steps, state)
    return agent, rollout, state


def _as_numpy(leaf):
    """`np.asarray` refuses a typed PRNG key array, and env states carry them."""
    if jax.dtypes.issubdtype(getattr(leaf, "dtype", None), jax.dtypes.prng_key):
        return np.asarray(jax.random.key_data(leaf))
    return np.asarray(leaf)


# Bookkeeping scalars a flashbax buffer/queue state carries next to its
# experience. Compared by name so a slot that stops advancing is caught.
_BUFFER_INDEX_FIELDS = ("is_full", "current_index", "read_index", "write_index")


def _buffer_leaves(buffer_state):
    """The buffer contents the agent can actually read back.

    flashbax initializes a buffer with `jnp.empty_like` (see
    `trajectory_buffer.init`), so slots no `add` has touched hold whatever the
    allocator handed out — comparing them across two runs compares uninitialized
    memory, not behaviour. The off-policy agents never notice: warmup writes
    their whole buffer before the comparison starts. PPO's queue is only
    `write_index` steps deep, so it is sliced to the window `sample` would
    return, which is the only part its update can ever see.
    """
    experience = buffer_state.experience
    write_index = getattr(buffer_state, "write_index", None)
    if write_index is not None:
        experience = jax.tree.map(lambda x: x[:, : int(write_index)], experience)

    out = {
        f"buffer/{i}": _as_numpy(leaf)
        for i, leaf in enumerate(jax.tree.leaves(experience))
    }
    for name in _BUFFER_INDEX_FIELDS:
        value = getattr(buffer_state, name, None)
        if value is not None:
            out[f"buffer/{name}"] = _as_numpy(value)
    return out


def _fingerprint(agent, rollout, state, sums):
    """Everything a chunk carries forward, flattened to comparable arrays."""
    out = _buffer_leaves(agent.state.buffer_state)
    out.update({
        f"obs_stats/{i}": _as_numpy(leaf)
        for i, leaf in enumerate(jax.tree.leaves(agent.state.obs_stats))
    })
    out.update({
        f"env/{i}": _as_numpy(leaf) for i, leaf in enumerate(jax.tree.leaves(state))
    })
    out["scores"] = np.asarray(rollout.scores)
    out["lengths"] = np.asarray(rollout.lengths)
    for key in ("ret", "ret_sq", "len", "len_sq", "count", "noise"):
        out[f"sums/{key}"] = np.asarray(sums[key])
    noise_module = getattr(agent, "noise_module", None)
    if noise_module is not None:
        out["noise_steps"] = np.asarray(noise_module.step_count.value)
    return out


def _run(agent_name, fused):
    agent, rollout, state = _build(agent_name)
    assert rollout._fusable(), f"{agent_name} should qualify for the fused path"
    learner = SyncLearner(agent, jax.random.PRNGKey(0))
    if fused:
        state, sums = rollout.collect(state, CHUNK, learner)
    else:
        state, sums = stepwise_collect(rollout, state, CHUNK, learner)
    return _fingerprint(agent, rollout, state, sums)


@pytest.mark.parametrize("agent_name", ["td3", "sac", "mpo", "ppo"])
def test_fused_collect_matches_the_per_step_loop(agent_name):
    fused, stepwise = _run(agent_name, True), _run(agent_name, False)
    assert set(fused) == set(stepwise)
    for key in sorted(fused):
        np.testing.assert_allclose(
            fused[key], stepwise[key], rtol=1e-5, atol=1e-5,
            err_msg=f"{agent_name}: {key} diverged between the two paths",
        )


def test_the_noise_schedule_advances_by_env_frames_inside_the_scan():
    """`add_noise` counts env frames, not calls, so the anneal means the same
    thing at 1 parallel env and at 4000. Inside the scan that counter lives on
    the carry — if it were dropped, exploration would never decay."""
    agent, rollout, state = _build("td3")
    learner = SyncLearner(agent, jax.random.PRNGKey(0))
    before = int(agent.noise_module.step_count.value)
    rollout.collect(state, CHUNK, learner)
    assert int(agent.noise_module.step_count.value) - before == CHUNK * NUM_ENVS
