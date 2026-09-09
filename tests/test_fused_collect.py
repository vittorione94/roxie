"""The fused acting path must be the per-step loop, only faster.

Both rollouts compile the work the per-step loop used to dispatch op by op, for
the same reason and to different depths:

* `JaxRollout` scans a whole update window — select, step, buffer, score — into
  ONE dispatch, because a per-step Python loop spends 3.6 ms of host dispatch on
  an env step whose device work is 0.15 ms.
* `EnvPoolRollout` cannot scan a chunk (its physics is C++ and untraceable), so
  it compiles the two halves that sit either side of the pool step instead: ~10
  dispatches per env step become 2.

Either is only a throughput change if it lands byte-for-byte where the old loop
landed, so these run the same seed through both paths and compare everything a
chunk carries forward: the replay buffer, the observation statistics, the noise
module's decay counter, the env state, the rollout's own rng, and the episode
sums the epoch metrics are built from.

Driven through a real (tiny) env rather than stubs: what the fused paths have to
get right is the nnx split/merge of a mutated train state — across `lax.scan` on
one, across a donated pytree threaded through a Python loop on the other — and a
stub cannot exercise that.
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
from roxie.utils.rollout import (
    _EVAL_SEED,
    build_rollout,
    fusable,
    stepwise_collect,
)

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


# The `backend` group decides which rollout `build_rollout` returns: the default
# (playground) is a vmapped JAX env, `envpool_cpu` a C++ pool.
BACKENDS = ("jax", "envpool")


def _build(agent_name, backend, extra=()):
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
                *((f"dmc/backend@backend=envpool_cpu",)
                  if backend == "envpool" else ()),
                "release.task=CartpoleBalance",
                f"env.parallel_envs={NUM_ENVS}",
                # Enough to condition the observation statistics first:
                # `obs_mean_std` takes the variance as E[x^2] - E[x]^2, which
                # cancels catastrophically while the samples are few, so float
                # reassociation between the two paths would move the action.
                "agent.learning_steps=1",
                "logging.wandb.enabled=false",
                *_OVERRIDES.get(agent_name, _OFF_POLICY_OVERRIDES),
                *extra,
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
    rollout = build_rollout(
        env, test_env, agent, NUM_ENVS, nnx.Rngs(envs=0), 2,
    )
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
    # The acting key stream. `EnvPoolRollout` draws its three-way split inside
    # the compiled act step rather than on the host, so this is what pins the
    # two paths to one stream rather than merely to one distribution.
    out["rng"] = np.asarray(jax.random.key_data(rollout.rng))
    return out


def _run(agent_name, backend, fused):
    agent, rollout, state = _build(agent_name, backend)
    learner = SyncLearner(agent, jax.random.PRNGKey(0))
    assert fusable(agent, learner), (
        f"{agent_name} should qualify for the fused path"
    )
    if fused:
        state, sums = rollout.collect(state, CHUNK, learner)
    else:
        state, sums = stepwise_collect(rollout, state, CHUNK, learner)
    return _fingerprint(agent, rollout, state, sums)


@pytest.mark.parametrize("backend", BACKENDS)
@pytest.mark.parametrize("agent_name", ["td3", "sac", "mpo", "ppo"])
def test_fused_collect_matches_the_per_step_loop(agent_name, backend):
    fused = _run(agent_name, backend, True)
    stepwise = _run(agent_name, backend, False)
    assert set(fused) == set(stepwise)
    for key in sorted(fused):
        np.testing.assert_allclose(
            fused[key], stepwise[key], rtol=1e-5, atol=1e-5,
            err_msg=f"{agent_name}/{backend}: {key} diverged between the paths",
        )


@pytest.mark.parametrize("backend", BACKENDS)
def test_the_async_learner_keeps_the_per_step_loop(backend):
    """The async learner's thread owns `agent.state`, so a rollout must hand it
    transitions through `buffer` rather than compiling its own writes to the
    buffer underneath it."""
    agent, _rollout, _state = _build("td3", backend)

    class _Owning:
        owns_state = True

    assert fusable(agent, SyncLearner(agent, jax.random.PRNGKey(0)))
    assert not fusable(agent, _Owning())


def test_the_noise_schedule_advances_by_env_frames_inside_the_scan():
    """`add_noise` counts env frames, not calls, so the anneal means the same
    thing at 1 parallel env and at 4000. Inside the scan that counter lives on
    the carry — if it were dropped, exploration would never decay."""
    agent, rollout, state = _build("td3", "jax")
    learner = SyncLearner(agent, jax.random.PRNGKey(0))
    before = int(agent.noise_module.step_count.value)
    rollout.collect(state, CHUNK, learner)
    assert int(agent.noise_module.step_count.value) - before == CHUNK * NUM_ENVS


# --- the eval loop ---------------------------------------------------------
#
# `EnvPoolRollout.evaluate` compiles its action selection for the same reason
# `collect` does, and has the same obligation: score a policy exactly as the
# `agent.step` loop it replaced did. Its episodes are capped hard here — the
# comparison is per-step, so 25 of them catch what 1000 would.

_SHORT_EPISODES = ("env.max_episode_steps=25",)


def _reference_eval(agent, rollout):
    """`EnvPoolRollout.evaluate` as it was before its selection was compiled.

    Kept verbatim rather than reached through a flag: what is under test is a
    rewrite, and a reference that shares code with the thing it checks would
    stop being one.
    """
    test_env = rollout.test_environment
    max_steps = int(test_env.max_episode_steps or 1000)
    reseed = getattr(test_env, "reseed", None)
    if reseed is not None:
        reseed(_EVAL_SEED)

    state, _ = test_env.reset()
    num_tests = int(state.obs.shape[0])
    scores = np.zeros(num_tests, dtype=np.float32)
    lengths = np.zeros(num_tests, dtype=np.int32)
    dones = np.zeros(num_tests, dtype=bool)
    start_obs = np.asarray(state.obs).reshape(num_tests, -1)
    eval_key = jax.random.PRNGKey(0)

    for _ in range(max_steps):
        if np.all(dones):
            break
        actions = agent.step(state.obs, evaluate=True, key=eval_key)
        state, timestep = test_env.step(state, actions)
        active = ~dones
        scores += np.asarray(timestep.reward) * active
        lengths += active.astype(np.int32)
        dones |= np.asarray(timestep.terminated | timestep.truncated)

    return scores, lengths, start_obs


@pytest.mark.parametrize("agent_name", ["td3", "sac", "mpo", "ppo"])
def test_envpool_eval_matches_the_per_step_loop(agent_name):
    """Both runs face the same pinned pool — `reseed` at `_EVAL_SEED` is what
    makes two evals of one policy comparable at all — so a difference in the
    score is a difference in the action, not in the draw."""
    agent, rollout, _state = _build(agent_name, "envpool", _SHORT_EPISODES)

    compiled = rollout.evaluate(agent)
    reference = _reference_eval(agent, rollout)

    for name, got, want in zip(
        ("scores", "lengths", "start_obs"), compiled, reference
    ):
        np.testing.assert_allclose(
            got, want, rtol=1e-5, atol=1e-5,
            err_msg=f"{agent_name}: eval {name} diverged from the per-step loop",
        )
    assert reference[1].max() > 0, "the reference eval never stepped"


def test_eval_leaves_the_exploration_noise_alone():
    """Eval must not advance the decay counter it is not exploring with: the
    compiled step carries the noise module through its trace, and adopting a
    mutated one back would anneal exploration by an epoch's eval every epoch."""
    agent, rollout, _state = _build("td3", "envpool", _SHORT_EPISODES)
    before = int(agent.noise_module.step_count.value)
    rollout.evaluate(agent)
    assert int(agent.noise_module.step_count.value) == before
