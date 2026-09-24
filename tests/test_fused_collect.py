"""The fused chunk must be the per-step loop, only faster.

The fusion is a throughput change only if it lands byte-for-byte where the
per-step loop lands, so these run the same seed through both paths and compare
everything a chunk carries forward: the replay buffer, the observation
statistics, the noise module's decay counter, the env state, the rollout's own
rng, and the episode sums the epoch metrics are built from. Both backends, since
a C++ pool is fused through an ordered `io_callback` rather than traced.

Driven through a real (tiny) env rather than stubs: what the fused path has to
get right is a mutated train state carried across `lax.scan` and donated back,
and a stub cannot exercise that.
"""

import copy

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.agents.utils import Transition, build_agent
from roxie.environment.functional import space_size
from roxie.agents.hyperparams import OffPolicyHyperparams
from roxie.environment.loader import build_env, publish_env_shapes
from roxie.utils.rollout import ChunkSums, _EVAL_SEED, build_rollout

CHUNK = 4
NUM_ENVS = 4

# Off-policy: a warmed buffer, and an update window exactly one chunk long.
_OFF_POLICY_OVERRIDES = (
    "agent.hyperparams.memory_warmup=512",
    f"agent.hyperparams.steps_between_updates={CHUNK * NUM_ENVS}",
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
        "agent.hyperparams.num_minibatches=1",
    ),
}


# The `backend` group decides which rollout `build_rollout` returns: `mjx_cpu`
# is a vmapped JAX env, `envpool_cpu` a C++ pool. Both are named explicitly —
# every `bench_<agent>.yaml` defaults the group to `envpool_cpu`, so leaving the
# JAX cell to the default ran the pool twice and never traced the MJX path.
BACKENDS = {"jax": "mjx_cpu", "envpool": "envpool_cpu"}


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
                f"dmc/backend@backend={BACKENDS[backend]}",
                "release.task=CartpoleBalance",
                f"env.parallel_envs={NUM_ENVS}",
                # Enough to condition the observation statistics first:
                # `obs_mean_std` takes the variance as E[x^2] - E[x]^2, which
                # cancels catastrophically while the samples are few, so float
                # reassociation between the two paths would move the action.
                "agent.hyperparams.learning_steps=1",
                "logging.wandb.enabled=false",
                *_OVERRIDES.get(agent_name, _OFF_POLICY_OVERRIDES),
                *extra,
            ],
        )
    env, test_env, _ = build_env(
        cfg.env, mode="train", num_envs=NUM_ENVS, test_episodes=2,
    )
    act_space = env.single_action_space
    # What `train.py` does between the env and the agent: the network blocks
    # interpolate `${env.obs_size}` and friends, which nothing knows until the
    # env exists.
    publish_env_shapes(
        cfg.env, space_size(env.single_observation_space), space_size(act_space),
    )
    kwargs = dict(
        env_obs_size=space_size(env.single_observation_space),
        env_action_size=space_size(act_space),
        action_low=jnp.asarray(act_space.low, jnp.float32),
        action_high=jnp.asarray(act_space.high, jnp.float32),
    )
    if "noise" in cfg:
        kwargs["noise_config"] = cfg.noise
    agent = build_agent(cfg.agent, **kwargs)
    rollout, rstate = build_rollout(
        env, test_env, NUM_ENVS, nnx.Rngs(envs=0).envs(),
    )
    # On-policy agents have no warmup to run (and no replay to fill) — the
    # type is what says so, exactly as `Trainer._precompile_update` asks it.
    warmup_steps = (
        agent.hp.memory_warmup // NUM_ENVS
        if isinstance(agent.hp, OffPolicyHyperparams)
        else 0
    )
    if warmup_steps > 0:
        rstate, _ = rollout.warmup(agent, rstate, warmup_steps)
    return agent, rollout, rstate


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


def _fingerprint(agent, rstate, sums):
    """Everything a chunk carries forward, flattened to comparable arrays."""
    out = _buffer_leaves(agent.state.buffer_state)
    out.update({
        f"obs_stats/{i}": _as_numpy(leaf)
        for i, leaf in enumerate(jax.tree.leaves(agent.state.obs_stats))
    })
    out.update({
        f"env/{i}": _as_numpy(leaf)
        for i, leaf in enumerate(jax.tree.leaves(rstate.env))
    })
    out["scores"] = np.asarray(rstate.scores)
    out["lengths"] = np.asarray(rstate.lengths)
    for name in ("ret", "ret_sq", "length", "length_sq", "count", "noise"):
        out[f"sums/{name}"] = np.asarray(getattr(sums, name))
    noise_module = getattr(agent, "noise_module", None)
    if noise_module is not None:
        out["noise_steps"] = np.asarray(noise_module.step_count.get_value())
    # The acting key stream: both paths must consume it identically, not merely
    # draw from one distribution.
    out["rng"] = np.asarray(jax.random.key_data(rstate.rng))
    return out


def _stepwise_collect(rollout, agent, rstate, n_steps):
    """`collect` as a Python loop, one host dispatch per env step.

    The reference the fused `collect` replaced. Kept here verbatim rather than
    shipped alongside the thing it checks: roxie has no non-fusable trainable
    agent left, so this is a test fixture, and a reference that shared code with
    its subject would stop being one.
    """
    zero = jnp.zeros((), jnp.float32)
    sums = {key: zero for key in
            ("ret", "ret_sq", "len", "len_sq", "count", "noise")}
    sums["metrics"] = {k: zero for k in rollout.metric_keys}
    train_state = agent.state
    noise = getattr(agent, "noise_module", None)
    # Acting is pinned for the whole chunk on BOTH paths: the fused one folds
    # the chunk's own observations into the statistics only once it is over, so
    # a reference that let them advance per step would score every step after
    # the first under a normalizer the real loop never used.
    pin = agent.freeze_acting_norm()
    if pin is None:
        pin = train_state.obs_stats
    scores, lengths = rstate.scores, rstate.lengths
    seen, landed = [], []
    for _ in range(n_steps):
        # Same three-way split as the fused body: both paths must consume the
        # same stream.
        rng, act_key, step_key = jax.random.split(rstate.rng, 3)
        rstate = rstate.replace(rng=rng)
        action, applied_noise, extras = agent.select_action(
            rstate.obs, act_key, evaluate=False,
            actor=train_state.actor, critic=train_state.critic,
            noise_module=noise, obs_stats=pin,
        )
        rstate, prev_obs, timestep = rollout.step(rstate, action, step_key)
        agent.buffer_transitions(
            Transition(
                observation=prev_obs,
                action=action,
                reward=timestep.reward,
                terminal=timestep.terminated,
                truncation=timestep.truncated,
                **(extras or {}),
            ),
            timestep.obs,
            state=train_state,
            update_stats=False,
        )
        seen.append(prev_obs)
        landed.append(timestep.obs)

        done = timestep.terminated | timestep.truncated
        scores = scores + timestep.reward
        lengths = lengths + 1
        done_f = done.astype(jnp.float32)
        ep_ret = scores * done_f
        ep_len = lengths.astype(jnp.float32) * done_f
        sums["ret"] = sums["ret"] + jnp.sum(ep_ret)
        sums["ret_sq"] = sums["ret_sq"] + jnp.sum(ep_ret ** 2)
        sums["len"] = sums["len"] + jnp.sum(ep_len)
        sums["len_sq"] = sums["len_sq"] + jnp.sum(ep_len ** 2)
        sums["count"] = sums["count"] + jnp.sum(done_f)
        sums["noise"] = sums["noise"] + jnp.mean(jnp.abs(applied_noise))
        step_metrics = timestep.info.get("metrics", {})
        for key in rollout.metric_keys:
            sums["metrics"][key] = (
                sums["metrics"][key] + jnp.mean(step_metrics[key])
            )
        scores = jnp.where(done, 0.0, scores)
        lengths = jnp.where(done, 0, lengths)

    agent.absorb_obs_stats(
        jnp.stack(seen), jnp.stack(landed), state=train_state,
    )
    # Only the CONTAINER is shared with the subject; every reduction above is
    # this reference's own.
    return rstate.replace(scores=scores, lengths=lengths), ChunkSums(
        ret=sums["ret"], ret_sq=sums["ret_sq"], length=sums["len"],
        length_sq=sums["len_sq"], count=sums["count"], noise=sums["noise"],
        metrics=sums["metrics"],
    )


def _fused_run(agent, rollout, rstate):
    agent.state, noise, rstate, sums = rollout.collect(
        agent.state, getattr(agent, "noise_module", None), rstate,
        agent.freeze_acting_norm(), agent=agent, n_steps=CHUNK,
    )
    if noise is not None:
        agent.noise_module = noise
    return _fingerprint(agent, rstate, sums)


def _stepwise_run(agent, rollout, rstate):
    rstate, sums = _stepwise_collect(rollout, agent, rstate, CHUNK)
    return _fingerprint(agent, rstate, sums)


def _both_paths(agent_name, backend):
    """One chunk through each path, from the same starting point.

    A traced JAX env is pure — all of its state is `rstate`, an immutable pytree
    both paths are handed — so one build serves both and the MJX physics
    compiles once rather than twice. A pool cannot be shared that way: it steps
    in C++ and owns its episode state, so the second path would continue where
    the first left off instead of repeating it. `_build` is deterministic (fixed
    seeds throughout), so building twice there gives two identical agents.
    """
    fused_side = _build(agent_name, backend)
    if backend == "jax":
        agent, rollout, rstate = fused_side
        stepwise_side = (copy.deepcopy(agent), rollout, rstate)
    else:
        stepwise_side = _build(agent_name, backend)
    return _fused_run(*fused_side), _stepwise_run(*stepwise_side)


# The combinations that are actually distinct. What varies with the AGENT is
# what rides in the scan carry — a noise module, obs statistics, PPO's stored
# behaviour extras — and that is the same whichever `advance` runs underneath.
# What varies with the BACKEND is `advance` alone. So every agent goes through
# the pool, and one goes through the traced JAX env, whose MJX physics costs a
# ~10s compile per case. TD3 is that one: its carry is the fullest of the four.
CASES = [
    ("td3", "jax"),
    ("td3", "envpool"),
    ("sac", "envpool"),
    ("mpo", "envpool"),
    ("ppo", "envpool"),
]


@pytest.mark.parametrize("agent_name,backend", CASES)
def test_fused_collect_matches_the_per_step_loop(agent_name, backend):
    fused, stepwise = _both_paths(agent_name, backend)
    assert set(fused) == set(stepwise)
    for key in sorted(fused):
        np.testing.assert_allclose(
            fused[key], stepwise[key], rtol=1e-5, atol=1e-5,
            err_msg=f"{agent_name}/{backend}: {key} diverged between the paths",
        )


def test_the_noise_schedule_advances_by_env_frames_inside_the_scan():
    """`add_noise` counts env frames, not calls, so the anneal means the same
    thing at 1 parallel env and at 4000. Inside the scan that counter lives on
    the carry — if it were dropped, exploration would never decay.

    One backend: `collect` is the SAME function for both (the seam is
    `advance`), which `test_trainer_bookkeeping.py` pins separately.
    """
    agent, rollout, rstate = _build("td3", "envpool")
    before = int(agent.noise_module.step_count.get_value())
    agent.state, agent.noise_module, rstate, _ = rollout.collect(
        agent.state, agent.noise_module, rstate, agent.freeze_acting_norm(),
        agent=agent, n_steps=CHUNK,
    )
    assert (
        int(agent.noise_module.step_count.get_value()) - before
        == CHUNK * NUM_ENVS
    )


# --- the eval loop ---------------------------------------------------------
#
# `evaluate` is compiled for the same reason `collect` is, and carries the same
# obligation: score a policy exactly as the `agent.select_action` loop it
# replaced did. Its episodes are capped hard here — the comparison is per-step,
# so 25 of them catch what 1000 would.

_SHORT_EPISODES = ("env.max_episode_steps=25",)


def _eval_stats(agent):
    """The statistics an eval scores against, exactly as `Trainer` picks them:
    live for most agents, the rollout's pin for an on-policy one."""
    pin = agent.freeze_acting_norm()
    return agent.state.obs_stats if pin is None else pin


def _reference_eval(agent, rollout):
    """The eval loop as it was before it was compiled, scoring the agent
    through a bare `select_action(evaluate=True)`.

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
        actions, _, _ = agent.select_action(
            state.obs, eval_key, evaluate=True,
        )
        state, timestep = test_env.step(state, actions)
        active = ~dones
        scores += np.asarray(timestep.reward) * active
        lengths += active.astype(np.int32)
        dones |= np.asarray(timestep.terminated | timestep.truncated)

    return scores, lengths, start_obs


@pytest.mark.parametrize("agent_name", ["td3", "ppo"])
def test_envpool_eval_matches_the_per_step_loop(agent_name):
    """Both runs face the same pinned pool — `reseed` at `_EVAL_SEED` is what
    makes two evals of one policy comparable at all — so a difference in the
    score is a difference in the action, not in the draw."""
    agent, rollout, _rstate = _build(agent_name, "envpool", _SHORT_EPISODES)

    compiled = rollout.evaluate(
        agent.state.actor, _eval_stats(agent), rollout.reset_test(),
        agent=agent,
    )
    reference = _reference_eval(agent, rollout)

    for name, got, want in zip(
        ("scores", "lengths", "start_obs"), compiled, reference
    ):
        np.testing.assert_allclose(
            got, want, rtol=1e-5, atol=1e-5,
            err_msg=f"{agent_name}: eval {name} diverged from the per-step loop",
        )
    assert reference[1].max() > 0, "the reference eval never stepped"


def test_repeated_evals_of_one_policy_agree():
    """Two evals in a row, which is what every epoch does.

    `reset_test` reseeds and `reseed` REBUILDS the pool, so this is also the
    path that has to survive a pool swap between two compiled evals. Two evals
    of the SAME weights must return the same numbers — that is the whole reason
    `reseed` pins the draw, and a drift here means an epoch's `test/score` is
    reporting the draw rather than the policy.
    """
    agent, rollout, _rstate = _build("td3", "envpool", _SHORT_EPISODES)
    stats = _eval_stats(agent)

    first = rollout.evaluate(
        agent.state.actor, stats, rollout.reset_test(), agent=agent,
    )
    second = rollout.evaluate(
        agent.state.actor, stats, rollout.reset_test(), agent=agent,
    )
    for name, a, b in zip(("scores", "lengths", "start_obs"), first, second):
        np.testing.assert_allclose(
            a, b, rtol=1e-5, atol=1e-5,
            err_msg=f"eval {name} changed between two evals of the same weights",
        )
    assert np.asarray(first[1]).max() > 0, "the eval never stepped"


def test_eval_leaves_the_exploration_noise_alone():
    """Eval must not advance the decay counter it is not exploring with, or an
    epoch's eval anneals exploration every epoch. `evaluate`, like `collect`, is
    shared by both backends."""
    agent, rollout, _rstate = _build("td3", "envpool", _SHORT_EPISODES)
    before = int(agent.noise_module.step_count.get_value())
    rollout.evaluate(
        agent.state.actor, _eval_stats(agent), rollout.reset_test(),
        agent=agent,
    )
    assert int(agent.noise_module.step_count.get_value()) == before
