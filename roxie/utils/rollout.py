"""Environment execution and rollout collection for the training loop."""

import functools
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from roxie.agents.utils import Transition
from roxie.environment.functional import space_size
from roxie.environment.vector import UNCAPPED_EVAL_STEPS
from roxie.utils.precision import FLOAT

_EVAL_SEED = 12345


def timed(fn, *args, label="", **kwargs):
    """Run `fn`, blocking on its output, and print how long the compile took."""
    print(f"Compiling {label}...", flush=True)
    t0 = time.time()
    out = fn(*args, **kwargs)
    jax.block_until_ready(out)
    print(f"  {time.time() - t0:.1f}s", flush=True)
    return out


@struct.dataclass
class RolloutState:
    """Everything one chunk of acting advances, as a single pytree."""

    env: Any
    rng: Any
    scores: Any
    lengths: Any
    params: Any = None
    reset_pool: Any = None

    @property
    def obs(self):
        """The observation the next action is selected from."""
        return self.env.obs


def fresh_tally(num_envs):
    """Zeroed in-flight per-env episode counters, as `(scores, lengths)`."""
    return (
        jnp.zeros(num_envs, dtype=FLOAT),
        jnp.zeros(num_envs, dtype=jnp.int32),
    )


def finished_episodes(scores, lengths, done):
    """This step's completed episodes, masked: `(ep_ret, ep_len, done_f)`."""
    done_f = done.astype(jnp.float32)
    return scores * done_f, lengths.astype(jnp.float32) * done_f, done_f


@struct.dataclass
class ChunkSums:
    """One chunk's scalars: the episode accumulators and the per-step means.

    Attributes:
        task_ret: `(K,)` finished-episode returns summed per task; None for a
            single-task env.
        task_count: `(K,)` finished episodes per task; None likewise.
    """

    ret: Any
    ret_sq: Any
    length: Any
    length_sq: Any
    count: Any
    noise: Any
    metrics: dict
    task_ret: Any = None
    task_count: Any = None

    @classmethod
    def zeros(cls, metric_keys, num_tasks=0):
        """An empty accumulator, keyed for the metrics this env reports."""
        zero = jnp.zeros((), FLOAT)
        per_task = jnp.zeros((num_tasks,), FLOAT) if num_tasks else None
        return cls(
            ret=zero, ret_sq=zero, length=zero, length_sq=zero, count=zero,
            noise=zero, metrics={key: zero for key in metric_keys},
            task_ret=per_task, task_count=per_task,
        )

    @classmethod
    def from_chunk(cls, ep_ret, ep_len, done_f, noise, metrics, n_steps,
                   task_hot=None):
        """Reduces one chunk, in a single pass per key.

        Every argument arrives stacked over the chunk's time axis, `(T, ...)`,
        but `task_hot`: the static `(N, K)` world-to-task one-hot of a
        multi-task env, or None.

        Returns:
            The chunk's `ChunkSums`.
        """
        def per_step_mean(x):
            return jnp.mean(x.reshape(n_steps, -1), axis=1)

        def per_task(x):
            # HIGHEST: a sum of returns, not an activation.
            return jnp.einsum(
                "n,nk->k", jnp.sum(x, axis=0), task_hot,
                precision=jax.lax.Precision.HIGHEST,
            )

        return cls(
            ret=jnp.sum(ep_ret),
            ret_sq=jnp.sum(jnp.square(ep_ret)),
            length=jnp.sum(ep_len),
            length_sq=jnp.sum(jnp.square(ep_len)),
            count=jnp.sum(done_f),
            noise=jnp.sum(per_step_mean(jnp.abs(noise))),
            metrics={key: jnp.sum(per_step_mean(value))
                     for key, value in metrics.items()},
            task_ret=None if task_hot is None else per_task(ep_ret),
            task_count=None if task_hot is None else per_task(done_f),
        )

    def __add__(self, other):
        """Fold another chunk in, leaf by leaf. Both carry the same metric keys
        — `zeros` is built from the same `metric_keys` the chunk reports."""
        return jax.tree.map(jnp.add, self, other)


def task_one_hot(environment):
    """The static `(N, K)` world-to-task one-hot of a multi-task env, or None."""
    task_ids = getattr(environment, "task_ids", None)
    if task_ids is None:
        return None
    return np.eye(len(environment.task_names), dtype=np.float32)[task_ids]


def chunk_step(carry, _, *, environment, reset_pool, agent, acting_stats,
               metric_keys):
    """Runs one environment step of the fused chunk: act, advance, buffer, tally.

    Args:
        carry: `(train_state, noise_module, state, rng, params, scores,
            lengths)` — only what a step must read back from the one before it.
        acting_stats: The normalizer, constant for the chunk.

    Returns:
        The scan pair `(carry, stacked)`.
    """
    train_state, noise_module, state, rng, params, scores, lengths = carry

    rng, act_key, step_key = jax.random.split(rng, 3)
    prev_obs = state.obs
    action, applied_noise, extras = agent.select_action(
        prev_obs, act_key, evaluate=False,
        actor=train_state.actor, critic=train_state.critic,
        noise_module=noise_module, obs_stats=acting_stats,
    )
    state, timestep, params = advance(environment, state, action, step_key,
                                      reset_pool, params)

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

    done = timestep.terminated | timestep.truncated
    scores = scores + timestep.reward
    lengths = lengths + 1
    ep_ret, ep_len, done_f = finished_episodes(scores, lengths, done)
    scores = jnp.where(done, 0.0, scores)
    lengths = jnp.where(done, 0, lengths)

    step_metrics = timestep.info.get("metrics", {})
    return (
        train_state, noise_module, state, rng, params, scores, lengths,
    ), (
        prev_obs, timestep.obs, ep_ret, ep_len, done_f, applied_noise,
        {key: step_metrics[key] for key in metric_keys},
    )


@functools.partial(jax.jit, static_argnames=("environment", "num_envs"))
def reset_envs(key, params=None, *, environment, num_envs=None):
    """Fresh episodes for `num_envs` worlds (default: the env's own count).

    Parameterized by WHICH env, so it is a function rather than a method: the
    train driver, the auto-reset pool and the eval driver all reset through it.
    `params` is traced, so an env that refreshes its own parameters each epoch
    does not retrigger a compile.
    """
    return environment.reset(key, params, num_envs=num_envs)[0]


def advance(environment, state, action, key, reset_pool, params):
    """Steps the envs and folds the outcome into their `params`.

    `FuncEnv.observe_params` is a no-op by default, so an env that does not
    adapt needs no branch.

    Returns:
        A tuple of `(state, timestep, params)`.
    """
    state, timestep = environment.step(state, action, key, reset_pool, params)
    # `terminated`, not `done`: a non-failure cutoff must not read as a failure.
    params = environment.func_env.observe_params(
        params, timestep.info, timestep.terminated,
    )
    return state, timestep, params


@functools.partial(jax.jit, static_argnames=("environment",))
def step_envs(rstate, actions, key, *, environment):
    """Advances the environments one step, with the env's own `params` update.

    Returns:
        A tuple of `(rstate, prev_obs, timestep)`.
    """
    state, timestep, params = advance(
        environment, rstate.env, actions, key, rstate.reset_pool, rstate.params,
    )
    return rstate.replace(env=state, params=params), rstate.obs, timestep


@functools.partial(
    jax.jit,
    static_argnames=("environment", "agent", "n_steps", "metric_keys",
                     "unroll"),
    donate_argnums=(0, 1),
)
def collect_chunk(train_state, noise_module, rstate, frozen_stats, *,
                  environment, agent, n_steps, metric_keys, unroll):
    """Fuses acting, physics, buffering and scoring into a single JAX scan.

    `chunk_step` holds the per-step work and the carry; this holds what the
    chunk keeps constant, and reduces what the scan stacked.

    Args:
        train_state: The agent's weights, optimizer state and replay buffer.
            Donated.
        noise_module: The exploration noise module, or None. Donated.
        rstate: The environment-side state carry (`RolloutState`).
        frozen_stats: Pinned normalizer statistics for on-policy rollouts, or
            None to pin to the statistics as they stood when the chunk opened.
        agent: The static agent instance; only its code may be used in a trace.
        unroll: Loop unroll count for XLA optimization.

    Returns:
        A tuple of `(train_state, noise_module, rstate, chunk_sums)`.
    """
    acting_stats = (train_state.obs_stats if frozen_stats is None
                    else frozen_stats)
    body = functools.partial(
        chunk_step, environment=environment, reset_pool=rstate.reset_pool,
        agent=agent, acting_stats=acting_stats, metric_keys=metric_keys,
    )

    init = (
        train_state, noise_module, rstate.env, rstate.rng, rstate.params,
        rstate.scores, rstate.lengths,
    )
    (train_state, noise_module, state, rng, params, scores,
     lengths), stacked = jax.lax.scan(
        body, init, None, length=n_steps, unroll=unroll,
    )
    prev_obs, next_obs, ep_ret, ep_len, done_f, noise, metrics = stacked

    agent.absorb_obs_stats(prev_obs, next_obs, state=train_state)
    return train_state, noise_module, rstate.replace(
        env=state, rng=rng, params=params, scores=scores, lengths=lengths,
    ), ChunkSums.from_chunk(
        ep_ret, ep_len, done_f, noise, metrics, n_steps,
        task_hot=task_one_hot(environment),
    )


@functools.partial(
    jax.jit,
    static_argnames=("environment", "num_envs", "action_size", "iters"),
)
def roll_random_actions(rstate, *, environment, num_envs, action_size, iters):
    """Rolls uniform-action warmup steps, as one `(T, B, ...)` block.

    Returns:
        A tuple of `(rstate, transitions, next_obs)`, the last two stacked over
        the time axis.
    """
    space = environment.single_action_space
    low = jnp.asarray(space.low, jnp.float32)
    high = jnp.asarray(space.high, jnp.float32)

    def body(carry, _):
        state, rng = carry
        rng, act_key, step_key = jax.random.split(rng, 3)
        u = jax.random.uniform(
            act_key, (num_envs, action_size), dtype=FLOAT
        )
        actions = low + (high - low) * u
        prev_obs = state.obs
        state, timestep = environment.step(
            state, actions, step_key, rstate.reset_pool, rstate.params,
        )
        return (state, rng), (
            Transition(
                observation=prev_obs,
                action=actions,
                reward=timestep.reward,
                terminal=timestep.terminated,
                truncation=timestep.truncated,  # pruned if unused
            ),
            timestep.obs,
        )

    (state, rng), (transitions, next_obs) = jax.lax.scan(
        body, (rstate.env, rstate.rng), None, length=iters,
    )
    return rstate.replace(env=state, rng=rng), transitions, next_obs


@functools.partial(
    jax.jit,
    static_argnames=("environment", "agent", "max_steps"),
)
def evaluate_policy(actor, obs_stats, state, *, environment, agent, max_steps):
    """Rolls the eval environments to their own ends, scoring as they go.

    Args:
        actor: The policy weights. Traced, not read off the static `agent`:
            baked in as a constant, every epoch after the first would score the
            first epoch's weights.
        obs_stats: The normalizer acting ran under. Traced, for the same reason.
        state: The eval envs from `reset_test`, so the draw is pinned.

    Returns:
        A tuple of `(scores, lengths, start_obs)`, per eval environment.
    """
    # A `while_loop`, so the termination check stays on device.
    num_tests = state.obs.shape[0]
    start_obs = state.obs.reshape(num_tests, -1)
    # Fixed, like the pinned draw: two evals of the same weights must agree.
    eval_key = jax.random.PRNGKey(0)

    def cond(carry):
        i, _state, dones, _scores, _lengths = carry
        return (i < max_steps) & (~jnp.all(dones))

    def body(carry):
        i, state, dones, scores, lengths = carry
        action = agent.eval_action(state.obs, actor, obs_stats)
        # No reset pool, so no auto-reset: finished worlds are masked instead.
        state, timestep = environment.step(state, action, eval_key, None, None)
        not_done = ~dones
        scores = scores + timestep.reward * not_done.astype(jnp.float32)
        lengths = lengths + not_done.astype(jnp.int32)
        dones = jnp.logical_or(
            dones,
            jnp.logical_or(timestep.terminated, timestep.truncated),
        )
        return (i + 1, state, dones, scores, lengths)

    init = (
        jnp.int32(0),
        state,
        jnp.zeros((num_tests,), dtype=bool),
        jnp.zeros((num_tests,), dtype=jnp.float32),
        jnp.zeros((num_tests,), dtype=jnp.int32),
    )
    _, _, _, scores, lengths = jax.lax.while_loop(cond, body, init)
    return scores, lengths, start_obs


class Rollout:
    """The training loop's env half: a dispatcher onto the functions above.

    Instances are frozen at construction: everything below feeds a static jit
    argument, so `_probe` settles it all before `__init__` runs and `build_rollout`
    is the only way to get one.

    Attributes:
        scan_unroll: How many chunk steps XLA sees as straight-line code per
            `lax.scan` trip. It buys fusion and register reuse ACROSS env steps
            at the price of a proportionally larger graph to compile.
        num_envs: The env width.
        metric_keys: What the env reports under `info["metrics"]`.
        task_names: A multi-task env's tasks, in one-hot order; empty if single.
    """

    scan_unroll = 2

    _frozen = False

    def __init__(self, environment, test_environment, num_envs, metric_keys):
        self.environment = environment
        self.test_environment = test_environment
        self.num_envs = int(num_envs)
        self.action_size = space_size(environment.single_action_space)
        self.metric_keys = tuple(metric_keys)
        self.task_names = tuple(getattr(environment, "task_names", ()))
        self._frozen = True

    def __setattr__(self, name, value):
        # `metric_keys` and `num_envs` feed static jit arguments: a later write
        # would silently reuse a stale compilation.
        if self._frozen:
            raise AttributeError(
                f"{type(self).__name__} is frozen: cannot set {name!r}. "
                f"Anything that varies belongs in `RolloutState`."
            )
        object.__setattr__(self, name, value)

    @classmethod
    def _probe(cls, environment, num_envs, rng):
        """Compiles every env program the loop will dispatch, then resets.

        Doing it upfront keeps first-iteration compiles out of the throughput
        print, and settles `metric_keys` from a real step: a vmapped env's
        `transition_info` is only known by running it.
        """
        num_envs = int(num_envs)
        rng, reset_key, pool_key = jax.random.split(rng, 3)
        # Traced, so refreshing it each epoch does not recompile.
        params = environment.func_env.init_params()

        def reset(n, key):
            return reset_envs(key, params, environment=environment, num_envs=n)

        state = timed(reset, num_envs, reset_key, label="reset")
        # Sized like the envs only so one reset program serves both.
        reset_pool = timed(reset, num_envs, pool_key, label="reset pool")
        scores, lengths = fresh_tally(num_envs)
        rstate = RolloutState(
            env=state, rng=rng, scores=scores, lengths=lengths, params=params,
            reset_pool=reset_pool,
        )

        action_size = space_size(environment.single_action_space)
        dummy_actions = jnp.zeros((num_envs, action_size), dtype=FLOAT)
        _, _, dummy_timestep = timed(
            step_envs, rstate, dummy_actions, rng, label="train step",
            environment=environment,
        )
        metric_keys = tuple(dummy_timestep.info.get("metrics", {}))

        # The compile call stepped this state; training must not start from it.
        rng, key = jax.random.split(rng)
        return rstate.replace(env=reset(num_envs, key), rng=rng), metric_keys

    def reset_test(self):
        return reset_envs(
            jax.random.PRNGKey(_EVAL_SEED), environment=self.test_environment,
        )

    def epoch_refresh(self, rstate):
        """Let the env refresh itself, then regenerate the reset pool (fresh
        random starts for auto-reset) from the refreshed `params`.

        An env that reports `invalidated` has changed something its in-progress
        episodes referred to, so the live envs are reset and the caller is told
        to drop their part-scored episodes.
        """
        params, invalidated = self.environment.func_env.epoch_refresh(
            rstate.params)
        reset = functools.partial(
            reset_envs, params=params, environment=self.environment,
            num_envs=self.num_envs,
        )

        rng, pool_key = jax.random.split(rstate.rng)
        rstate = rstate.replace(
            params=params, rng=rng, reset_pool=reset(pool_key))
        if invalidated:
            rng, reset_key = jax.random.split(rstate.rng)
            rstate = rstate.replace(rng=rng, env=reset(reset_key))
        return rstate, invalidated

    def reset_tally(self, rstate):
        """`rstate` with the in-flight per-env episode counters dropped."""
        scores, lengths = fresh_tally(self.num_envs)
        return rstate.replace(scores=scores, lengths=lengths)

    def step(self, rstate, actions, key):
        """Advances the envs one step. See `step_envs`."""
        return step_envs(rstate, actions, key, environment=self.environment)

    def collect(self, train_state, noise_module, rstate, frozen_stats, *,
                agent, n_steps):
        """Collects one chunk. See `collect_chunk`, which this only binds."""
        return collect_chunk(
            train_state, noise_module, rstate, frozen_stats,
            environment=self.environment, agent=agent, n_steps=n_steps,
            metric_keys=self.metric_keys,
            # A chunk shorter than the unroll would otherwise be traced as a
            # peeled remainder with no scan left around it.
            unroll=max(1, min(self.scan_unroll, n_steps)),
        )

    def evaluate(self, actor, obs_stats, state, *, agent):
        """Runs the held-out eval. See `evaluate_policy`."""
        return evaluate_policy(
            actor, obs_stats, state, environment=self.test_environment,
            agent=agent,
            max_steps=int(
                self.test_environment.max_episode_steps or UNCAPPED_EVAL_STEPS
            ),
        )

    def warmup(self, agent, rstate, iters):
        """Rolls `iters` random-action steps into the agent's replay buffer."""
        print(f"Rolling {iters} warmup iters (compiles on the first)...",
              flush=True)
        t0 = time.time()
        rstate, transitions, next_obs = roll_random_actions(
            rstate, environment=self.environment, num_envs=self.num_envs,
            action_size=self.action_size, iters=iters,
        )
        jax.block_until_ready(next_obs)
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Filling the replay buffer...", flush=True)
        t0 = time.time()
        agent.fill_buffer(transitions, next_obs)
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)

        return rstate, int(jnp.sum(transitions.terminal | transitions.truncation))


def build_rollout(environment, test_environment, num_envs, rng):
    """Probes the environment, then builds the rollout for it.

    The probe runs first and its results are construction arguments, so the
    rollout is frozen from the moment it exists.

    Returns:
        A tuple of `(rollout, rstate)` — the frozen rollout, and the run's
        first `RolloutState` with every env program already compiled.
    """
    rstate, metric_keys = Rollout._probe(environment, num_envs, rng)
    rollout = Rollout(environment, test_environment, num_envs, metric_keys)
    return rollout, rstate
