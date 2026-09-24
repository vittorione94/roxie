"""Environment execution and rollout collection for the training loop."""

import functools
import time
from typing import Any

import numpy as np
import jax
import jax.numpy as jnp
from flax import struct
from jax.experimental import io_callback

from roxie.agents.agent import Agent
from roxie.agents.utils import Transition
from roxie.environment.functional import space_size
from roxie.environment.vector import EnvPoolVectorEnv, Timestep, VecState
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
    """One chunk's scalars: the episode accumulators and the per-step means."""

    ret: Any
    ret_sq: Any
    length: Any
    length_sq: Any
    count: Any
    noise: Any
    metrics: dict

    @classmethod
    def zeros(cls, metric_keys):
        """An empty accumulator, keyed for the metrics this env reports."""
        zero = jnp.zeros((), FLOAT)
        return cls(
            ret=zero, ret_sq=zero, length=zero, length_sq=zero, count=zero,
            noise=zero, metrics={key: zero for key in metric_keys},
        )

    @classmethod
    def from_chunk(cls, ep_ret, ep_len, done_f, noise, metrics, n_steps):
        """Reduces one chunk, in a single pass per key.

        Every argument arrives stacked over the chunk's time axis, `(T, ...)`.

        Returns:
            The chunk's `ChunkSums`.
        """
        def per_step_mean(x):
            return jnp.mean(x.reshape(n_steps, -1), axis=1)

        return cls(
            ret=jnp.sum(ep_ret),
            ret_sq=jnp.sum(jnp.square(ep_ret)),
            length=jnp.sum(ep_len),
            length_sq=jnp.sum(jnp.square(ep_len)),
            count=jnp.sum(done_f),
            noise=jnp.sum(per_step_mean(jnp.abs(noise))),
            metrics={key: jnp.sum(per_step_mean(value))
                     for key, value in metrics.items()},
        )

    def __add__(self, other):
        """Fold another chunk in, leaf by leaf. Both carry the same metric keys
        — `zeros` is built from the same `metric_keys` the chunk reports."""
        return jax.tree.map(jnp.add, self, other)


def chunk_step(carry, _, *, agent, advance, observe_params, acting_stats,
               metric_keys):
    """Runs one environment step of the fused chunk: act, advance, buffer, tally.

    Args:
        carry: `(train_state, noise_module, state, rng, params, scores,
            lengths)` — only what a step must read back from the one before it.
        _: The scan's unused `xs` slot.
        agent: The static agent instance.
        advance: The backend's step, with the env and reset pool bound.
        observe_params: The backend's `params` update, likewise bound.
        acting_stats: The normalizer, constant for the chunk.
        metric_keys: Tuple of metric keys reported by the environment.

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
    state, timestep = advance(state, action, step_key, params)
    params = observe_params(params, timestep)

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

def jax_advance(environment, state, action, key, reset_pool, params):
    """Advances a vmapped JAX env by one step, as `(state, timestep)`."""
    return environment.step(state, action, key, reset_pool, params)


def jax_observe_params(environment, params, timestep):
    """Folds one step's outcome into the env's own `params`.

    Reached through `environment` rather than a captured `func_env` so this
    stays a plain function. `FuncEnv.observe_params` is a no-op by default, so
    an env that does not adapt needs no branch.
    """
    # `terminated` rather than `done`: a non-failure cutoff must never be
    # presented to the env as a failure.
    return environment.func_env.observe_params(
        params, timestep.info, timestep.terminated,
    )


def _pool_stepper(environment):
    """The host callable the fused chunk steps a C++ pool through.

    Returns exactly `environment.step_spec` — the driver's own contract — so the
    callback's structure is known before the pool has ever been stepped. A
    pool's other `info` keys are dropped here; nothing above the rollout reads
    them.
    """
    metric_keys = tuple(environment.step_spec[-1])

    def step(action):
        # `state` is unused: a pool keeps everything in C++.
        _state, timestep = environment.step(None, np.asarray(action))
        metrics = timestep.info.get("metrics", {})
        return (timestep.obs, timestep.reward, timestep.terminated,
                timestep.truncated, {k: metrics[k] for k in metric_keys})

    return step


def pool_advance(environment, state, action, key, reset_pool, params):
    """Advances a C++ pool by one step, through an ordered host callback.

    A pool cannot be traced, but it can be CALLED from inside one. Ordered is
    what makes it sound: the pool is stateful, so its steps must run in sequence
    with each other and with the loop that feeds them.
    """
    del state, key, reset_pool, params  # the pool owns all of it, in C++
    obs, reward, terminated, truncated, metrics = io_callback(
        _pool_stepper(environment), environment.step_spec, action,
        ordered=True,
    )
    # The pool auto-resets in C++ and hands back only the post-reset
    # observation, so the timestep's obs and the next state's are one array.
    return VecState(env_state=None, obs=obs), Timestep(
        obs=obs, reward=reward, terminated=terminated, truncated=truncated,
        info={"metrics": metrics},
    )


def keep_params(environment, params, timestep):
    """Returns `params` untouched: a C++ pool owns such state natively."""
    del environment, timestep
    return params


@functools.partial(
    jax.jit, static_argnames=("environment", "advance", "observe_params"),
)
def step_envs(rstate, actions, key, *, environment, advance, observe_params):
    """Advances the environments one step, with the env's own `params` update.

    Returns:
        A tuple of `(rstate, prev_obs, timestep)`.
    """
    state, timestep = advance(
        environment, rstate.env, actions, key, rstate.reset_pool, rstate.params,
    )
    params = observe_params(environment, rstate.params, timestep)
    return rstate.replace(env=state, params=params), rstate.obs, timestep


@functools.partial(
    jax.jit,
    static_argnames=("environment", "advance", "observe_params", "agent",
                     "n_steps", "metric_keys", "unroll"),
    donate_argnums=(0, 1),
)
def collect_chunk(train_state, noise_module, rstate, frozen_stats, *,
                  environment, advance, observe_params, agent, n_steps,
                  metric_keys, unroll):
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
        environment: The training environment driver.
        advance: The backend-specific step function.
        observe_params: The backend-specific parameter update function.
        agent: The static agent instance; only its code may be used in a trace.
        n_steps: The number of environment steps to collect.
        metric_keys: Tuple of metric keys reported by the environment.
        unroll: Loop unroll count for XLA optimization.

    Returns:
        A tuple of `(train_state, noise_module, rstate, chunk_sums)`.
    """
    reset_pool = rstate.reset_pool

    def advance_step(state, action, key, params):
        return advance(environment, state, action, key, reset_pool, params)

    def observe(params, timestep):
        return observe_params(environment, params, timestep)

    acting_stats = (train_state.obs_stats if frozen_stats is None
                    else frozen_stats)
    body = functools.partial(
        chunk_step, agent=agent, advance=advance_step, observe_params=observe,
        acting_stats=acting_stats, metric_keys=metric_keys,
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
    ), ChunkSums.from_chunk(ep_ret, ep_len, done_f, noise, metrics, n_steps)


@functools.partial(
    jax.jit,
    static_argnames=("environment", "advance", "num_envs", "action_size",
                     "iters"),
)
def roll_random_actions(rstate, *, environment, advance, num_envs,
                        action_size, iters):
    """Rolls uniform-action warmup steps, as one `(T, B, ...)` block.

    Args:
        rstate: The environment-side state carry (`RolloutState`).
        environment: The training environment driver.
        advance: The backend-specific step function.
        num_envs: Worlds stepped per iteration.
        action_size: Width of one action.
        iters: The number of steps to roll.

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
        state, timestep = advance(
            environment, state, actions, step_key, rstate.reset_pool,
            rstate.params,
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
    static_argnames=("environment", "advance", "agent", "max_steps"),
)
def evaluate_policy(actor, obs_stats, state, *, environment, advance, agent,
                    max_steps):
    """Rolls the eval environments to their own ends, scoring as they go.

    Args:
        actor: The policy weights. Traced, not read off the static `agent`:
            baked in as a constant, every epoch after the first would score the
            first epoch's weights.
        obs_stats: The normalizer acting ran under. Traced, for the same reason.
        state: The eval envs from `reset_test`, so the draw is pinned.
        environment: The evaluation environment driver.
        advance: The backend-specific step function.
        agent: The static agent instance.
        max_steps: Episode-length cap for the loop.

    Returns:
        A tuple of `(scores, lengths, start_obs)`, per eval environment.
    """
    # A `while_loop`, so the termination check is on device and a run of short
    # episodes costs what it should.
    num_tests = state.obs.shape[0]
    start_obs = state.obs.reshape(num_tests, -1)
    # Fixed for the same reason the draw is: an eval is a measurement, and
    # nothing in it may vary between two evals of the same weights.
    eval_key = jax.random.PRNGKey(0)

    def cond(carry):
        i, _state, dones, _scores, _lengths = carry
        return (i < max_steps) & (~jnp.all(dones))

    def body(carry):
        i, state, dones, scores, lengths = carry
        action = agent.eval_action(state.obs, actor, obs_stats)
        # reset_pool=None: no auto-reset — each episode runs to its own end
        # and finished worlds are masked out below.
        state, timestep = advance(environment, state, action, eval_key, None,
                                  None)
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

    Subclasses supply only what a backend cannot share — its `advance` and
    `observe_params` pair, how a batch of envs is reset, and what the epoch
    boundary does to the env. The fused chunk, the warmup roll, the eval loop
    and the episode bookkeeping are all module-level and identical for both.

    Instances are frozen at construction: everything below feeds a static jit
    argument, so `_probe` settles it all before `__init__` runs and `build_rollout`
    is the only way to get one.

    Attributes:
        advance: The backend's step function, `self`-free so it can be a static
            argument without dragging the instance into the cache key.
        observe_params: The backend's `params` update, likewise.
        scan_unroll: How many chunk steps XLA sees as straight-line code per
            `lax.scan` trip. It buys fusion and register reuse ACROSS env steps
            at the price of a proportionally larger graph to compile.
        num_envs: The backend's authoritative env width — a C++ pool's own
            count, which need not be the one that was asked for.
        metric_keys: What the env reports under `info["metrics"]`.
    """

    advance = None
    observe_params = staticmethod(keep_params)
    scan_unroll = 2

    _frozen = False

    def __init__(self, environment, test_environment, num_envs, metric_keys):
        self.environment = environment
        self.test_environment = test_environment
        self.num_envs = int(num_envs)
        self.action_size = space_size(environment.single_action_space)
        self.metric_keys = tuple(metric_keys)
        self._frozen = True

    def __setattr__(self, name, value):
        # `metric_keys` and `num_envs` feed STATIC jit arguments, so a write
        # after the first trace would silently reuse a stale compilation.
        # `_probe` settles both before an instance exists; nothing may move
        # them afterwards.
        if self._frozen:
            raise AttributeError(
                f"{type(self).__name__} is frozen: cannot set {name!r}. "
                f"Anything that varies belongs in `RolloutState`."
            )
        object.__setattr__(self, name, value)

    # --- what a backend supplies -------------------------------------------

    @classmethod
    def _probe(cls, environment, num_envs, rng):
        """Resets the envs and settles what the compiled programs need static.

        A classmethod, so it runs before `__init__` and cannot mutate a
        rollout: the backend's authoritative env width and metric keys are
        construction arguments, not something `prepare` writes back.

        Returns:
            A tuple of `(rstate, num_envs, metric_keys)`.
        """
        raise NotImplementedError

    def reset_test(self):
        """The eval envs, reset to a pinned start.

        Consecutive evals of one policy must face the same draw, or a change in
        `test/score` could be the draw rather than the policy.
        """
        raise NotImplementedError

    def epoch_refresh(self, rstate):
        """`(rstate, invalidated)` — see each backend's own docstring."""
        raise NotImplementedError

    # --- the loop, shared --------------------------------------------------

    def reset_tally(self, rstate):
        """`rstate` with the in-flight per-env episode counters dropped."""
        scores, lengths = fresh_tally(self.num_envs)
        return rstate.replace(scores=scores, lengths=lengths)

    def step(self, rstate, actions, key):
        """Advances the envs one step. See `step_envs`."""
        return step_envs(
            rstate, actions, key, environment=self.environment,
            advance=self.advance, observe_params=self.observe_params,
        )

    def collect(self, train_state, noise_module, rstate, frozen_stats, *,
                agent, n_steps):
        """Collects one chunk. See `collect_chunk`, which this only binds."""
        return collect_chunk(
            train_state, noise_module, rstate, frozen_stats,
            environment=self.environment, advance=self.advance,
            observe_params=self.observe_params, agent=agent, n_steps=n_steps,
            metric_keys=self.metric_keys,
            # A chunk shorter than the unroll would otherwise be traced as a
            # peeled remainder with no scan left around it.
            unroll=max(1, min(self.scan_unroll, n_steps)),
        )

    def roll_random(self, rstate, *, iters):
        """Rolls `iters` random-action steps. See `roll_random_actions`."""
        return roll_random_actions(
            rstate, environment=self.environment, advance=self.advance,
            num_envs=self.num_envs, action_size=self.action_size, iters=iters,
        )

    def evaluate(self, actor, obs_stats, state, *, agent):
        """Runs the held-out eval. See `evaluate_policy`."""
        return evaluate_policy(
            actor, obs_stats, state, environment=self.test_environment,
            advance=self.advance, agent=agent,
            max_steps=int(self.test_environment.max_episode_steps or 1000),
        )

    def warmup(self, agent, rstate, iters):
        """Rolls `iters` random-action steps into the agent's replay buffer."""
        print(f"Rolling {iters} warmup iters (compiles on the first)...",
              flush=True)
        t0 = time.time()
        rstate, transitions, next_obs = self.roll_random(rstate, iters=iters)
        jax.block_until_ready(next_obs)
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Filling the replay buffer...", flush=True)
        t0 = time.time()
        agent.fill_buffer(transitions, next_obs)
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)

        return rstate, int(jnp.sum(transitions.terminal | transitions.truncation))


class JaxRollout(Rollout):
    """Vmapped JAX environments: physics and auto-reset all on device."""

    advance = staticmethod(jax_advance)
    observe_params = staticmethod(jax_observe_params)

    def __init__(self, environment, test_environment, num_envs, metric_keys):
        # Before `super().__init__`, which freezes the instance.
        #
        # `FuncEnv` declares `init_params`/`observe_params`/`epoch_refresh` as
        # no-ops, so an env that does not adapt needs no branch here — and the
        # driver vmaps `truncal` unconditionally, so an env missing roxie's
        # method set could not reach a rollout in the first place.
        self.func_env = environment.func_env
        # Gathered from, so it need not match num_envs; equal means one reset
        # program serves both.
        self.pool_size = int(num_envs)
        super().__init__(environment, test_environment, num_envs, metric_keys)

    def _reset(self, n, key, params):
        return reset_envs(key, params, environment=self.environment, num_envs=n)

    @classmethod
    def _probe(cls, environment, num_envs, rng):
        """Compiles every env program the loop will dispatch, then resets.

        Doing it upfront keeps first-iteration compiles out of the throughput
        print, and settles `metric_keys` from a real step: a vmapped env's
        `transition_info` is only known by running it.
        """
        num_envs = int(num_envs)
        rng, reset_key, pool_key = jax.random.split(rng, 3)
        # `params` travels as the traced FuncEnv argument, so refreshing it each
        # epoch does not retrigger a compile. None for envs that do not adapt.
        params = environment.func_env.init_params()

        def reset(n, key):
            return reset_envs(key, params, environment=environment, num_envs=n)

        state = timed(reset, num_envs, reset_key, label="reset")
        # The pool is gathered from, so it need not match `num_envs`; equal
        # means one reset program serves both.
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
            environment=environment, advance=cls.advance,
            observe_params=cls.observe_params,
        )
        metric_keys = tuple(dummy_timestep.info.get("metrics", {}))

        # The compile call above stepped this state; training must not continue
        # from it.
        rng, key = jax.random.split(rng)
        return (rstate.replace(env=reset(num_envs, key), rng=rng),
                num_envs, metric_keys)

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
        params, invalidated = self.func_env.epoch_refresh(rstate.params)

        rng, pool_key = jax.random.split(rstate.rng)
        rstate = rstate.replace(
            params=params, rng=rng,
            reset_pool=self._reset(self.pool_size, pool_key, params),
        )
        if invalidated:
            rng, reset_key = jax.random.split(rstate.rng)
            rstate = rstate.replace(
                rng=rng, env=self._reset(self.num_envs, reset_key, params),
            )
        return rstate, invalidated


class EnvPoolRollout(Rollout):
    """C++ pools: native physics, auto-reset and adaptive state owned by the pool.

    `pool_advance` bridges the pool into the shared chunk through an ordered
    host callback, so that chunk compiles for this backend too: one dispatch
    per chunk, with acting, buffering and the episode bookkeeping staying on
    device between pool steps instead of round-tripping through Python ten
    times a step. `observe_params` stays the inherited no-op — a pool owns its
    adaptive state in C++.
    """

    advance = staticmethod(pool_advance)

    @classmethod
    def _probe(cls, environment, num_envs, rng):
        """Resets the pool and reads its declared width and metric keys."""
        del num_envs  # the pool is authoritative about its own width
        print("Resetting environment...", flush=True)
        t0 = time.time()
        state, _ = environment.reset()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        num_envs = int(state.obs.shape[0])
        # Declared by the driver rather than discovered by a probe step: a pool
        # step cannot be taken back, and `step_spec` is what the fused chunk's
        # callback is traced against anyway.
        metric_keys = tuple(environment.step_spec[-1])
        scores, lengths = fresh_tally(num_envs)
        return RolloutState(
            env=VecState(env_state=None, obs=jnp.asarray(state.obs, FLOAT)),
            rng=rng, scores=scores, lengths=lengths,
        ), num_envs, metric_keys

    def reset_test(self):
        test_env = self.test_environment
        # `reset()` advances the pool's RNG, so consecutive evals would start
        # from different states; `reseed` rebuilds the pool at a fixed seed.
        reseed = getattr(test_env, "reseed", None)
        if reseed is not None:
            reseed(_EVAL_SEED)
        state, _ = test_env.reset()
        return VecState(env_state=None, obs=jnp.asarray(state.obs, FLOAT))

    def epoch_refresh(self, rstate):
        refresh = getattr(self.environment, "epoch_refresh", None)
        return rstate, (bool(refresh()) if refresh is not None else False)


def build_rollout(environment, test_environment, num_envs, rng):
    """Probes the environment, then builds the rollout for it.

    The only place a backend is named. The probe runs first and its results are
    construction arguments, so the rollout is frozen from the moment it exists
    and `num_envs` in the returned instance is the backend's authoritative
    width — a C++ pool's own count, not the one that was asked for. Callers
    should do their step accounting off `rollout.num_envs` for that reason.

    Returns:
        A tuple of `(rollout, rstate)` — the frozen rollout, and the run's
        first `RolloutState` with every env program already compiled.
    """
    cls = (EnvPoolRollout if isinstance(environment, EnvPoolVectorEnv)
           else JaxRollout)
    rstate, num_envs, metric_keys = cls._probe(environment, num_envs, rng)
    rollout = cls(environment, test_environment, num_envs, metric_keys)
    return rollout, rstate
