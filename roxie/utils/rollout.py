"""The backend-specific half of the training loop.

`Trainer._run` is a single loop that serves both backends. Everything that
differs between a vmapped JAX env (device-side physics, auto-reset by gather
from a pre-built pool, env `params` threaded through the trainer because a
traced env cannot own mutable state) and a C++ pool (native physics, auto-reset
and any such state owned by the pool) lives behind the surface below.

    xp              array namespace for the rollout's episode accumulation
    supports_async  whether an async learner can overlap acting and learning
    prepare()       compile/reset, return the first carry state
    warmup()        fill the replay buffer with random-action transitions
    step()          advance the envs by one step
    collect()       advance the envs `n` times, buffering and scoring as it goes
    epoch_refresh() the epoch boundary: let the env refresh itself, rebuild the
                    reset pool, report whether live episodes were invalidated
    evaluate()      the held-out eval rollout

`collect` is the loop's unit of work, not `step`. A per-step Python loop pays
one host dispatch per env step, and on GPU physics that dispatch IS the cost:
one `env.step` call costs 3.6 ms against 0.15 ms for the same step inside a
`lax.scan` (AcrobotSwingup, 256 envs, mujoco_warp on an RTX 5080). So the JAX
rollout scans a whole chunk of acting — select, step, buffer, score — into ONE
dispatch. Measured on that machine, end to end: 22.9k -> 103k env steps/s.

The EnvPool rollout cannot do that: its physics runs in C++ and cannot be
traced, so a chunk is irreducibly a Python loop. What it CAN do is compile
everything between two pool steps, which is two dispatches an env step against
about ten — see `EnvPoolRollout._make_step_fns`, and note there that WHICH state
each compiled half is handed matters more than how many dispatches are saved.
The eval rollout gets the same treatment and gains more from it, having no
gradient burst to hide behind — see `_make_eval_act_fn`.

Measured on a 12C/24T 7900X, CheetahRun / 256 envs, per-epoch steady state at
the release grid's 500k epoch cadence: TD3 33.9k -> 35.2k env steps/s and PPO
82.8k -> 96.4k. The split is the whole story — TD3's gradient burst is ~80% of
its loop, where PPO's acting is 62% of its own.

Both rollouts return the same per-chunk episode summary, which is what keeps one
trainer loop over both, and `tests/test_fused_collect.py` holds each compiled
path byte-for-byte against the per-step loop it replaced.

Fitting the chunk cost against its length gave 5.6 ms fixed per dispatch plus
0.134 ms per env step — and that marginal figure IS the raw scanned physics, so
the body itself was never the problem. Most of the fixed half was `nnx.jit`
walking the module graph in Python on every call, paid twice per window (once
here and once in the agent's `_grad_steps`); `agents.utils.SplitNodes` holds the
graph split across calls and took it from 2.2 ms to 0.4 ms on DDPG's train
state, 3.5 ms to 0.6 ms on TD3's. End to end on the same box, AcrobotSwingup /
warp_gpu / 256 envs: DDPG 130k -> 245k sps, TD3 97k -> 199k, GPU util 53% ->
99% and 88%. Bit-for-bit identical curves either way — it moves no computation.

WHERE THE REST STILL GOES, if someone picks this up again. HOW MUCH host time is
even worth chasing was measured by injecting 2 ms of it per window: only 71% of
that showed up in wall time on DDPG and 32% on TD3 (the device absorbs the rest,
and TD3 has more device work per window to absorb it with). So a millisecond
saved on the host returns well under a millisecond, and run-to-run spread on a
3M-step run is ~2% — anything predicted below that cannot be measured here.

Two candidates, both weighed against that:
  * Folding the chunk into the epoch accumulator INSIDE the scan, instead of the
    trainer's host-side `_merge_chunk`. Tried and reverted: the host work it
    removes is 7 one-element device adds, 0.13 ms/window, so its ceiling is
    1.1% on DDPG and 0.4% on TD3 — under the noise floor, and the A/B duly
    measured a wash. Do not retry it.
  * Running several windows — acting AND their gradient bursts — inside one
    dispatch. This is the only remaining lever with a ceiling above the noise
    (~6% DDPG / ~4% TD3, from the two `graph_jit` boundaries at the exposure
    above), but it is a real change to how the trainer and the learner divide up
    the work: `due_for_update` has to become traced, per-window metrics have to
    be stacked in-scan rather than appended on the host, and the epoch cadence
    stops landing on the same step counts as every curve on disk.

The chunk cannot simply be widened instead: it is pinned to one
`steps_between_updates` window because that is what keeps the actor fixed for
its duration. And raising `parallel_envs` is NOT the lever: the window is a
fixed count of env steps, so more envs only shortens the scan, and the physics
it shortens is ~7% of the window.

`step` returns `(state, prev_obs, timestep)`. `prev_obs` is the observation the
action was selected from; `timestep` carries pre-auto-reset values — the true
next observation for the replay buffer — while `state.obs` is what the next
action is selected from, a fresh start for a done env. On the JAX path those are
distinct arrays; a C++ pool hands back only the post-reset observation, so there
they are the same object. Harmless, because `terminated` zeroes the bootstrap.
"""

import dataclasses
import functools
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent
from roxie.agents.utils import Transition
from roxie.environment.functional import space_size
from roxie.environment.vector import EnvPoolVectorEnv
from roxie.models.actors import deterministic_action

_EVAL_SEED = 12345


def timed(fn, *args, label=""):
    """Run `fn`, blocking on its output, and print how long the compile took."""
    print(f"Compiling {label}...", flush=True)
    t0 = time.time()
    out = fn(*args)
    jax.block_until_ready(out)
    print(f"  {time.time() - t0:.1f}s", flush=True)
    return out


CHUNK_SUMS = ("ret", "ret_sq", "len", "len_sq", "count", "noise")


def new_chunk_sums(xp, metric_keys=()):
    """Zeroed chunk accumulators in `xp`'s namespace.

    Array zeros rather than Python floats: on the JAX path this dict is a
    `lax.scan` carry, and a carry whose leaves change type between the initial
    value and the body's output does not typecheck.
    """
    return {
        **{k: xp.zeros((), dtype=xp.float32) for k in CHUNK_SUMS},
        "metrics": {k: xp.zeros((), dtype=xp.float32) for k in metric_keys},
    }


def accumulate_episodes(acc, scores, lengths, done, xp):
    """Fold this step's completed episodes into `acc` and return how many
    finished.

    `xp` is `jnp` on the JAX rollout, where a host reduction would sync the
    pipeline every step, and `np` on EnvPool, where arrays are host-side. The
    two are pinned to agree by `tests/test_trainer_bookkeeping.py`.

    `ret`/`len` collect only episodes that actually terminated; their `_sq`
    companions recover the per-episode std at the epoch boundary via
    sqrt(E[x^2] - E[x]^2), avoiding a host-side list and a per-step sync.
    """
    done_f = done.astype(xp.float32)
    done_sum = xp.sum(done_f)
    ep_ret, ep_len = scores * done_f, lengths.astype(xp.float32) * done_f
    acc["ret"] = acc["ret"] + xp.sum(ep_ret)
    acc["ret_sq"] = acc["ret_sq"] + xp.sum(ep_ret ** 2)
    acc["len"] = acc["len"] + xp.sum(ep_len)
    acc["len_sq"] = acc["len_sq"] + xp.sum(ep_len ** 2)
    acc["count"] = acc["count"] + done_sum
    return done_sum


class _BufferSlots:
    """The two `TrainState` fields buffering writes, on their own.

    `Agent.buffer_transitions` and PPO's override touch exactly
    `state.buffer_state` and `state.obs_stats`, so handing them these instead of
    the whole train state is what keeps `EnvPoolRollout`'s compiled buffer step
    at 10 pytree leaves rather than TD3's 142 — the difference between 0.038 and
    0.726 ms per env step, since a jit call flattens its arguments and books
    their donation in Python, per leaf.

    `__slots__` so that an agent whose buffering grows a third field fails here,
    loudly, instead of silently writing it to a shim nobody reads back.
    """

    __slots__ = ("buffer_state", "obs_stats")

    def __init__(self, buffer_state, obs_stats):
        self.buffer_state = buffer_state
        self.obs_stats = obs_stats


def fusable(agent, learner) -> bool:
    """May a rollout compile acting and buffering itself, rather than
    dispatching them op by op through the learner?

    Two conditions, and both rollouts ask the same question.

    The AGENT has to expose the pure `select_action` / `buffer_transitions`
    pair — the same capability test `build_learner` uses for the async
    learner. Pure means they take the actor, the observation statistics and the
    train state EXPLICITLY instead of reading `self.state`, so they can run
    against a `lax.scan` carry or a donated pytree. Every agent here does;
    a bare baseline (`agents.basic`) does not, and keeps the per-step loop.

    The LEARNER must not own `agent.state`. The async learner's thread is its
    sole owner, and there `act`/`buffer` are a hand-off — behaviour snapshot
    out, transitions into a queue — rather than plain calls, so bypassing them
    would both race it and drop transitions. Only reachable from
    `EnvPoolRollout`, the one backend whose `supports_async` is True.
    """
    return not learner.owns_state and all(
        hasattr(agent, name)
        for name in ("select_action", "buffer_transitions")
    )


def stepwise_collect(rollout, state, n_steps, learner):
    """`collect` as a Python loop, one host dispatch per env step.

    What every backend used to do, and what still runs for a C++ pool (whose
    step cannot be traced) and for an agent that does not expose the pure
    `select_action` / `buffer_transitions` pair the fused path needs — PPO
    today. Goes through the learner rather than the agent so the async hand-off
    (behaviour-actor snapshot out, transitions into a queue) is unchanged.
    """
    xp = rollout.xp
    sums = new_chunk_sums(xp)
    # Same chunk boundary the fused path pins at.
    rollout.agent.freeze_acting_norm()
    for _ in range(n_steps):
        # Same three-way split as the fused body: both paths must consume the
        # same stream. `tests/test_fused_collect.py` pins that.
        rollout.rng, act_key, step_key = jax.random.split(rollout.rng, 3)
        actions, applied_noise = learner.act(state.obs, act_key)
        state, prev_obs, timestep = rollout.step(state, actions, step_key)
        learner.buffer(prev_obs, timestep, actions)

        reward = xp.asarray(timestep.reward)
        done = xp.asarray(timestep.terminated) | xp.asarray(timestep.truncated)
        rollout.scores = rollout.scores + reward
        rollout.lengths = rollout.lengths + 1
        accumulate_episodes(sums, rollout.scores, rollout.lengths, done, xp)
        if applied_noise is not None:
            sums["noise"] = sums["noise"] + xp.mean(xp.abs(applied_noise))
        for key, value in timestep.info.get("metrics", {}).items():
            sums["metrics"][key] = sums["metrics"].get(key, 0.0) + xp.mean(value)
        rollout.scores = xp.where(done, 0, rollout.scores)
        rollout.lengths = xp.where(done, 0, rollout.lengths)
    return state, sums


class JaxRollout:
    """Vmapped JAX environments: physics and auto-reset all on device."""

    xp = jnp
    # The env step is GPU-bound, so a background learner would contend for the
    # device rather than overlap with it.
    supports_async = False

    def __init__(self, environment, test_environment, agent, num_envs, rngs,
                 test_episodes):
        self.environment = environment
        self.test_environment = test_environment
        self.agent = agent
        self.num_envs = num_envs
        self.num_test_episodes = int(test_episodes)
        self.rng = rngs.envs()
        self.action_size = space_size(environment.single_action_space)
        self.action_low, self.action_high = agent.action_low, agent.action_high
        # Gathered from, so it need not match num_envs; equal means one reset
        # program serves both.
        self.pool_size = num_envs
        self._eval_fn = None
        # Recompiled if the chunk length changes: `n_steps` is a scan length.
        self._collect_fn = None
        self._collect_steps = None
        self.scores = None
        self.lengths = None

        # `params` travels as the traced FuncEnv argument, so refreshing it each
        # epoch does not retrigger a compile. None for envs that do not adapt.
        func_env = environment.func_env
        self.params = getattr(func_env, "init_params", lambda: None)()
        self._observe_params = getattr(
            func_env, "observe_params", lambda params, info, terminated: params,
        )
        self._env_epoch_refresh = getattr(
            func_env, "epoch_refresh", lambda params: (params, False),
        )

        self._jit_reset = jax.jit(
            self.environment.reset, static_argnames=("num_envs",),
        )
        self._train_step = jax.jit(self._step_and_observe)

    def _step_and_observe(self, state, actions, rng, reset_pool, params):
        """One env step plus the env's own `params` update, in one dispatch.

        Passes `terminated` rather than `done` so a non-failure cutoff is never
        presented to the env as a failure.
        """
        state, timestep = self.environment.step(
            state, actions, rng, reset_pool, params,
        )
        params = self._observe_params(
            params, timestep.info, timestep.terminated,
        )
        return state, timestep, params

    def _random_actions(self, key):
        u = jax.random.uniform(key, (self.num_envs, self.action_size))
        return self.action_low + (self.action_high - self.action_low) * u

    def _reset(self, n, key):
        return self._jit_reset(key, self.params, num_envs=n)[0]

    def prepare(self):
        """Compile every program the loop will dispatch, then return a clean
        reset state. Doing it upfront keeps first-iteration compiles out of the
        throughput print."""
        agent = self.agent

        self.rng, reset_key, pool_key = jax.random.split(self.rng, 3)
        state = timed(self._reset, self.num_envs, reset_key, label="reset")
        self.reset_pool = timed(
            self._reset, self.pool_size, pool_key, label="reset pool",
        )

        dummy_actions = jnp.zeros((self.num_envs, self.action_size))
        _, dummy_timestep, _ = timed(
            self._train_step, state, dummy_actions, self.rng, self.reset_pool,
            self.params, label="train step",
        )
        # The fused collect carries these sums through a `lax.scan`, and a
        # carry's keys must be known before tracing.
        self._metric_keys = tuple(dummy_timestep.info.get("metrics", {}))

        if getattr(agent, "memory_warmup", 0) > 0:
            dummy_t = jax.tree.map(
                lambda leaf: jnp.zeros((self.num_envs,) + leaf.shape[2:], leaf.dtype),
                agent.state.buffer_state.experience,
            )
            # Against a COPY: `replay_add` donates its input, so handing it the
            # live buffer would leave the agent holding a deleted array.
            timed(agent.replay_add,
                  jax.tree.map(jnp.copy, agent.state.buffer_state), dummy_t,
                  label="replay add")

        # The compile calls above stepped their states; training must not
        # continue from them.
        self.reset_tally()
        self.rng, key = jax.random.split(self.rng)
        return self._reset(self.num_envs, key)

    def reset_tally(self):
        """Drop the in-flight per-env episode counters.

        Called at startup and whenever the env reports that its epoch refresh
        invalidated the episodes in progress — their part-scored returns would
        otherwise be credited to states the env no longer has.
        """
        self.scores = jnp.zeros(self.num_envs)
        self.lengths = jnp.zeros(self.num_envs, dtype=jnp.int32)

    def warmup(self, agent, iters, state):
        """Fill the replay buffer with `iters` scanned random-action steps.

        Scanned rather than looped so the whole fill is one dispatch — a per-step
        Python loop costs minutes at the step counts warmup needs.
        """
        @jax.jit
        def warmup_rollout(state, rng, reset_pool, params):
            def body(carry, _):
                state, rng = carry
                rng, act_key, step_key = jax.random.split(rng, 3)
                actions = self._random_actions(act_key)
                prev_obs = state.obs
                # No `observe_params`: random-action terminations are not
                # failures any policy caused.
                state, timestep = self.environment.step(
                    state, actions, step_key, reset_pool, params,
                )
                transition = Transition(
                    observation=prev_obs,
                    action=actions,
                    reward=timestep.reward,
                    terminal=timestep.terminated,
                    truncation=timestep.truncated,  # pruned below if unused
                )
                return (state, rng), (transition, timestep.obs)
            return jax.lax.scan(body, (state, rng), None, length=iters)

        print("Compiling warmup rollout...", flush=True)
        t0 = time.time()
        compiled = warmup_rollout.lower(
            state, self.rng, self.reset_pool, self.params,
        ).compile()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Running warmup rollout...", flush=True)
        t0 = time.time()
        (state, self.rng), (transitions, next_obs) = compiled(
            state, self.rng, self.reset_pool, self.params,
        )
        state.obs.block_until_ready()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        # Prune to the fields the buffer stores, so the pytrees match at add.
        proto = agent.state.buffer_state.experience
        transitions = Transition(**{
            f.name: (getattr(transitions, f.name)
                     if getattr(proto, f.name) is not None else None)
            for f in dataclasses.fields(Transition)
        })

        # Without donation XLA allocates a full output copy — a transient 2x of
        # the buffer's obs store that OOMs right here.
        @functools.partial(jax.jit, donate_argnums=(0,))
        def batch_add(buffer_state, transitions):
            def add_one(bs, t):
                # Un-jitted: already inside a `jax.jit`, so the donating
                # wrapper would only nest a `pjit`.
                return agent._replay_add(bs, t), None
            bs, _ = jax.lax.scan(add_one, buffer_state, transitions)
            return bs

        print("Compiling replay fill...", flush=True)
        t0 = time.time()
        compiled_add = batch_add.lower(agent.state.buffer_state, transitions).compile()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Running replay fill...", flush=True)
        t0 = time.time()
        agent.state.buffer_state = compiled_add(agent.state.buffer_state, transitions)
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)

        # The scanned fill bypassed `agent.add_transitions`, which normally
        # folds observations into the stats.
        if agent.normalize_observations:
            all_obs = jnp.concatenate([
                transitions.observation.reshape(-1, transitions.observation.shape[-1]),
                next_obs.reshape(-1, next_obs.shape[-1]),
            ], axis=0)
            agent.state.obs_stats = Agent.update_obs_stats(
                agent.state.obs_stats, all_obs,
            )

        return state, int(jnp.sum(transitions.terminal))

    def step(self, state, actions, key=None):
        if key is None:
            self.rng, key = jax.random.split(self.rng)
        prev_obs = state.obs
        state, timestep, self.params = self._train_step(
            state, actions, key, self.reset_pool, self.params,
        )
        return state, prev_obs, timestep

    def _make_collect_fn(self, n_steps: int):
        """Compile one `n_steps` acting burst into a single dispatch.

        The carry is everything a step mutates: the agent's train state and
        noise module (as one nnx node pair, split/merged around the body exactly
        as `fused_grad_steps` does for a gradient burst), the env state, the
        rng, the env's own `params`, the per-env episode counters and this
        chunk's sums. `reset_pool` is loop-constant — the epoch boundary
        regenerates it — so it rides in as a plain argument.

        The actor does NOT change inside a burst: the trainer sizes the chunk to
        one `steps_between_updates` window and runs the gradient burst at its
        end, which is where the per-step loop ran it too. That is what makes
        this a pure throughput change rather than a different algorithm.
        """
        agent = self.agent
        environment = self.environment
        observe_params = self._observe_params
        metric_keys = self._metric_keys

        # `jax.jit` with the graphdefs static, not `nnx.jit`, which re-walks the
        # module graph in Python on every call — see `agents.utils.SplitNodes`.
        # The train state carries the replay buffer, so donate it.
        @functools.partial(
            jax.jit, static_argnums=(0, 1), donate_argnums=(2,)
        )
        def collect_fn(agent_graphdef, noise_graphdef, agent_pytree,
                       noise_pytree, state, rng, reset_pool, params, scores,
                       lengths):
            # Read once here, not off the carry: inside the scan `obs_stats`
            # advances on every add. Merging is trace-time only.
            frozen_stats = nnx.merge(agent_graphdef, agent_pytree)[0].obs_stats

            def body(carry, _):
                (agent_pytree, noise_pytree, state, rng, params, scores,
                 lengths, sums) = carry
                # `agent_pytree` covers every node the gradient burst owns;
                # acting reads only the train state, the rest ride through so
                # both bursts share one split.
                agent_nodes = nnx.merge(agent_graphdef, agent_pytree)
                train_state = agent_nodes[0]
                noise_module = (
                    None if noise_graphdef is None
                    else nnx.merge(noise_graphdef, noise_pytree)[0]
                )

                rng, act_key, step_key = jax.random.split(rng, 3)
                prev_obs = state.obs
                acting_stats = (
                    frozen_stats if agent.freeze_obs_norm_per_chunk
                    else train_state.obs_stats
                )
                action, applied_noise, extras = agent.select_action(
                    train_state.actor, acting_stats, prev_obs, act_key,
                    noise_module=noise_module, critic=train_state.critic,
                )
                state, timestep = environment.step(
                    state, action, step_key, reset_pool, params,
                )
                # `terminated`, not `done`: a non-failure cutoff is not a
                # failure the env should see.
                params = observe_params(
                    params, timestep.info, timestep.terminated,
                )

                agent.buffer_transitions(
                    train_state, prev_obs, action, timestep.reward,
                    timestep.terminated, timestep.truncated, timestep.obs,
                    extras,
                )
                # On top of the update `buffer_transitions` already performs.
                # Redundant-looking, but dropping it reweights the statistics
                # away from every run logged so far.
                train_state.obs_stats = Agent.update_obs_stats(
                    train_state.obs_stats, timestep.obs,
                )

                done = timestep.terminated | timestep.truncated
                scores = scores + timestep.reward
                lengths = lengths + 1
                accumulate_episodes(sums, scores, lengths, done, jnp)
                sums["noise"] = sums["noise"] + jnp.mean(jnp.abs(applied_noise))
                step_metrics = timestep.info.get("metrics", {})
                for key in metric_keys:
                    sums["metrics"][key] = (
                        sums["metrics"][key] + jnp.mean(step_metrics[key])
                    )
                scores = jnp.where(done, 0.0, scores)
                lengths = jnp.where(done, 0, lengths)

                # The writes above went through `train_state`, which is
                # `agent_nodes[0]`, so re-splitting gives the post-step carry.
                _, agent_pytree = nnx.split(agent_nodes)
                if noise_graphdef is not None:
                    _, noise_pytree = nnx.split((noise_module,))
                return (
                    agent_pytree, noise_pytree, state, rng, params, scores,
                    lengths, sums,
                ), None

            init = (
                agent_pytree, noise_pytree, state, rng, params, scores, lengths,
                new_chunk_sums(jnp, metric_keys),
            )
            carry, _ = jax.lax.scan(body, init, None, length=n_steps)
            (agent_pytree, noise_pytree, state, rng, params, scores, lengths,
             sums) = carry
            return (
                agent_pytree, noise_pytree, state, rng, params,
                scores, lengths, sums,
            )

        return collect_fn

    def collect(self, state, n_steps, learner):
        """Advance the envs `n_steps` times, buffering and scoring as it goes.

        One dispatch for the whole chunk on the fused path. The `learner` only
        decides WHICH path: `supports_async` is False for this backend, so it is
        always the synchronous one and `fusable` never rejects it on that count;
        past the check it drives the per-step fallback for agents that cannot be
        traced.
        """
        if not fusable(self.agent, learner):
            return stepwise_collect(self, state, n_steps, learner)

        # Before the burst, which donates the train state: the snapshot has to
        # be taken while those arrays are still alive.
        self.agent.freeze_acting_norm()

        compiling = self._collect_steps != n_steps
        if compiling:
            self._collect_fn = self._make_collect_fn(n_steps)
            self._collect_steps = n_steps
            print(f"Compiling {n_steps}-step acting burst...", flush=True)
            t0 = time.time()

        agent_nodes = self.agent.burst_nodes
        agent_graphdef, agent_pytree = agent_nodes.split()
        # None for agents that explore from their own stochastic policy: a
        # static None graphdef, so the merge is skipped inside the trace.
        noise_nodes = getattr(self.agent, "_noise_nodes", None)
        noise_graphdef, noise_pytree = (
            noise_nodes.split() if noise_nodes is not None else (None, None)
        )

        (agent_pytree, noise_pytree, state, self.rng, self.params, self.scores,
         self.lengths, sums) = self._collect_fn(
            agent_graphdef, noise_graphdef, agent_pytree, noise_pytree, state,
            self.rng, self.reset_pool, self.params, self.scores, self.lengths,
        )
        if compiling:
            jax.block_until_ready(jax.tree.leaves(sums))
            print(f"  {time.time() - t0:.1f}s", flush=True)
        # The burst donated its inputs, so the agent must adopt what came back
        # or the next one reads deleted buffers.
        agent_nodes.replace(agent_pytree)
        if noise_nodes is not None:
            noise_nodes.replace(noise_pytree)
        return state, sums

    def epoch_refresh(self, state):
        """Let the env refresh itself, then regenerate the reset pool (fresh
        random starts for auto-reset) from the refreshed `params`.

        An env that reports `invalidated` has changed something its in-progress
        episodes referred to, so the live envs are reset and the trainer is told
        to drop their part-scored episodes.
        """
        self.params, invalidated = self._env_epoch_refresh(self.params)

        self.rng, pool_key = jax.random.split(self.rng)
        self.reset_pool = self._reset(self.pool_size, pool_key)

        if invalidated:
            self.rng, reset_key = jax.random.split(self.rng)
            state = self._reset(self.num_envs, reset_key)
        return state, invalidated

    def _make_eval_fn(self, num_tests, max_steps):
        """Build a single compiled eval rollout.

        The episode loop runs inside ``jax.lax.while_loop`` so the termination
        check is on-device. Actor / obs-stats are traced args, so one compile is
        reused every epoch.
        """
        agent = self.agent
        normalize = agent.normalize_observations
        test_env = self.test_environment
        # Fixed, so consecutive evals of the same policy are identical.
        eval_key = jax.random.PRNGKey(0)

        @nnx.jit
        def eval_fn(actor, obs_stats, state):
            def cond(carry):
                i, _state, dones, _scores, _lengths = carry
                return (i < max_steps) & (~jnp.all(dones))

            def body(carry):
                i, state, dones, scores, lengths = carry

                obs = state.obs
                if normalize:
                    mean, std = Agent.obs_mean_std(obs_stats, agent.obs_eps)
                    obs = Agent.normalize_obs(obs, mean, std, agent.obs_clip)
                # No noise module: its stateful update cannot be mutated
                # across the while_loop trace level.
                # Off the `select_action` path, so it needs its own scrub: a
                # NaN here would otherwise reach the physics through the clip,
                # which passes NaN straight through.
                action = jnp.clip(
                    Agent.finite_or_zero(deterministic_action(actor(obs))), -1.0, 1.0
                )
                action = Agent.scale_to_env(action, agent.action_low, agent.action_high)

                # reset_pool=None: no auto-reset — each episode runs to its
                # own end and finished worlds are masked out below.
                state, timestep = test_env.step(state, action, eval_key, None)
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
            return scores, lengths

        return eval_fn

    def evaluate(self, agent):
        num_tests = int(self.test_environment.num_envs)
        max_steps = int(self.test_environment.max_episode_steps or 1000)

        if self._eval_fn is None:
            # Eager, a vmapped reset re-dispatches the physics op by op.
            self._jit_test_reset = jax.jit(self.test_environment.reset)
            self._eval_fn = self._make_eval_fn(num_tests, max_steps)

        state, _ = self._jit_test_reset(jax.random.PRNGKey(_EVAL_SEED))
        scores, lengths = self._eval_fn(
            agent.state.actor, agent.state.obs_stats, state,
        )
        start_obs = np.asarray(state.obs).reshape(num_tests, -1)
        return np.array(scores), np.array(lengths), start_obs


class EnvPoolRollout:
    """C++ pools: native physics, auto-reset and adaptive state owned by the pool.

    The pool batches natively, so no vmap/jit wrapping is needed: the loop over
    steps is Python and the episode bookkeeping is numpy on host-side arrays.
    The agent still runs in JAX, and `collect` compiles its per-step work rather
    than dispatching it op by op — see `_make_step_fns`.
    """

    xp = np
    # Acting is on the CPU, so a background GPU learner genuinely overlaps.
    supports_async = True

    def __init__(self, environment, test_environment, agent, num_envs, rngs,
                 test_episodes):
        self.environment = environment
        self.test_environment = test_environment
        self.agent = agent
        self.num_envs = num_envs
        self.num_test_episodes = int(test_episodes)
        self.rng = rngs.envs()
        self.action_size = space_size(environment.single_action_space)
        self.action_low, self.action_high = agent.action_low, agent.action_high
        # Built on first use and reused for the run: unlike `JaxRollout`'s
        # scanned chunk these do not close over the chunk length.
        self._step_fns = None
        self._eval_act_fn = None

    def _random_actions(self, key):
        u = jax.random.uniform(key, (self.num_envs, self.action_size))
        return self.action_low + (self.action_high - self.action_low) * u

    def prepare(self):
        print("Resetting environment...", flush=True)
        t0 = time.time()
        state, _ = self.environment.reset()
        print(f"  {time.time() - t0:.1f}s", flush=True)
        self.reset_tally()
        return state

    def reset_tally(self):
        """Drop the in-flight per-env episode counters. See `JaxRollout`."""
        self.scores = np.zeros(self.num_envs)
        self.lengths = np.zeros(self.num_envs, dtype=np.int32)

    def warmup(self, agent, iters, state):
        """Fill the replay buffer with `iters` random-action steps.

        A plain loop, unlike the JAX path's scan: the pool steps in C++ and
        cannot be traced, so there is nothing to fuse.
        """
        episodes = 0
        t0 = time.time()
        for _ in range(iters):
            self.rng, act_key = jax.random.split(self.rng)
            actions = self._random_actions(act_key)
            agent.last_action = actions
            prev_obs = state.obs
            state, timestep = self.environment.step(state, actions)
            agent.add(prev_obs, timestep)
            episodes += int(np.sum(timestep.terminated | timestep.truncated))
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return state, episodes

    def step(self, state, actions, key=None):
        # Ignored for signature parity: a pool owns its own RNG in C++.
        del key
        prev_obs = state.obs
        # The pool auto-resets in C++, so `timestep.obs` and `state.obs` are the
        # same array (see the module docstring).
        state, timestep = self.environment.step(state, actions)
        return state, prev_obs, timestep

    def _make_step_fns(self):
        """Compile the two halves of an env step: acting, and buffering.

        The pool steps in C++ and cannot be traced, so there is no whole chunk
        to scan into one dispatch the way `JaxRollout` does — but everything
        BETWEEN two pool steps can still be compiled instead of dispatched op by
        op. The per-step loop this replaces issues about ten: `random.split`,
        `obs_mean_std` (six UN-jitted eager ops), `normalize_obs`, the actor's
        step function, `scale_to_env`, `_pruned_transition` (whose
        `finite_or_zero` scrub is eager too), `replay_add`, `concatenate` and
        `update_obs_stats`.

        The actor's step function is the expensive one, because it is an
        `nnx.jit` and therefore re-walks the module graph IN PYTHON on every env
        step — the exact cost `agents.utils.SplitNodes` was written to remove,
        and the one path it was never removed from, since only a C++ pool still
        calls it per step.

        TWO programs rather than one, because the pool step sits between them:
        acting has to finish before the physics can run, and what to buffer is
        not known until it has.

        EACH TAKES ONLY WHAT IT TOUCHES, which is the whole difficulty. The
        obvious version hands both halves the agent's `burst_nodes` pytree, as
        the gradient burst and the JAX rollout do — and it is SLOWER than the
        loop it replaces, because a jit call flattens its arguments and books
        their donation in PYTHON, per leaf, and TD3's train state is 142 of them
        against the 10 buffering writes. Measured on TD3 / CheetahRun / 256
        envs, per env step:

                                        act     buffer    total
            the per-step loop          0.570    0.164     0.734 ms
            whole train state in       0.635    0.726     1.361 ms   <- worse
            only what it touches       0.369    0.038     0.407 ms

        So acting takes the actor and critic as their own split (34 leaves) —
        sound because the trainer sizes a chunk to one update window, so neither
        moves inside one — and buffering takes the two `TrainState` fields it
        writes (10). The observation statistics thread through both, because
        acting normalizes against the live ones exactly as the per-step loop
        did.

        The critic rides along for every agent though only PPO reads it at
        acting time, which costs the other six 0.095 ms a step (0.369 -> 0.274
        with an actor-only split), or ~1% end to end. Deliberately not taken:
        the saving is small, and the flag it needs would be a silent
        correctness hazard rather than a loud one — `PPO.select_action` falls
        back to `self.state.critic` when handed None, which inside this trace is
        a captured constant that would go stale at the first gradient burst.
        """
        agent = self.agent

        @functools.partial(jax.jit, static_argnums=(0, 1), donate_argnums=(3,))
        def act_fn(acting_graphdef, noise_graphdef, acting_pytree, noise_pytree,
                   frozen_stats, obs_stats, rng, obs):
            actor, critic = nnx.merge(acting_graphdef, acting_pytree)
            noise_module = (
                None if noise_graphdef is None
                else nnx.merge(noise_graphdef, noise_pytree)[0]
            )
            # The SAME three-way split, in the same order, as the per-step loop:
            # a C++ pool owns its RNG and drops `step_key`, but drawing it is
            # what keeps the two paths on one stream.
            rng, act_key, _step_key = jax.random.split(rng, 3)
            # `None` means "normalize against the live statistics"; PPO hands
            # back its pin. See `Agent.freeze_acting_norm`.
            acting_stats = frozen_stats if frozen_stats is not None else obs_stats
            action, applied_noise, extras = agent.select_action(
                actor, acting_stats, obs, act_key,
                noise_module=noise_module, critic=critic,
            )
            if noise_graphdef is not None:
                _, noise_pytree = nnx.split((noise_module,))
            return noise_pytree, rng, action, applied_noise, extras

        @functools.partial(jax.jit, donate_argnums=(0, 1))
        def buffer_fn(buffer_state, obs_stats, prev_obs, action, reward,
                      terminated, truncated, next_obs, extras):
            slots = _BufferSlots(buffer_state, obs_stats)
            agent.buffer_transitions(
                slots, prev_obs, action, reward, terminated, truncated,
                next_obs, extras,
            )
            # On top of the update `buffer_transitions` already performs — the
            # same deliberate double count `SyncLearner.buffer` and the fused
            # JAX burst both make. Dropping it reweights the statistics away
            # from every run logged so far.
            slots.obs_stats = Agent.update_obs_stats(slots.obs_stats, next_obs)
            return slots.buffer_state, slots.obs_stats

        return act_fn, buffer_fn

    def collect(self, state, n_steps, learner):
        """Advance the pool `n_steps` times, buffering and scoring as it goes.

        Two dispatches per env step, against about ten for the per-step loop it
        replaces. The physics runs in C++ between them and cannot be traced,
        which is what stops the whole chunk collapsing into one dispatch as it
        does on `JaxRollout`; episode bookkeeping stays numpy on host arrays,
        where it costs 0.009 ms a step and a device round trip would cost more.

        Falls back to the per-step loop for an agent without the pure pair and
        for the async learner, which owns `agent.state` — see `fusable`.
        """
        if not fusable(self.agent, learner):
            return stepwise_collect(self, state, n_steps, learner)

        agent = self.agent
        # Before anything is split or donated: PPO pins its acting statistics
        # here, and reading `agent.state` to do it re-materializes the live
        # nodes. `None` from any other agent.
        frozen_stats = agent.freeze_acting_norm()

        compiling = self._step_fns is None
        if compiling:
            self._step_fns = self._make_step_fns()
            print("Compiling fused acting step...", flush=True)
            t0 = time.time()
        act_fn, buffer_fn = self._step_fns

        # Re-split every chunk rather than held across them: the gradient burst
        # between two chunks updates these modules in place through
        # `burst_nodes`, which a cached split of its own would not see.
        acting_graphdef, acting_pytree = nnx.split(
            (agent.state.actor, agent.state.critic)
        )
        # None for agents that explore from their own stochastic policy: a
        # static None graphdef, so the merge is skipped inside the trace.
        noise_nodes = getattr(agent, "_noise_nodes", None)
        noise_graphdef, noise_pytree = (
            noise_nodes.split() if noise_nodes is not None else (None, None)
        )
        buffer_state = agent.state.buffer_state
        obs_stats = agent.state.obs_stats

        sums = new_chunk_sums(np)
        # A pool hands back one numpy observation that is both `state.obs` and
        # `timestep.obs` (see the class docstring), so one transfer per step
        # serves acting now and the `next_obs` of the transition buffered below.
        obs = jnp.asarray(state.obs)
        for _ in range(n_steps):
            noise_pytree, self.rng, action, applied_noise, extras = act_fn(
                acting_graphdef, noise_graphdef, acting_pytree, noise_pytree,
                frozen_stats, obs_stats, self.rng, obs,
            )
            prev_obs = obs
            state, timestep = self.environment.step(state, action)
            obs = jnp.asarray(state.obs)
            buffer_state, obs_stats = buffer_fn(
                buffer_state, obs_stats, prev_obs, action, timestep.reward,
                timestep.terminated, timestep.truncated, obs, extras,
            )

            reward = np.asarray(timestep.reward)
            done = np.asarray(timestep.terminated) | np.asarray(timestep.truncated)
            self.scores = self.scores + reward
            self.lengths = self.lengths + 1
            accumulate_episodes(sums, self.scores, self.lengths, done, np)
            sums["noise"] = sums["noise"] + np.mean(np.abs(applied_noise))
            for key, value in timestep.info.get("metrics", {}).items():
                sums["metrics"][key] = sums["metrics"].get(key, 0.0) + np.mean(value)
            self.scores = np.where(done, 0, self.scores)
            self.lengths = np.where(done, 0, self.lengths)

        # The buffering donated its inputs, so the agent must adopt what came
        # back or the gradient burst reads deleted buffers. The acting modules
        # were never written to and need no write-back.
        agent.state.buffer_state = buffer_state
        agent.state.obs_stats = obs_stats
        if noise_nodes is not None:
            noise_nodes.replace(noise_pytree)
        if compiling:
            print(f"  {time.time() - t0:.1f}s (first chunk, includes compile)",
                  flush=True)
        return state, sums

    def epoch_refresh(self, state):
        refresh = getattr(self.environment, "epoch_refresh", None)
        return state, (bool(refresh()) if refresh is not None else False)

    def _make_eval_act_fn(self):
        """Action selection for the eval loop, compiled.

        The eval loop stays Python for the same reason `collect`'s does — a C++
        pool between the steps — and it was paying the same per-step dispatch
        tax, `agent.step` being about ten of them with the actor's `nnx.jit`
        re-walking the module graph in Python every time. It is a LARGER share
        here than in collect, because eval has no gradient burst to hide behind.
        Measured on CheetahRun, per eval step, of which there are up to
        `max_episode_steps` once an epoch:

                    eval step    of which `agent.step`
            TD3      0.766 ms          0.445 ms
            PPO      1.138 ms          0.810 ms

        Its own program rather than `_make_step_fns`'s `act_fn`, because each of
        the three differences would be a bug if forced through one function.
        `evaluate` is static and flips a stochastic actor from a sample to its
        mode. NOTHING is donated: eval must not consume the weights it is
        scoring. And eval reuses ONE fixed key for every step instead of
        advancing a stream, which is half of what makes consecutive evals of the
        same policy identical (the pinned pool reseed below is the other half).
        """
        agent = self.agent

        @functools.partial(jax.jit, static_argnums=(0, 1))
        def eval_act_fn(acting_graphdef, noise_graphdef, acting_pytree,
                        noise_pytree, obs_stats, obs, key):
            actor, critic = nnx.merge(acting_graphdef, acting_pytree)
            # Carried, though no agent's noise module does anything under
            # `evaluate=True` — `add_noise` returns the action untouched and
            # never advances its counter, which is why eval can leave the live
            # module alone rather than adopting one back.
            noise_module = (
                None if noise_graphdef is None
                else nnx.merge(noise_graphdef, noise_pytree)[0]
            )
            action, _noise, _extras = agent.select_action(
                actor, obs_stats, obs, key, evaluate=True,
                noise_module=noise_module, critic=critic,
            )
            return action

        return eval_act_fn

    def _eval_select(self, agent):
        """The `obs -> action` the eval loop calls, compiled where it can be.

        Only the AGENT half of `fusable` applies: eval reads the actor and
        writes nothing, and `Trainer._end_of_epoch` quiesces the learner around
        it, so which thread owns `agent.state` cannot matter. An agent without
        the pure `select_action` — a bare baseline — keeps `agent.step`.
        """
        eval_key = jax.random.PRNGKey(0)
        if not hasattr(agent, "select_action"):
            return lambda obs: agent.step(obs, evaluate=True, key=eval_key)

        if self._eval_act_fn is None:
            self._eval_act_fn = self._make_eval_act_fn()
        # Re-split per eval, not held across them: the point of an eval is to
        # score THIS epoch's weights, which the gradient bursts since the last
        # one updated in place through `burst_nodes`.
        acting_graphdef, acting_pytree = nnx.split(
            (agent.state.actor, agent.state.critic)
        )
        noise_nodes = getattr(agent, "_noise_nodes", None)
        noise_graphdef, noise_pytree = (
            noise_nodes.split() if noise_nodes is not None else (None, None)
        )
        # Exactly the statistics `agent.step` would have used: the live ones for
        # everyone, except PPO, which scores against the pin its own rollout ran
        # under. See `Agent.freeze_acting_norm`.
        frozen_stats = agent.freeze_acting_norm()
        obs_stats = (
            agent.state.obs_stats if frozen_stats is None else frozen_stats
        )

        def select(obs):
            return self._eval_act_fn(
                acting_graphdef, noise_graphdef, acting_pytree, noise_pytree,
                obs_stats, obs, eval_key,
            )

        return select

    def evaluate(self, agent):
        """Eval rollout: a Python loop until all episodes are done or
        max_episode_steps is reached.

        The loop is host-side on purpose, unlike `JaxRollout`'s compiled
        `while_loop`: the pool hands back numpy, so `np.all(dones)` is a free
        host read here rather than the pipeline-serializing device sync it would
        be there. Only the action selection inside it is compiled — see
        `_eval_select`.
        """
        test_env = self.test_environment
        max_steps = int(test_env.max_episode_steps or 1000)

        # `reset()` advances the pool's RNG, so consecutive evals would start
        # from different states; `reseed` rebuilds it at a fixed seed.
        reseed = getattr(test_env, "reseed", None)
        if reseed is not None:
            reseed(_EVAL_SEED)

        state, _ = test_env.reset()
        # From the reset, not trainer.test_episodes: a mismatch fails the
        # broadcast below.
        num_tests = int(state.obs.shape[0])
        scores = np.zeros(num_tests, dtype=np.float32)
        lengths = np.zeros(num_tests, dtype=np.int32)
        dones = np.zeros(num_tests, dtype=bool)
        start_obs = np.asarray(state.obs).reshape(num_tests, -1)

        select = self._eval_select(agent)

        for _ in range(max_steps):
            if np.all(dones):
                break
            actions = select(state.obs)
            state, timestep = test_env.step(state, actions)
            active = ~dones
            scores += np.asarray(timestep.reward) * active
            lengths += active.astype(np.int32)
            dones |= np.asarray(timestep.terminated | timestep.truncated)

        return scores, lengths, start_obs


def build_rollout(environment, test_environment, agent, num_envs, rngs,
                  test_episodes):
    """Pick the rollout for this environment. The only place the backend is
    named."""
    cls = (EnvPoolRollout if isinstance(environment, EnvPoolVectorEnv)
           else JaxRollout)
    return cls(environment, test_environment, agent, num_envs, rngs, test_episodes)
