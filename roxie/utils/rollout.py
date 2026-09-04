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
dispatch, and the EnvPool rollout, whose physics runs in C++ and cannot be
traced, runs the same body as a Python loop. Both return the same per-chunk
episode summary, which is what keeps one trainer loop over both.

Measured on that machine, end to end: 22.9k -> 103k env steps/s.

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

    def _fusable(self) -> bool:
        """Can this agent's acting and buffering be traced?

        The same capability test `build_learner` already uses for the async
        learner: an agent qualifies by exposing PURE `select_action` and
        `buffer_transitions` — ones that take the actor, the obs stats and the
        train state explicitly instead of reading `self.state`, so they can run
        against a `lax.scan` carry. Every off-policy agent does — DDPG (and
        TD3/TD4/D4PG), SAC and MPO; PPO does not, and falls back to the per-step
        loop until it does.
        """
        return all(
            hasattr(self.agent, name)
            for name in ("select_action", "buffer_transitions")
        )

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

        One dispatch for the whole chunk on the fused path. `learner` is unused
        there — `supports_async` is False for this backend, so the learner is
        always the synchronous one and `agent.state` is not shared with another
        thread — and drives the per-step fallback for agents that cannot be
        traced.
        """
        if not self._fusable():
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
                action = jnp.clip(deterministic_action(actor(obs)), -1.0, 1.0)
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

    The pool batches natively, so no vmap/jit wrapping is needed: the rollout is
    a plain Python loop over host-side arrays. The agent still runs in JAX.
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

    def collect(self, state, n_steps, learner):
        """Advance the pool `n_steps` times, buffering and scoring as it goes.

        A plain Python loop: the physics runs in C++ and cannot be traced, so
        there is nothing to fuse — and going through the learner per step is
        what lets the async learner overlap those C++ steps with GPU gradient
        bursts, which is this backend's whole reason for supporting it.
        """
        return stepwise_collect(self, state, n_steps, learner)

    def epoch_refresh(self, state):
        refresh = getattr(self.environment, "epoch_refresh", None)
        return state, (bool(refresh()) if refresh is not None else False)

    def evaluate(self, agent):
        """Eval rollout: a plain Python loop until all episodes are done or
        max_episode_steps is reached."""
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


def build_rollout(environment, test_environment, agent, num_envs, rngs,
                  test_episodes):
    """Pick the rollout for this environment. The only place the backend is
    named."""
    cls = (EnvPoolRollout if isinstance(environment, EnvPoolVectorEnv)
           else JaxRollout)
    return cls(environment, test_environment, agent, num_envs, rngs, test_episodes)
