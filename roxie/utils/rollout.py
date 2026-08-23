"""The backend-specific half of the training loop.

`Trainer._run` is a single loop that serves both backends. Everything that
genuinely differs between a vmapped JAX env (device-side physics, auto-reset by
gather from a pre-built pool, mining state threaded through `reset` as traced
arguments) and an EnvPool pool (C++ physics, auto-reset and mining owned by the
pool) lives behind the small surface below. The two loops this replaced were
~150 lines each and had already drifted apart in five places; the trainer now
has no idea which backend it is driving.

The surface is exactly the set of things that differed, nothing more:

    xp              array namespace for the trainer's per-step accumulation
    supports_async  whether an async learner can overlap acting and learning
    prepare()       compile/reset, return the first carry state
    warmup()        fill the replay buffer with random-action transitions
    step()          advance the envs by one step
    epoch_refresh() regenerate whatever is per-epoch (reset pool, mining table)
    mining_stats()  the negative-mining diagnostics, or None
    evaluate()      the held-out eval rollout

`step` returns `(carry, prev_env_state, next_env_state)`. `next_env_state` is
the *pre-auto-reset* state — the true next observation for the replay buffer —
while `carry` is what the next action is selected from, which for a done env is
a fresh start state. On the JAX path these are two distinct trees; EnvPool
resets in C++ and hands back only the post-reset observation, so there they are
the same object. That difference predates this module and is preserved: the
terminal transition's stored `next_obs` is the reset obs on the EnvPool path,
which is harmless because `terminal` zeroes the bootstrap for true
terminations.
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
from roxie.models.actors import deterministic_action

# Constant across epochs and runs so every eval rollout starts from the same
# states. Distinct from any training seed, so eval starts are never a subset of
# what the policy trained on.
_EVAL_SEED = 12345


def timed(fn, *args, label=""):
    """Run `fn`, blocking on its output, and print how long the compile took."""
    print(f"Compiling {label}...", flush=True)
    t0 = time.time()
    out = fn(*args)
    jax.block_until_ready(out)
    print(f"  {time.time() - t0:.1f}s", flush=True)
    return out


def _agent_replay_add(agent, buffer_state, transitions):
    """Add a (B, ...) batch via the agent's `replay_add` when it has one (DDPG/TD3
    insert the time axis their trajectory buffer expects); raw add otherwise."""
    fn = getattr(agent, "replay_add", None)
    return fn(buffer_state, transitions) if fn is not None \
        else agent.replay.add(buffer_state, transitions)


class JaxRollout:
    """Vmapped JAX environments: physics, auto-reset and mining all on device."""

    xp = jnp
    # The env step is GPU-bound here, so a background learner thread would
    # contend for the same device rather than overlap with it. Async learning
    # only pays off when acting is on the CPU (see EnvPoolRollout).
    supports_async = False

    def __init__(self, environment, test_environment, agent, num_envs, rngs,
                 test_episodes):
        self.environment = environment
        self.test_environment = test_environment
        self.agent = agent
        self.num_envs = num_envs
        self.num_test_episodes = int(test_episodes)
        self.rng = rngs.envs()
        self.action_size = environment.action_size
        self.action_low, self.action_high = agent.action_low, agent.action_high
        # The auto-reset pool is gathered from, so it need not match num_envs;
        # keeping them equal makes one reset program serve both.
        self.pool_size = num_envs
        self._eval_fn = None

        # Optional negative mining over start states (mocap). The env owns the
        # difficulty table; `mining_weights` is a TRACED reset argument so
        # refreshing it each epoch does not retrigger compilation.
        self.mining_env = getattr(environment, "unwrapped", environment)
        self.mining_on = (hasattr(self.mining_env, "mining_init")
                          and self.mining_env.mining_bins > 0)
        if self.mining_on:
            self.mining_weights, self.mining_counts = self.mining_env.mining_init()
            v_reset = jax.vmap(environment.reset, in_axes=(0, None))
            print(f"Negative mining ON: {self.mining_env.mining_bins} phase bins",
                  flush=True)
        else:
            self.mining_weights, self.mining_counts = None, None
            _plain_reset = jax.vmap(environment.reset)

            # Uniform signature at every call site; the weights are ignored.
            def v_reset(keys, _w=None):
                return _plain_reset(keys)

        self._v_step = jax.vmap(environment.step)
        self._jit_v_reset = jax.jit(v_reset)
        self._train_step = jax.jit(self._step_and_autoreset)

    # -- core ---------------------------------------------------------------

    def _step_and_autoreset(self, states, actions, rng, reset_pool, mining_counts):
        step_key, pool_key = jax.random.split(rng)
        new_states = self._v_step(states, actions)
        dones = new_states.env_state.done
        idx = jax.random.randint(pool_key, (self.num_envs,), 0, self.pool_size)

        # Here because this is the only place that sees every env's phase and
        # done flag on-device. `termination` is done-minus-truncation: clip-end
        # and step-limit cutoffs are not failures and must not be mined for.
        if mining_counts is not None:
            es = new_states.env_state
            mining_counts = self.mining_env.mining_observe(
                mining_counts, es.info, es.info["termination"]
            )

        def _autoreset_leaf(pool_leaf, s):
            # Leaves without a per-env leading dim (e.g. warp's world-flattened
            # contact arena) would be indexed out of bounds by the pool gather;
            # the physics recomputes them each step, so keep the stepped value.
            if not (isinstance(s, jnp.ndarray) and s.shape[:1] == dones.shape):
                return s
            return jnp.where(
                dones.reshape(dones.shape + (1,) * (s.ndim - 1)),
                pool_leaf[idx], s,
            )

        auto_states = jax.tree.map(_autoreset_leaf, reset_pool, new_states)
        return new_states, auto_states, mining_counts

    def _random_actions(self, key):
        u = jax.random.uniform(key, (self.num_envs, self.action_size))
        return self.action_low + (self.action_high - self.action_low) * u

    def _reset(self, n, key):
        return self._jit_v_reset(jax.random.split(key, n), self.mining_weights)

    # -- trainer-facing surface ---------------------------------------------

    def prepare(self):
        """Compile every program the loop will dispatch, then return a clean
        reset state. Doing it upfront keeps the first training iterations from
        being dominated by compiles that the throughput print would then blame
        on the physics."""
        agent = self.agent

        self.rng, reset_key, pool_key = jax.random.split(self.rng, 3)
        states = timed(self._reset, self.num_envs, reset_key, label="reset")
        self.reset_pool = timed(
            self._reset, self.pool_size, pool_key, label="reset pool",
        )

        dummy_actions = jnp.zeros((self.num_envs, self.action_size))
        timed(self._train_step, states, dummy_actions, self.rng, self.reset_pool,
              self.mining_counts, label="train step")

        # Off-policy replay add precompile. Skipped for on-policy agents (PPO),
        # whose buffer uses a different Transition layout and has no warmup.
        if getattr(agent, "memory_warmup", 0) > 0:
            # Build the dummy from the agent's OWN buffer prototype (leaves are
            # (add_batch, time, ...)) so this stays correct across per-agent
            # Transition layouts (e.g. DDPG/TD3 store `truncation`, SAC not).
            dummy_t = jax.tree.map(
                lambda leaf: jnp.zeros((self.num_envs,) + leaf.shape[2:], leaf.dtype),
                agent.state.buffer_state.experience,
            )
            timed(functools.partial(_agent_replay_add, agent),
                  agent.state.buffer_state, dummy_t, label="replay add")

        # Clean reset: the compile calls above stepped the states they were
        # handed, so training must not continue from them.
        self.rng, key = jax.random.split(self.rng)
        return self._reset(self.num_envs, key)

    def warmup(self, agent, iters, state):
        """Fill the replay buffer with `iters` scanned random-action steps.

        Scanned rather than looped so the whole fill is one dispatch, then added
        to the buffer in a single donated batch — a per-step Python loop here
        costs minutes at the step counts warmup needs.
        """
        @jax.jit
        def warmup_rollout(state, rng, reset_pool):
            def body(carry, _):
                state, rng = carry
                rng, act_key, step_key = jax.random.split(rng, 3)
                actions = self._random_actions(act_key)
                new_states, auto_states, _ = self._step_and_autoreset(
                    state, actions, step_key, reset_pool, None,
                )
                transition = Transition(
                    observation=state.env_state.obs,
                    action=actions,
                    reward=new_states.env_state.reward,
                    terminal=new_states.env_state.info["termination"],
                    # Only stored by agents whose prototype carries it
                    # (DDPG/TD3 n-step); pruned below for the others.
                    truncation=new_states.env_state.info["truncation"],
                )
                return (auto_states, rng), (transition, new_states.env_state.obs)
            return jax.lax.scan(body, (state, rng), None, length=iters)

        print("Compiling warmup rollout...", flush=True)
        t0 = time.time()
        compiled = warmup_rollout.lower(state, self.rng, self.reset_pool).compile()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Running warmup rollout...", flush=True)
        t0 = time.time()
        (state, self.rng), (transitions, next_obs) = compiled(
            state, self.rng, self.reset_pool,
        )
        state.env_state.obs.block_until_ready()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        # Prune to the fields the agent's buffer actually stores (SAC's
        # prototype has no `truncation`, DDPG/TD3's does) so the pytree
        # structures match at add time.
        proto = agent.state.buffer_state.experience
        transitions = Transition(**{
            f.name: (getattr(transitions, f.name)
                     if getattr(proto, f.name) is not None else None)
            for f in dataclasses.fields(Transition)
        })

        # Donate the buffer state: without it XLA keeps the input alive and
        # allocates a full output copy — a transient 2x of the buffer's obs
        # store that OOMs right here.
        @functools.partial(jax.jit, donate_argnums=(0,))
        def batch_add(buffer_state, transitions):
            def add_one(bs, t):
                return _agent_replay_add(agent, bs, t), None
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

        # The scanned fill bypasses `agent.add_transitions`, which is what
        # normally folds observations into the stats — so do it here, over the
        # same [prev, next] pair that path would have seen.
        if agent.normalize_observations:
            all_obs = jnp.concatenate([
                transitions.observation.reshape(-1, transitions.observation.shape[-1]),
                next_obs.reshape(-1, next_obs.shape[-1]),
            ], axis=0)
            agent.state.obs_stats = Agent.update_obs_stats(
                agent.state.obs_stats, all_obs,
            )

        return state, int(jnp.sum(transitions.terminal))

    def step(self, state, actions):
        self.rng, step_key = jax.random.split(self.rng)
        new_states, auto_states, self.mining_counts = self._train_step(
            state, actions, step_key, self.reset_pool, self.mining_counts,
        )
        return auto_states, state.env_state, new_states.env_state

    def epoch_refresh(self, state):
        """Regenerate the reset pool (fresh random starts for auto-reset) and
        reshuffle the GPU clip subset if the env supports it.

        Only a real clip swap invalidates in-progress episodes (their stored clip
        indices reference the old chunk), so only then are the live envs reset —
        reported back as `invalidated` so the trainer drops their part-scored
        episodes. Terminations fold into the start distribution first, so the new
        pool already reflects them.
        """
        if self.mining_on:
            self.mining_weights, self.mining_counts = self.mining_env.mining_refresh(
                self.mining_weights, self.mining_counts,
            )

        self.rng, pool_key = jax.random.split(self.rng)
        self.reset_pool = self._reset(self.pool_size, pool_key)

        swapped = (hasattr(self.environment, "swap_clips")
                   and bool(self.environment.swap_clips()))
        if swapped:
            self.rng, reset_key = jax.random.split(self.rng)
            state = self._reset(self.num_envs, reset_key)
        return state, swapped

    def mining_stats(self):
        if not self.mining_on:
            return None
        return self.mining_env.mining_stats(self.mining_weights, self.mining_counts)

    # -- eval ---------------------------------------------------------------

    def _make_eval_fn(self, v_step, num_tests, max_steps):
        """Build a single compiled eval rollout.

        The episode loop runs inside ``jax.lax.while_loop`` so the termination
        check is evaluated on-device — no per-step host sync, full GPU pipelining.
        A static ``max_steps`` cap bounds compute and guarantees termination.
        Actor / obs-stats are traced args, so one compile is reused every epoch.
        """
        agent = self.agent
        normalize = agent.normalize_observations

        @nnx.jit
        def eval_fn(actor, obs_stats, states):
            def cond(carry):
                i, _states, dones, _scores, _lengths = carry
                return (i < max_steps) & (~jnp.all(dones))

            def body(carry):
                i, states, dones, scores, lengths = carry

                obs = states.env_state.obs
                if normalize:
                    mean, std = Agent.obs_mean_std(obs_stats, agent.obs_eps)
                    obs = Agent.normalize_obs(obs, mean, std, agent.obs_clip)
                # Deterministic eval: actor output scaled to env units, no noise
                # module (its stateful update can't be mutated across the
                # while_loop trace level). A stochastic actor (PPO) returns a
                # distribution, so the mean is taken here — never a sample.
                action = jnp.clip(deterministic_action(actor(obs)), -1.0, 1.0)
                action = Agent.scale_to_env(action, agent.action_low, agent.action_high)

                next_states = v_step(states, action)
                not_done = ~dones
                scores = scores + next_states.env_state.reward * not_done.astype(jnp.float32)
                lengths = lengths + not_done.astype(jnp.int32)
                dones = jnp.logical_or(dones, next_states.env_state.done)
                return (i + 1, next_states, dones, scores, lengths)

            init = (
                jnp.int32(0),
                states,
                jnp.zeros((num_tests,), dtype=bool),
                jnp.zeros((num_tests,), dtype=jnp.float32),
                jnp.zeros((num_tests,), dtype=jnp.int32),
            )
            _, _, _, scores, lengths = jax.lax.while_loop(cond, body, init)
            return scores, lengths

        return eval_fn

    def evaluate(self, agent):
        num_tests = int(self.num_test_episodes)
        max_steps = int(getattr(self.test_environment, "max_episode_steps", 1000))

        if self._eval_fn is None:
            self._v_test_reset = jax.jit(jax.vmap(self.test_environment.reset))
            self._v_test_step = jax.jit(jax.vmap(self.test_environment.step))
            self._eval_fn = self._make_eval_fn(
                self._v_test_step, num_tests, max_steps,
            )

        # Fixed keys, not a fresh draw: the eval env pins its own start state, so
        # the keys only decide WHICH clip. Holding them constant means a change in
        # test/score is a change in the POLICY, not a different draw of clips.
        states = self._v_test_reset(
            jax.random.split(jax.random.PRNGKey(_EVAL_SEED), num_tests)
        )
        scores, lengths = self._eval_fn(
            agent.state.actor, agent.state.obs_stats, states,
        )
        start_obs = np.asarray(states.env_state.obs).reshape(num_tests, -1)
        return np.array(scores), np.array(lengths), start_obs


class EnvPoolRollout:
    """EnvPool pools: C++ physics, auto-reset and mining owned by the pool.

    No vmap/jit wrapping of the env step is needed — the pool batches natively —
    so the whole rollout is a plain Python loop and the arrays are already
    host-side. The agent still runs in JAX on whatever device is active.
    """

    xp = np
    # Acting is on the CPU here, so a background GPU learner genuinely overlaps.
    supports_async = True

    def __init__(self, environment, test_environment, agent, num_envs, rngs,
                 test_episodes):
        self.environment = environment
        self.test_environment = test_environment
        self.agent = agent
        self.num_envs = num_envs
        self.num_test_episodes = int(test_episodes)
        self.rng = rngs.envs()
        self.action_size = environment.action_size
        self.action_low, self.action_high = agent.action_low, agent.action_high

        # Only the location of the difficulty table differs from the JAX path: a
        # JAX env cannot own mutable state inside a trace, so there the trainer
        # threads `mining_weights` through reset, whereas a CPU pool resets in
        # plain Python and owns its own table.
        self.mining_on = (hasattr(environment, "mining_refresh")
                          and getattr(environment, "mining_bins", 0) > 0)
        if self.mining_on:
            print(f"Negative mining ON: {environment.mining_bins} phase bins",
                  flush=True)

    def _random_actions(self, key):
        u = jax.random.uniform(key, (self.num_envs, self.action_size))
        return self.action_low + (self.action_high - self.action_low) * u

    def prepare(self):
        print("Resetting environment...", flush=True)
        t0 = time.time()
        state = self.environment.reset()
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return state

    def warmup(self, agent, iters, state):
        """Fill the replay buffer with `iters` random-action steps.

        A plain loop, unlike the JAX path's scan: the pool steps in C++ and
        cannot be traced, so there is nothing to fuse. Episodes that finish here
        are counted — they are real env steps, and the JAX path counts them too.
        """
        episodes = 0
        t0 = time.time()
        for _ in range(iters):
            self.rng, act_key = jax.random.split(self.rng)
            actions = self._random_actions(act_key)
            agent.last_action = actions
            old_state = state
            state = self.environment.step(old_state, actions)
            agent.add(old_state.env_state, state.env_state)
            episodes += int(np.sum(np.asarray(state.env_state.done)))
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return state, episodes

    def step(self, state, actions):
        new_state = self.environment.step(state, actions)
        # The pool auto-resets in C++, so there is no separate pre-reset tree:
        # the stepped state serves as both the buffer's next state and the next
        # action's input. See the module docstring.
        return new_state, state.env_state, new_state.env_state

    def epoch_refresh(self, state):
        if self.mining_on:
            self.environment.mining_refresh()
        # Regenerate the auto-reset pool AFTER the mining refresh, so the new
        # pool reflects this epoch's terminations.
        refresh_pool = getattr(self.environment, "refresh_reset_pool", None)
        if refresh_pool is not None:
            refresh_pool()
        # The pool keeps stepping its live envs across the refresh, so no
        # in-progress episode is invalidated.
        return state, False

    def mining_stats(self):
        return self.environment.mining_stats() if self.mining_on else None

    def evaluate(self, agent):
        """Eval rollout: a plain Python loop until all episodes are done or
        max_episode_steps is reached."""
        test_env = self.test_environment
        max_steps = int(getattr(test_env, "max_episode_steps", 1000))

        # EnvPool's `reset()` advances the pool's RNG, so consecutive evals would
        # otherwise start from different states and a change in `test/score`
        # could be a different draw rather than a better policy. `reseed` rebuilds
        # the pool at a fixed seed (~3ms), matching the JAX path's fixed eval
        # keys. Envs whose eval reset is already deterministic (mocap pins frame 0
        # with no reset noise) are unaffected either way.
        reseed = getattr(test_env, "reseed", None)
        if reseed is not None:
            reseed(_EVAL_SEED)

        state = test_env.reset()
        # Size the eval buffers from the pool the reset actually returns, not from
        # trainer.test_episodes: env.test_episodes and trainer.test_episodes are
        # separate config keys and a mismatch would fail the broadcast below.
        num_tests = int(np.asarray(state.env_state.reward).shape[0])
        scores = np.zeros(num_tests, dtype=np.float32)
        lengths = np.zeros(num_tests, dtype=np.int32)
        dones = np.zeros(num_tests, dtype=bool)
        start_obs = np.asarray(state.env_state.obs).reshape(num_tests, -1)

        # Fixed key for eval (noise is bypassed when evaluate=True).
        eval_key = jax.random.PRNGKey(0)

        for _ in range(max_steps):
            if np.all(dones):
                break
            actions = agent.step(state.env_state.obs, evaluate=True, key=eval_key)
            state = test_env.step(state, actions)
            active = ~dones
            scores += np.array(state.env_state.reward) * active
            lengths += active.astype(np.int32)
            dones |= np.array(state.env_state.done)

        return scores, lengths, start_obs


def build_rollout(environment, test_environment, agent, num_envs, rngs,
                  test_episodes):
    """Pick the rollout for this environment. The only place the backend is
    named; everything downstream goes through the common surface."""
    from roxie.environment.envpool_adapter import EnvPoolWrapper

    cls = EnvPoolRollout if isinstance(environment, EnvPoolWrapper) else JaxRollout
    return cls(environment, test_environment, agent, num_envs, rngs, test_episodes)
