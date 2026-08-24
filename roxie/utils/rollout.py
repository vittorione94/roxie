"""The backend-specific half of the training loop.

`Trainer._run` is a single loop that serves both backends. Everything that
genuinely differs between a vmapped JAX env (device-side physics, auto-reset by
gather from a pre-built pool, env `params` threaded through the trainer because
a traced env cannot own mutable state) and a C++ pool (native physics,
auto-reset and any such state owned by the pool itself) lives behind the small
surface below. The trainer has no idea which backend it is driving.

    xp              array namespace for the trainer's per-step accumulation
    supports_async  whether an async learner can overlap acting and learning
    prepare()       compile/reset, return the first carry state
    warmup()        fill the replay buffer with random-action transitions
    step()          advance the envs by one step
    epoch_refresh() the epoch boundary: let the env refresh itself, rebuild the
                    reset pool, report whether live episodes were invalidated
    evaluate()      the held-out eval rollout

`step` returns `(state, prev_obs, timestep)`. `prev_obs` is the observation the
action was selected from; `timestep` is the Gymnasium 5-tuple with PRE-auto-reset
values — the true next observation for the replay buffer — while `state.obs` is
what the NEXT action is selected from, which for a done env is a fresh start.
On the JAX path those are two distinct arrays; a C++ pool resets internally and
hands back only the post-reset observation, so there they are the same object.
That difference predates this module and is preserved: the terminal transition's
stored `next_obs` is the reset obs on the pool path, which is harmless because
`terminated` zeroes the bootstrap for true terminations.

The two classes stay two classes on purpose. One traces its whole step into XLA
and scans its warmup into a single dispatch; the other steps C++ from a Python
loop with the arrays already host-side. That is irreducible — what they now
share is the env protocol, not the loop.
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
    """Vmapped JAX environments: physics and auto-reset all on device."""

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
        self.action_size = space_size(environment.single_action_space)
        self.action_low, self.action_high = agent.action_low, agent.action_high
        # The auto-reset pool is gathered from, so it need not match num_envs;
        # keeping them equal makes one reset program serve both.
        self.pool_size = num_envs
        self._eval_fn = None

        # `params` for an env that adapts its own (see FuncEnv.init_params);
        # None for every other env. It travels as the FuncEnv `params` argument,
        # which is TRACED — so refreshing it each epoch does not retrigger a
        # compile of the reset. The three hooks are bound once, defaulted to
        # no-ops, so nothing below branches on whether this env has them.
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

    # -- core ---------------------------------------------------------------

    def _step_and_observe(self, state, actions, rng, reset_pool, params):
        """One env step plus the env's own `params` update, fused into one
        dispatch.

        The update lives here rather than in the driver because this is the only
        place that sees every env's info and termination flag on device at once.
        It passes `terminated` — done MINUS truncation — so a non-failure cutoff
        is never presented to the env as a failure.
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

    # -- trainer-facing surface ---------------------------------------------

    def prepare(self):
        """Compile every program the loop will dispatch, then return a clean
        reset state. Doing it upfront keeps the first training iterations from
        being dominated by compiles that the throughput print would then blame
        on the physics."""
        agent = self.agent

        self.rng, reset_key, pool_key = jax.random.split(self.rng, 3)
        state = timed(self._reset, self.num_envs, reset_key, label="reset")
        self.reset_pool = timed(
            self._reset, self.pool_size, pool_key, label="reset pool",
        )

        dummy_actions = jnp.zeros((self.num_envs, self.action_size))
        timed(self._train_step, state, dummy_actions, self.rng, self.reset_pool,
              self.params, label="train step")

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
        def warmup_rollout(state, rng, reset_pool, params):
            def body(carry, _):
                state, rng = carry
                rng, act_key, step_key = jax.random.split(rng, 3)
                actions = self._random_actions(act_key)
                prev_obs = state.obs
                # `observe_params` is deliberately NOT called here: these are
                # random-action terminations, and an env adapting itself to them
                # would be adapting to failures no policy caused. `params` still
                # travels, since the env is entitled to read it in `transition`.
                state, timestep = self.environment.step(
                    state, actions, step_key, reset_pool, params,
                )
                transition = Transition(
                    observation=prev_obs,
                    action=actions,
                    reward=timestep.reward,
                    terminal=timestep.terminated,
                    # Only stored by agents whose prototype carries it
                    # (DDPG/TD3 n-step); pruned below for the others.
                    truncation=timestep.truncated,
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
        prev_obs = state.obs
        state, timestep, self.params = self._train_step(
            state, actions, step_key, self.reset_pool, self.params,
        )
        return state, prev_obs, timestep

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

    # -- eval ---------------------------------------------------------------

    def _make_eval_fn(self, num_tests, max_steps):
        """Build a single compiled eval rollout.

        The episode loop runs inside ``jax.lax.while_loop`` so the termination
        check is evaluated on-device — no per-step host sync, full GPU pipelining.
        A static ``max_steps`` cap bounds compute and guarantees termination.
        Actor / obs-stats are traced args, so one compile is reused every epoch.
        """
        agent = self.agent
        normalize = agent.normalize_observations
        test_env = self.test_environment
        # Deterministic: the env's own stochasticity rides in its state, so a
        # fixed key here makes consecutive evals of the same policy identical.
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
                # Deterministic eval: actor output scaled to env units, no noise
                # module (its stateful update can't be mutated across the
                # while_loop trace level). A stochastic actor (PPO) returns a
                # distribution, so the mean is taken here — never a sample.
                action = jnp.clip(deterministic_action(actor(obs)), -1.0, 1.0)
                action = Agent.scale_to_env(action, agent.action_low, agent.action_high)

                # reset_pool=None: no auto-reset. Each episode runs to its own
                # end and finished worlds are masked out below.
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
            # Jitted, not eager: this runs once per epoch and an eager vmapped
            # reset re-dispatches the whole physics op by op every time.
            self._jit_test_reset = jax.jit(self.test_environment.reset)
            self._eval_fn = self._make_eval_fn(num_tests, max_steps)

        # Fixed key, not a fresh draw: the eval env pins its own start state, so
        # the key only decides WHICH clip. Holding it constant means a change in
        # test/score is a change in the POLICY, not a different draw of clips.
        state, _ = self._jit_test_reset(jax.random.PRNGKey(_EVAL_SEED))
        scores, lengths = self._eval_fn(
            agent.state.actor, agent.state.obs_stats, state,
        )
        start_obs = np.asarray(state.obs).reshape(num_tests, -1)
        return np.array(scores), np.array(lengths), start_obs


class EnvPoolRollout:
    """C++ pools: native physics, auto-reset and any adaptive state owned by
    the pool itself.

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
            prev_obs = state.obs
            state, timestep = self.environment.step(state, actions)
            agent.add(prev_obs, timestep)
            episodes += int(np.sum(timestep.terminated | timestep.truncated))
        jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return state, episodes

    def step(self, state, actions):
        prev_obs = state.obs
        # The pool auto-resets in C++, so there is no separate pre-reset
        # observation: `timestep.obs` and `state.obs` are the same array. See
        # the module docstring.
        state, timestep = self.environment.step(state, actions)
        return state, prev_obs, timestep

    def epoch_refresh(self, state):
        # There is no `params` to thread here: a pool owns whatever it
        # regenerates per epoch (its own auto-reset pool, a start distribution
        # it adapts) and does the lot inside this one call. Nothing is
        # invalidated unless the pool says so — it keeps stepping its live envs
        # across the refresh.
        refresh = getattr(self.environment, "epoch_refresh", None)
        return state, (bool(refresh()) if refresh is not None else False)

    def evaluate(self, agent):
        """Eval rollout: a plain Python loop until all episodes are done or
        max_episode_steps is reached."""
        test_env = self.test_environment
        max_steps = int(test_env.max_episode_steps or 1000)

        # A pool's `reset()` advances its RNG, so consecutive evals would
        # otherwise start from different states and a change in `test/score`
        # could be a different draw rather than a better policy. `reseed`
        # rebuilds the pool at a fixed seed (~3ms), matching the JAX path's
        # fixed eval keys. Envs whose eval reset is already deterministic
        # (mocap pins frame 0 with no reset noise) are unaffected either way.
        reseed = getattr(test_env, "reseed", None)
        if reseed is not None:
            reseed(_EVAL_SEED)

        state, _ = test_env.reset()
        # Size the eval buffers from the pool the reset actually returns, not
        # from trainer.test_episodes: a mismatch would fail the broadcast below.
        num_tests = int(state.obs.shape[0])
        scores = np.zeros(num_tests, dtype=np.float32)
        lengths = np.zeros(num_tests, dtype=np.int32)
        dones = np.zeros(num_tests, dtype=bool)
        start_obs = np.asarray(state.obs).reshape(num_tests, -1)

        # Fixed key for eval (noise is bypassed when evaluate=True).
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
    named; everything downstream goes through the common surface."""
    cls = (EnvPoolRollout if isinstance(environment, EnvPoolVectorEnv)
           else JaxRollout)
    return cls(environment, test_environment, agent, num_envs, rngs, test_episodes)
