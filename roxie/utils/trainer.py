import dataclasses
import functools
import os
import shutil
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from roxie.utils import logger
from roxie.agents.agent import Agent
from roxie.agents.utils import Transition
from roxie.models.actors import deterministic_action

# Constant across epochs and runs so every eval rollout starts from the same
# states. Distinct from any training seed, so eval starts are never a subset of
# what the policy trained on.
_EVAL_SEED = 12345


def _agent_replay_add(agent, buffer_state, transitions):
    """Add a (B, ...) batch via the agent's `replay_add` when it has one (DDPG/TD3
    insert the time axis their trajectory buffer expects); raw add otherwise."""
    fn = getattr(agent, "replay_add", None)
    return fn(buffer_state, transitions) if fn is not None \
        else agent.replay.add(buffer_state, transitions)


def _new_epoch_acc():
    """Per-epoch accumulators, reset at every epoch boundary.

    `ret`/`len` collect only episodes that actually terminated; their `_sq`
    companions recover the per-episode std at the boundary via
    sqrt(E[x^2] - E[x]^2), avoiding a host-side list (and a per-step sync).
    `metrics` sums the env's own metric dict, keys discovered lazily. `noise`
    needs its own counter: it is added on policy steps only, not during warmup.
    Zeros are plain floats — jnp promotes them on first use.
    """
    return dict(
        ret=0.0, ret_sq=0.0, len=0.0, len_sq=0.0, count=0.0,
        metrics={}, metric_iters=0, noise=0.0, noise_iters=0,
    )


class Trainer:

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(3e5), save_steps=int(1e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
        async_learner=False, learner_chunk=8, save_buffer=False, resume=None,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.output_dir = output_dir
        # Off by default: the buffer dominates checkpoint size and the host-RAM
        # cost of writing it. On, a resumed off-policy run skips warmup.
        self.save_buffer = bool(save_buffer)
        # Resume counters keep a restarted run on the SAME curve: env steps keyed
        # identically, save cadence and step budget measured against the total.
        resume = resume or {}
        (self.initial_steps, self.initial_epochs, self.initial_episodes,
         self.initial_gradient_steps) = (
            int(resume.get(k) or 0)
            for k in ("steps", "epochs", "episodes", "gradient_steps")
        )
        # Only set when the run that wrote the checkpoint had `save_buffer`;
        # otherwise a resumed agent would sample a buffer of zeros.
        self.skip_warmup = bool(resume.get("buffer_restored", False))
        # Background learner thread overlapping CPU acting with GPU gradient
        # bursts (envpool loop only). `learner_chunk` = fused grad steps per
        # dispatch: smaller overlaps better, larger amortizes dispatch overhead.
        self.async_learner = async_learner
        self.learner_chunk = int(learner_chunk)

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def _warmup_iters(self, memory_warmup, num_envs):
        """Zero when a resume restored a populated buffer — refilling would
        prepend a block of RANDOM-action transitions to a trained policy's data."""
        if self.skip_warmup:
            print("Resumed with a restored replay buffer: skipping warmup.",
                  flush=True)
            return 0
        return memory_warmup // num_envs

    def _save(self, agent, epochs, episodes, gradient_steps):
        """Write a checkpoint, tagged with the progress needed to resume it."""
        path = os.path.join(self.output_dir, "checkpoints")
        if os.path.isdir(path) and self.replace_checkpoint:
            for file in os.listdir(path):
                if file.startswith("step_"):
                    # A checkpoint is a DIRECTORY (orbax writes a tree), so this
                    # needs rmtree — `os.remove` raises IsADirectoryError.
                    shutil.rmtree(os.path.join(path, file), ignore_errors=True)
        agent.save(
            os.path.join(path, f"step_{self.steps}"),
            include_buffer=self.save_buffer,
            extra_metadata={
                "steps": int(self.steps),
                "epochs": int(epochs),
                "episodes": int(episodes),
                "gradient_steps": int(gradient_steps),
            },
        )

    def _timed(self, fn, *args, label=""):
        print(f"Compiling {label}...", flush=True)
        t0 = time.time()
        out = fn(*args)
        jax.block_until_ready(out)
        print(f"  {time.time() - t0:.1f}s", flush=True)
        return out

    def run(self, NUM_ENVS, rngs):
        """Dispatch to the appropriate training loop based on env backend."""
        from roxie.environment.envpool_adapter import EnvPoolWrapper
        if isinstance(self.environment, EnvPoolWrapper):
            self._run_envpool(NUM_ENVS, rngs)
        else:
            self._run_jax(NUM_ENVS, rngs)

    def _run_jax(self, NUM_ENVS, rngs):

        start_time = last_epoch_time = time.time()
        agent = self.agent
        # Optional negative mining over start states (mocap). The env owns the
        # difficulty table; `mining_weights` is a TRACED reset argument so
        # refreshing it each epoch does not retrigger compilation.
        mining_env = self.environment.unwrapped if hasattr(
            self.environment, "unwrapped"
        ) else self.environment
        mining_on = hasattr(mining_env, "mining_init") and mining_env.mining_bins > 0
        if mining_on:
            mining_weights, mining_counts = mining_env.mining_init()
            v_reset = jax.vmap(self.environment.reset, in_axes=(0, None))
            print(f"Negative mining ON: {mining_env.mining_bins} phase bins", flush=True)
        else:
            mining_weights, mining_counts = None, None
            _plain_reset = jax.vmap(self.environment.reset)
            # Uniform signature at every call site; the weights are ignored.
            def v_reset(keys, _w=None):
                return _plain_reset(keys)
        v_step = jax.vmap(self.environment.step)
        action_size = self.environment.action_size
        action_low, action_high = agent.action_low, agent.action_high
        POOL_SIZE = NUM_ENVS

        # --- Core: step envs, auto-reset done ones from a pre-built pool ---

        def _step_and_autoreset(states, actions, rng, reset_pool, mining_counts):
            step_key, pool_key = jax.random.split(rng)
            new_states = v_step(states, actions)
            dones = new_states.env_state.done
            idx = jax.random.randint(pool_key, (NUM_ENVS,), 0, POOL_SIZE)

            # Here because this is the only place that sees every env's phase and
            # done flag on-device. `termination` is done-minus-truncation: clip-end
            # and step-limit cutoffs are not failures and must not be mined for.
            if mining_counts is not None:
                es = new_states.env_state
                mining_counts = mining_env.mining_observe(
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

        def _random_actions(key):
            u = jax.random.uniform(key, (NUM_ENVS, action_size))
            return action_low + (action_high - action_low) * u

        # --- JIT wrappers ---

        @jax.jit
        def train_step(states, actions, rng, reset_pool, mining_counts):
            return _step_and_autoreset(
                states, actions, rng, reset_pool, mining_counts
            )

        jit_v_reset = jax.jit(v_reset)

        v_test_reset = jax.jit(jax.vmap(self.test_environment.reset))
        v_test_step = jax.jit(jax.vmap(self.test_environment.step))

        # --- Compile everything upfront ---

        loop_rng = rngs.envs()

        reset_keys = jax.random.split(loop_rng, NUM_ENVS)
        wrapped_states = self._timed(
            jit_v_reset, reset_keys, mining_weights, label="reset"
        )

        # Initial reset pool (reused for auto-reset via gather)
        loop_rng, pool_rng = jax.random.split(loop_rng)
        reset_pool = self._timed(
            jit_v_reset, jax.random.split(pool_rng, POOL_SIZE), mining_weights,
            label="reset pool",
        )

        dummy_actions = jnp.zeros((NUM_ENVS, action_size))
        self._timed(
            train_step, wrapped_states, dummy_actions, loop_rng, reset_pool,
            mining_counts, label="train step",
        )

        self._timed(
            functools.partial(
                agent.step, evaluate=False, key=loop_rng,
            ), wrapped_states.env_state.obs, label="agent step",
        )

        # Off-policy replay add precompile. Skipped for on-policy agents (PPO),
        # whose buffer uses a different Transition layout and has no warmup.
        if getattr(agent, "memory_warmup", 0) > 0:
            # Build the dummy from the agent's OWN buffer prototype (leaves are
            # (add_batch, time, ...)) so this stays correct across per-agent
            # Transition layouts (e.g. DDPG/TD3 store `truncation`, SAC not).
            dummy_t = jax.tree.map(
                lambda leaf: jnp.zeros((NUM_ENVS,) + leaf.shape[2:], leaf.dtype),
                agent.state.buffer_state.experience,
            )
            self._timed(
                functools.partial(_agent_replay_add, agent),
                agent.state.buffer_state, dummy_t, label="replay add",
            )

        # Clean reset
        loop_rng, rng = jax.random.split(loop_rng)
        wrapped_states = jit_v_reset(jax.random.split(rng, NUM_ENVS), mining_weights)

        # --- Warmup: scan with random actions ---

        agent_key = rngs.agent()
        memory_warmup = getattr(agent, 'memory_warmup', 0)
        warmup_iters = self._warmup_iters(memory_warmup, NUM_ENVS)
        # A resumed run continues its predecessor's counters (all zero for a fresh
        # one), so `steps` keeps meaning TOTAL env steps: epoch and save cadences
        # stay phase-aligned across the restart and the logs continue one curve.
        self.steps = self.initial_steps
        epochs, episodes = self.initial_epochs, self.initial_episodes
        tot_gradient_steps = self.initial_gradient_steps
        epoch_steps = self.steps % self.epoch_steps
        steps_since_save = self.steps % self.save_steps
        actor_losses, critic_losses = [], []

        if warmup_iters > 0:
            print(f"Warmup: {warmup_iters} iters ({memory_warmup:,} steps)...", flush=True)

            @jax.jit
            def warmup_rollout(state, rng, reset_pool):
                def body(carry, _):
                    state, rng = carry
                    rng, act_key, step_key = jax.random.split(rng, 3)
                    actions = _random_actions(act_key)
                    new_states, auto_states, _ = _step_and_autoreset(
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
                return jax.lax.scan(body, (state, rng), None, length=warmup_iters)

            print("Compiling warmup rollout...", flush=True)
            t0 = time.time()
            _compiled_warmup = warmup_rollout.lower(
                wrapped_states, loop_rng, reset_pool,
            ).compile()
            print(f"  {time.time() - t0:.1f}s", flush=True)

            print("Running warmup rollout...", flush=True)
            t0 = time.time()
            (wrapped_states, loop_rng), (transitions, next_obs) = _compiled_warmup(
                wrapped_states, loop_rng, reset_pool,
            )
            wrapped_states.env_state.obs.block_until_ready()
            print(f"  {time.time() - t0:.1f}s", flush=True)

            # Prune to the fields the agent's buffer actually stores (SAC's
            # prototype has no `truncation`, DDPG/TD3's does) so the pytree
            # structures match at add time.
            proto = agent.state.buffer_state.experience
            transitions = Transition(**{
                f.name: (
                    getattr(transitions, f.name)
                    if getattr(proto, f.name) is not None else None
                )
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
            _compiled_batch_add = batch_add.lower(
                agent.state.buffer_state, transitions,
            ).compile()
            print(f"  {time.time() - t0:.1f}s", flush=True)

            print("Running replay fill...", flush=True)
            t0 = time.time()
            agent.state.buffer_state = _compiled_batch_add(
                agent.state.buffer_state, transitions,
            )
            jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
            print(f"  {time.time() - t0:.1f}s", flush=True)

            if agent.normalize_observations:
                all_obs = jnp.concatenate([
                    transitions.observation.reshape(-1, transitions.observation.shape[-1]),
                    next_obs.reshape(-1, next_obs.shape[-1]),
                ], axis=0)
                agent.state.obs_stats = Agent.update_obs_stats(agent.state.obs_stats, all_obs)

            # Warmup transitions are real env steps, so they count — on a resume
            # they add on top of the restored total rather than replacing it.
            self.steps = self.initial_steps + warmup_iters * NUM_ENVS
            epoch_steps = self.steps % self.epoch_steps
            steps_since_save = self.steps % self.save_steps
            episodes = episodes + int(jnp.sum(transitions.terminal))
            print(f"  Done: {self.steps:,} steps, {episodes} episodes", flush=True)

        # Precompile the gradient step (buffer is now full of warmup data).
        if hasattr(agent, "update") and hasattr(agent, "steps_before_learning"):
            print("Compiling gradient step...", flush=True)
            t0 = time.time()
            agent_key, warm_key = jax.random.split(agent_key)
            agent.update(steps=agent.steps_before_learning, agent_rng=warm_key)
            jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
            # Hand back the boundary that call consumed, so the loop runs the full
            # schedule and the realized replay ratio is the one the config asks for.
            agent._last_update_boundary = -1
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # --- Training loop ---

        print("Training...", flush=True)
        # In-flight per-env counters; `acc` collects only completed episodes.
        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
        acc = _new_epoch_acc()
        bench_t0, bench_steps = time.time(), 0

        while True:
            loop_rng, action_key, step_key = jax.random.split(loop_rng, 3)

            use_agent_policy = (
                not hasattr(agent, "memory_warmup")
                or self.steps > getattr(agent, "memory_warmup", 0)
            )
            if use_agent_policy:
                actions = agent.step(
                    wrapped_states.env_state.obs, evaluate=False, key=action_key,
                )
                # Mean |noise| per joint in normalized action units. Only
                # deterministic agents (DDPG/TD3) expose last_noise.
                last_noise = getattr(agent, "last_noise", None)
                if last_noise is not None:
                    acc["noise"] += jnp.mean(jnp.abs(last_noise))
                    acc["noise_iters"] += 1
            else:
                actions = _random_actions(action_key)
                agent.last_action = actions

            old_wrapped_states = wrapped_states
            new_wrapped_states, wrapped_states, mining_counts = train_step(
                old_wrapped_states, actions, step_key, reset_pool, mining_counts,
            )

            agent_key, update_key = jax.random.split(agent_key)
            agent.add(old_wrapped_states.env_state, new_wrapped_states.env_state)

            # Track the CURRENT policy's state distribution rather than freezing
            # after warmup. Obs are stored raw and normalized at sample time, so
            # updated stats stay consistent for old and new data alike.
            if getattr(agent, "normalize_observations", False):
                agent.state.obs_stats = Agent.update_obs_stats(
                    agent.state.obs_stats, new_wrapped_states.env_state.obs,
                )

            # The agent gates its own updates. The trainer must NOT read buffer
            # device state here: a Python branch on a device array forces a
            # blocking host sync every iteration and serializes the GPU pipeline.
            gradient_steps, actor_loss, critic_loss = agent.update(
                steps=self.steps, agent_rng=update_key,
            )
            if gradient_steps > 0:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)
            tot_gradient_steps += gradient_steps

            env_state = new_wrapped_states.env_state
            scores += env_state.reward
            lengths += 1
            # Per-env vectors; the mean is deferred to the epoch boundary to avoid
            # one dispatch per metric per step.
            for k, v in env_state.metrics.items():
                acc["metrics"][k] = acc["metrics"].get(k, 0.0) + v
            acc["metric_iters"] += 1

            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS

            done = env_state.done.astype(jnp.float32)
            done_sum = jnp.sum(done)  # once — reused for `episodes` and `count`
            episodes = episodes + done_sum
            # `scores`/`lengths` already include the terminal transition and are
            # zeroed for done envs below; non-done envs contribute 0 here.
            ep_ret, ep_len = scores * done, lengths.astype(jnp.float32) * done
            acc["ret"] += jnp.sum(ep_ret)
            acc["ret_sq"] += jnp.sum(ep_ret ** 2)
            acc["len"] += jnp.sum(ep_len)
            acc["len_sq"] += jnp.sum(ep_len ** 2)
            acc["count"] += done_sum

            if bench_steps == NUM_ENVS * 20:
                wrapped_states.env_state.obs.block_until_ready()
                elapsed = time.time() - bench_t0
                print(
                    f"\n  Speed: {bench_steps / elapsed:.0f} steps/s "
                    f"({elapsed / 20 * 1000:.1f}ms/iter)",
                    flush=True,
                )

            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            if epoch_steps >= self.epoch_steps:
                if hasattr(agent, "noise_module"):
                    agent.noise_module.reset_noise()

                if self.test_environment and hasattr(agent, "state"):
                    self._test(rngs.envs(), v_test_reset, v_test_step)

                epochs += 1
                epoch_steps = 0

                self._store_epoch_metrics(
                    agent, self._epoch_stats(acc, scores, lengths), acc,
                    epochs=epochs, episodes=episodes,
                    sps=self.steps / (time.time() - start_time),
                    gradient_steps=tot_gradient_steps,
                    losses=(actor_losses, critic_losses),
                    times=(start_time, last_epoch_time),
                    # Read before the refresh below, which zeroes the counters.
                    mining_stats=(mining_env.mining_stats(
                        mining_weights, mining_counts
                    ) if mining_on else None),
                )
                logger.dump(step=self.steps)

                actor_losses, critic_losses = [], []
                acc = _new_epoch_acc()

                # Regenerate the reset pool (fresh random starts for auto-reset)
                # and reshuffle the GPU clip subset if the env supports it. Only a
                # real clip swap invalidates in-progress episodes (their stored
                # clip indices reference the old chunk), so only then are the live
                # envs reset. Terminations fold into the start distribution first,
                # so the new pool already reflects them.
                if mining_on:
                    mining_weights, mining_counts = mining_env.mining_refresh(
                        mining_weights, mining_counts
                    )

                loop_rng, pool_rng = jax.random.split(loop_rng)
                reset_pool = jit_v_reset(jax.random.split(pool_rng, POOL_SIZE), mining_weights)
                swapped = (hasattr(self.environment, 'swap_clips')
                           and bool(self.environment.swap_clips()))
                if swapped:
                    loop_rng, reset_rng = jax.random.split(loop_rng)
                    wrapped_states = jit_v_reset(
                        jax.random.split(reset_rng, NUM_ENVS), mining_weights
                    )
                    scores = jnp.zeros(NUM_ENVS)
                    lengths = jnp.zeros(NUM_ENVS, dtype=jnp.int32)

                last_epoch_time = time.time()

            scores = jnp.where(env_state.done, 0, scores)
            lengths = jnp.where(env_state.done, 0, lengths)

            stop_training = self.steps >= self.max_steps
            if stop_training or steps_since_save >= self.save_steps:
                self._save(agent, epochs, episodes, tot_gradient_steps)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

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

    @staticmethod
    def _epoch_stats(acc, scores, lengths):
        """(n, mean/std of return, mean/std of length) over episodes that
        COMPLETED this epoch, falling back to the in-flight per-env counters when
        none did (episodes longer than an epoch) so the line is never blank."""
        n = int(acc["count"])
        if n == 0:
            return (n, float(np.mean(scores)), float(np.std(scores)),
                    float(np.mean(lengths)), float(np.std(lengths)))
        # Divide by the accumulator, not by `n`: that keeps the arithmetic in the
        # accumulator's own dtype (f32 on the jax path) instead of widening first.
        mean_ret = float(acc["ret"] / acc["count"])
        mean_len = float(acc["len"] / acc["count"])

        # Population std, clamped against tiny negative variances from float
        # round-off in the E[x^2] - E[x]^2 subtraction.
        def std(sq, mean):
            return float(np.sqrt(max(float(acc[sq] / acc["count"]) - mean ** 2, 0.0)))

        return (n, mean_ret, std("ret_sq", mean_ret),
                mean_len, std("len_sq", mean_len))

    def _store_epoch_metrics(
        self, agent, stats, acc, *, epochs, episodes, sps, gradient_steps,
        losses, times, mining_stats=None,
    ):
        """Store one epoch's metrics under the shared namespace scheme.

        Both training loops funnel through here so they cannot drift into logging
        different key sets for the same quantities. Every metric is namespaced by
        its producer: ``epoch``/``steps`` are the run axes, ``train/*`` the
        behaviour policy and learner, ``test/*`` the held-out eval (see `_test`),
        ``sys/*`` throughput and host/device health. Anything a producer already
        prefixes for itself (``reward/``, ``td3/``, ``ppo/``, ``mining/``) nests
        UNDER ``train/`` — this is the only place that knows the namespace.
        """
        ep_n, score, score_std, length, length_std = stats
        actor_losses, critic_losses = losses
        start_time, last_epoch_time = times

        logger.store("epoch", epochs)
        logger.store("steps", self.steps)

        logger.store("train/score", score)
        logger.store("train/score/std", score_std)
        logger.store("train/length", length)
        logger.store("train/length/std", length_std)
        logger.store("train/episodes/epoch", ep_n)
        logger.store("train/episodes/total", int(episodes))
        logger.store("train/gradient_steps", gradient_steps)
        # None, not 0.0, when the epoch ran no gradient bursts: a logged zero is
        # indistinguishable from a real converged loss, and that ambiguity hid the
        # v1 release grid's zero-gradient-step bug for a full overnight sweep.
        # Backends render an absent metric as a gap (see `WandbBackend.log`).
        for name, values in (("actor", actor_losses), ("critic", critic_losses)):
            logger.store(f"train/loss/{name}",
                         float(np.mean(values)) if values else None)

        # Epoch-mean of each env metric, e.g. `train/reward/upright`. Per-env
        # vectors (jax loop) and scalars (envpool loop) both reduce via np.mean.
        # (`float()` last, so the division keeps the accumulator's own dtype.)
        extra = {k: float(np.mean(v) / acc["metric_iters"])
                 for k, v in acc["metrics"].items()} if acc["metric_iters"] else {}
        # Post-clip noise in normalized [-1, 1] action units; compare against the
        # noise module's scheduled scale to see how much clipping eats.
        if acc["noise_iters"]:
            extra["noise/per_joint_abs"] = float(acc["noise"] / acc["noise_iters"])
        # Optional per-agent diagnostics (TD3 saturation/value, PPO trust region).
        diagnostics = getattr(agent, "pop_diagnostics", None)
        extra.update(diagnostics() if diagnostics is not None else {})
        # `mining/effective_bins` collapsing toward 1 means the start distribution
        # has degenerated onto a single region.
        extra.update(mining_stats or {})
        for k, v in extra.items():
            logger.store(f"train/{k}", float(v))

        now = time.time()
        logger.store("sys/sps", sps)
        logger.store("sys/time/total_s", now - start_time)
        logger.store("sys/time/epoch_s", now - last_epoch_time)
        # Device telemetry is WHOLE-CARD, so a run not on the GPU would attribute
        # another process's memory and utilisation to itself. No-op without
        # nvidia-smi.
        if any(d.platform == "gpu" for d in jax.devices()):
            for k, v in logger.gpu_stats().items():
                logger.store(k, v)
        # Host-memory watch, so creep toward an OOM is visible. /proc is Linux-only.
        try:
            rss_pages = int(open("/proc/self/statm").read().split()[1])
            logger.store("sys/mem/rss_gb",
                         rss_pages * os.sysconf("SC_PAGE_SIZE") / 1e9)
            logger.store("sys/mem/live_arrays", len(jax.live_arrays()))
        except (OSError, ValueError):
            pass

    def _test(self, rng, v_reset, v_step):
        """Run the held-out eval rollouts. `rng` is accepted for call-site
        symmetry but intentionally unused — the reset keys are fixed."""
        del rng
        num_tests = int(self.test_episodes)
        max_steps = int(getattr(self.test_environment, "max_episode_steps", 1000))

        # Fixed keys, not a fresh draw: the eval env pins its own start state, so
        # the keys only decide WHICH clip. Holding them constant means a change in
        # test/score is a change in the POLICY, not a different draw of clips.
        states = v_reset(jax.random.split(jax.random.PRNGKey(_EVAL_SEED), num_tests))

        # How many genuinely distinct states the eval batch starts from — the env
        # decides its own reset stochasticity in Python and none of that reaches
        # the logged config. `test/length/std` cannot stand in: 0.00 there means a
        # degenerate eval OR a policy saturating the episode cap, opposite news.
        start_obs = np.asarray(states.env_state.obs).reshape(num_tests, -1)
        logger.store("test/distinct_starts", float(len(np.unique(start_obs, axis=0))))

        if getattr(self, "_eval_fn", None) is None:
            self._eval_fn = self._make_eval_fn(v_step, num_tests, max_steps)

        scores, lengths = self._eval_fn(
            self.agent.state.actor, self.agent.state.obs_stats, states,
        )
        self._store_test_metrics(np.array(scores), np.array(lengths))

    @staticmethod
    def _store_test_metrics(scores, lengths):
        """Stored, not printed: `_test*` runs just before `logger.dump()`, so eval
        stats land in the same dump as training stats."""
        logger.store("test/score", float(np.mean(scores)))
        logger.store("test/score/std", float(np.std(scores)))
        logger.store("test/length", float(np.mean(lengths)))
        logger.store("test/length/std", float(np.std(lengths)))
        # Episode return is a SUM, so with spread start phases it is bounded by how
        # much clip was left at reset: a policy starting late in a non-cyclic clip
        # cannot score what a frame-0 start can, however well it tracks. The
        # per-step rate divides that out and is start-phase invariant.
        logger.store("test/score_per_step",
                     float(np.mean(scores / np.maximum(lengths, 1))))

    # ------------------------------------------------------------------
    # EnvPool (CPU) training path
    # ------------------------------------------------------------------

    def _run_envpool(self, NUM_ENVS, rngs):
        """Training loop for EnvPool (CPU) environments.

        EnvPool handles batching and auto-reset in C++, so no jax.vmap/jit
        wrapping of the env step is needed. The agent still runs in JAX on
        whatever device is active.
        """
        start_time = last_epoch_time = time.time()
        env, agent = self.environment, self.agent
        action_size = env.action_size
        action_low, action_high = agent.action_low, agent.action_high

        loop_rng, agent_key = rngs.envs(), rngs.agent()

        # Negative mining, mirroring _run_jax. Only the location of the difficulty
        # table differs: a JAX env cannot own mutable state inside a trace, so the
        # trainer threads `mining_weights` through reset, whereas a CPU pool resets
        # in plain Python and owns its own table.
        mining_on = (
            hasattr(env, "mining_refresh") and getattr(env, "mining_bins", 0) > 0
        )
        if mining_on:
            print(f"Negative mining ON: {env.mining_bins} phase bins", flush=True)

        def _random_actions(key):
            u = jax.random.uniform(key, (NUM_ENVS, action_size))
            return action_low + (action_high - action_low) * u

        # --- Reset and compile agent step ---

        print("Resetting environment...", flush=True)
        t0 = time.time()
        state = env.reset()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        self._timed(
            functools.partial(agent.step, evaluate=False, key=loop_rng),
            state.env_state.obs, label="agent step",
        )

        # --- Warmup: fill replay buffer with random actions ---

        memory_warmup = getattr(agent, "memory_warmup", 0)
        warmup_iters = self._warmup_iters(memory_warmup, NUM_ENVS)
        # See `_run_jax`: a resumed run continues its predecessor's counters.
        self.steps = self.initial_steps
        epochs, episodes = self.initial_epochs, self.initial_episodes
        tot_gradient_steps = self.initial_gradient_steps
        epoch_steps = self.steps % self.epoch_steps
        steps_since_save = self.steps % self.save_steps
        actor_losses, critic_losses = [], []

        if warmup_iters > 0:
            print(f"Warmup: {warmup_iters} iters ({memory_warmup:,} steps)...", flush=True)
            t0 = time.time()
            for _ in range(warmup_iters):
                loop_rng, act_key = jax.random.split(loop_rng)
                actions = _random_actions(act_key)
                agent.last_action = actions
                old_state = state
                state = env.step(old_state, actions)
                agent.add(old_state.env_state, state.env_state)
            jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
            self.steps = self.initial_steps + warmup_iters * NUM_ENVS
            epoch_steps = self.steps % self.epoch_steps
            steps_since_save = self.steps % self.save_steps
            print(f"  {time.time() - t0:.1f}s — {self.steps:,} steps", flush=True)

        # Precompile gradient step (buffer is now populated).
        if hasattr(agent, "update") and hasattr(agent, "steps_before_learning"):
            print("Compiling gradient step...", flush=True)
            t0 = time.time()
            agent_key, warm_key = jax.random.split(agent_key)
            agent.update(steps=agent.steps_before_learning, agent_rng=warm_key)
            jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
            # See the sync path: give back the boundary the precompile consumed.
            agent._last_update_boundary = -1
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # --- Optional async learner (overlap CPU acting + GPU learning) ---
        # A background thread owns agent.state and runs the gradient bursts, while
        # this thread acts from a behaviour-actor snapshot and hands transitions
        # off through a queue.
        learner = behavior_actor = behavior_stats = None
        behavior_version = -1
        if self.async_learner and hasattr(agent, "learn"):
            import copy

            from roxie.utils.async_learner import AsyncLearner

            agent_key, learner_key = jax.random.split(agent_key)
            behavior_actor = copy.deepcopy(agent.state.actor)
            behavior_stats = agent.state.obs_stats
            # Compile the behaviour path on THIS thread before the learner starts
            # dispatching, so the two never race a first compile of the same
            # program.
            _warm_a, _ = agent.select_action(
                behavior_actor, behavior_stats, state.env_state.obs,
                loop_rng, evaluate=False,
            )
            jax.block_until_ready(_warm_a)
            # The learner submits `learner_chunk` fused steps at a time, a
            # different program from the full-burst precompile above; compile it
            # here so the learner thread never pays a cold compile mid-run.
            agent_key, chunk_key = jax.random.split(agent_key)
            agent.learn(chunk_key, n_steps=self.learner_chunk)
            jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
            learner = AsyncLearner(
                agent, learner_key, initial_steps=self.steps,
                chunk=self.learner_chunk,
            )
            learner.start()
            print("Async learner started "
                  "(CPU acting overlaps GPU gradient bursts).", flush=True)

        # --- Training loop ---

        print("Training...", flush=True)
        scores = np.zeros(NUM_ENVS)
        lengths = np.zeros(NUM_ENVS, dtype=np.int32)
        acc = _new_epoch_acc()
        bench_t0, bench_steps = time.time(), 0

        while True:
            loop_rng, action_key = jax.random.split(loop_rng)

            if learner is not None:
                # Act from the behaviour snapshot (decoupled from the learner's
                # live, mutating networks); re-sync only when it advances.
                ver, snap = learner.latest_snapshot()
                if ver != behavior_version and snap is not None:
                    nnx.update(behavior_actor, snap[0])
                    behavior_stats = snap[1]
                    behavior_version = ver
                actions, last_noise = agent.select_action(
                    behavior_actor, behavior_stats, state.env_state.obs,
                    action_key, evaluate=False,
                )
            else:
                actions = agent.step(
                    state.env_state.obs, evaluate=False, key=action_key
                )
                last_noise = getattr(agent, "last_noise", None)
            if last_noise is not None:
                acc["noise"] += float(jnp.mean(jnp.abs(last_noise)))
                acc["noise_iters"] += 1

            old_state = state
            state = env.step(old_state, actions)

            agent_key, update_key = jax.random.split(agent_key)

            if learner is not None:
                # Hand the transition to the learner thread; pull its progress.
                learner.push(
                    old_state.env_state.obs,
                    actions,
                    state.env_state.reward,
                    state.env_state.info["termination"],
                    state.env_state.info["truncation"],
                    state.env_state.obs,
                )
                tot_gradient_steps, new_losses = learner.drain_metrics()
                for a_loss, c_loss in new_losses:
                    actor_losses.append(a_loss)
                    critic_losses.append(c_loss)
            else:
                agent.add(old_state.env_state, state.env_state)

                # The same extra update _run_jax performs on top of `agent.add`'s
                # own. It looks redundant but must stay: dropping it would leave
                # the two backends normalizing observations with differently
                # weighted statistics, breaking CPU-vs-GPU comparability.
                if getattr(agent, "normalize_observations", False):
                    agent.state.obs_stats = Agent.update_obs_stats(
                        agent.state.obs_stats, state.env_state.obs,
                    )

                gradient_steps, actor_loss, critic_loss = agent.update(
                    steps=self.steps, agent_rng=update_key,
                )
                if gradient_steps > 0:
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                tot_gradient_steps += gradient_steps

            done_np = np.array(state.env_state.done)
            scores += np.array(state.env_state.reward)
            lengths += 1
            for k, v in state.env_state.metrics.items():
                acc["metrics"][k] = acc["metrics"].get(k, 0.0) + float(np.mean(v))
            acc["metric_iters"] += 1

            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS

            done_sum = int(np.sum(done_np))
            episodes += done_sum
            ep_ret, ep_len = scores * done_np, lengths * done_np
            acc["ret"] += float(np.sum(ep_ret))
            acc["ret_sq"] += float(np.sum(ep_ret ** 2))
            acc["len"] += float(np.sum(ep_len))
            acc["len_sq"] += float(np.sum(ep_len ** 2))
            acc["count"] += done_sum

            if bench_steps == NUM_ENVS * 20:
                elapsed = time.time() - bench_t0
                print(
                    f"\n  Speed: {bench_steps / elapsed:.0f} steps/s "
                    f"({elapsed / 20 * 1000:.1f}ms/iter)",
                    flush=True,
                )

            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            if epoch_steps >= self.epoch_steps:
                if hasattr(agent, "noise_module"):
                    agent.noise_module.reset_noise()

                if self.test_environment and hasattr(agent, "state"):
                    # Eval reads agent.state; pause the learner so it observes a
                    # quiescent state and doesn't contend for the GPU.
                    if learner is not None:
                        learner.pause()
                    self._test_envpool()
                    if learner is not None:
                        learner.resume()

                epochs += 1
                epoch_steps = 0

                self._store_epoch_metrics(
                    agent, self._epoch_stats(acc, scores, lengths), acc,
                    epochs=epochs, episodes=episodes,
                    sps=self.steps / (time.time() - start_time),
                    gradient_steps=tot_gradient_steps,
                    losses=(actor_losses, critic_losses),
                    times=(start_time, last_epoch_time),
                    # Read before the refresh below, which zeroes the counters.
                    mining_stats=env.mining_stats() if mining_on else None,
                )
                if mining_on:
                    env.mining_refresh()
                # Regenerate the auto-reset pool AFTER the mining refresh, so the
                # new pool reflects this epoch's terminations.
                refresh_pool = getattr(env, "refresh_reset_pool", None)
                if refresh_pool is not None:
                    refresh_pool()
                logger.dump(step=self.steps)

                actor_losses, critic_losses = [], []
                acc = _new_epoch_acc()
                last_epoch_time = time.time()

            scores = np.where(done_np, 0.0, scores)
            lengths = np.where(done_np, 0, lengths)

            stop_training = self.steps >= self.max_steps
            if stop_training or steps_since_save >= self.save_steps:
                # Checkpointing reads (and momentarily detaches) agent.state;
                # pause the learner so it can't grad-step a half-detached state.
                if learner is not None:
                    learner.pause()
                self._save(agent, epochs, episodes, tot_gradient_steps)
                if learner is not None:
                    learner.resume()
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

    def _test_envpool(self):
        """Eval loop for EnvPool: a plain Python loop against the test pool until
        all episodes are done or max_episode_steps is reached."""
        test_env, agent = self.test_environment, self.agent
        max_steps = int(getattr(test_env, "max_episode_steps", 1000))

        state = test_env.reset()
        # Size the eval buffers from the pool the reset actually returns, not from
        # self.test_episodes: env.test_episodes and trainer.test_episodes are
        # separate config keys and a mismatch would fail the broadcast below.
        num_tests = int(np.asarray(state.env_state.reward).shape[0])
        scores = np.zeros(num_tests, dtype=np.float32)
        lengths = np.zeros(num_tests, dtype=np.int32)
        dones = np.zeros(num_tests, dtype=bool)

        # See the note in `_test`.
        start_obs = np.asarray(state.env_state.obs).reshape(num_tests, -1)
        logger.store("test/distinct_starts", float(len(np.unique(start_obs, axis=0))))

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

        self._store_test_metrics(scores, lengths)
