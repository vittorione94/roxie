import dataclasses
import functools
import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from roxie.utils import logger
from roxie.agents.agent import Agent
from roxie.agents.utils import Transition
from roxie.models.actors import deterministic_action

# Seed for the evaluation reset keys. Held constant across epochs AND across
# runs so every eval rollout starts from the same fixed set of states: test
# curves are then comparable epoch-to-epoch within a run and arm-to-arm across
# a sweep. Distinct from any training seed so eval starts are never a subset of
# what the policy trained on by construction.
_EVAL_SEED = 12345


def _agent_replay_add(agent, buffer_state, transitions):
    """Add a (B, ...) batch through the agent's `replay_add` when it has one
    (DDPG/TD3 insert the time axis their trajectory buffer expects when
    n_step > 1); fall back to the raw flat-buffer add otherwise (SAC)."""
    fn = getattr(agent, "replay_add", None)
    if fn is not None:
        return fn(buffer_state, transitions)
    return agent.replay.add(buffer_state, transitions)


class Trainer:

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(3e5), save_steps=int(1e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
        async_learner=False, learner_chunk=8,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.output_dir = output_dir
        # Overlap CPU env-stepping with GPU gradient bursts via a background
        # learner thread (envpool loop only; see roxie.utils.async_learner).
        # Opt-in: only pays off when the learner is the dominant cost and the
        # env runs on CPU while the learner runs on GPU.
        self.async_learner = async_learner
        # Fused grad steps the async learner submits per GPU dispatch. Smaller =
        # more acting/learning interleave (better overlap) but more per-chunk
        # Python/dispatch overhead; larger = fewer, longer GPU submissions that
        # stall acting. See roxie.utils.async_learner.
        self.learner_chunk = int(learner_chunk)

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

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
        # difficulty table; the trainer only threads it. `mining_weights` is a
        # TRACED reset argument rather than an env attribute so refreshing it
        # each epoch does not retrigger compilation.
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
        action_low = agent.action_low
        action_high = agent.action_high
        POOL_SIZE = NUM_ENVS

        # --- Core: step envs, auto-reset done ones from a pre-built pool ---

        def _step_and_autoreset(states, actions, rng, reset_pool, mining_counts):
            step_key, pool_key = jax.random.split(rng)
            new_states = v_step(states, actions)
            dones = new_states.env_state.done
            idx = jax.random.randint(pool_key, (NUM_ENVS,), 0, POOL_SIZE)

            # Two scatter-adds on a (bins,) array — negligible against the
            # physics step, and it has to live here because this is the only
            # place that sees every env's phase and done flag on-device.
            if mining_counts is not None:
                es = new_states.env_state
                # `info["termination"]` is done-minus-truncation, already
                # separated by TerminationWrapper — the same flag the critic
                # bootstraps on. Clip-end and step-limit cutoffs are NOT
                # failures and must not be mined for.
                mining_counts = mining_env.mining_observe(
                    mining_counts, es.info, es.info["termination"]
                )

            def _autoreset_leaf(pool_leaf, s):
                # Leaves without a per-env leading dim (e.g. the warp backend's
                # world-flattened contact arena) can't be reset per-env and would
                # be indexed out of bounds by the pool gather; the physics
                # recomputes them each step, so keep the stepped value.
                if not (isinstance(s, jnp.ndarray) and s.shape[:1] == dones.shape):
                    return s
                picked = pool_leaf[idx]
                return jnp.where(
                    dones.reshape(dones.shape + (1,) * (s.ndim - 1)), picked, s
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

        # Build initial reset pool (reused for auto-reset via gather)
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

        print("Compiling agent step...", flush=True)
        t0 = time.time()
        _ = agent.step(wrapped_states.env_state.obs, evaluate=False, key=loop_rng)
        jax.block_until_ready(_)
        print(f"  {time.time() - t0:.1f}s", flush=True)

        # Off-policy replay add precompile. Skipped for on-policy agents (PPO),
        # whose buffer uses a different Transition layout (log_probs/value + time
        # axis) and which has no warmup phase.
        if getattr(agent, "memory_warmup", 0) > 0:
            print("Compiling replay add...", flush=True)
            t0 = time.time()
            # Build the dummy from the agent's OWN buffer prototype (leaves are
            # (add_batch, time, ...)) so this stays correct across per-agent
            # Transition layouts (e.g. DDPG/TD3 store `truncation`, SAC not).
            dummy_t = jax.tree.map(
                lambda leaf: jnp.zeros((NUM_ENVS,) + leaf.shape[2:], leaf.dtype),
                agent.state.buffer_state.experience,
            )
            _ = _agent_replay_add(agent, agent.state.buffer_state, dummy_t)
            jax.block_until_ready(jax.tree.leaves(_))
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # Clean reset
        loop_rng, rng = jax.random.split(loop_rng)
        wrapped_states = jit_v_reset(jax.random.split(rng, NUM_ENVS), mining_weights)

        # --- Warmup: scan with random actions ---

        agent_key = rngs.agent()
        memory_warmup = getattr(agent, 'memory_warmup', 0)
        warmup_iters = memory_warmup // NUM_ENVS
        self.steps = 0
        epoch_steps = 0
        epochs = 0
        episodes = 0
        tot_gradient_steps = 0
        steps_since_save = 0
        actor_losses = []
        critic_losses = []

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
                        # (DDPG/TD3 n-step); harmless extra field otherwise —
                        # batch_add below prunes to the buffer's own layout.
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

            # Donate the buffer state: without donation XLA keeps the input
            # buffer alive AND allocates a full output copy — a transient 2x
            # of the largest array in the program (the buffer's obs store:
            # ~3GB per copy at look_ahead=5), which OOMs the pool right here.
            # The input is dead after the reassignment below — same pattern
            # as _grad_steps' donate_argnums in the agents.
            # Prune the rollout transitions to the fields the agent's buffer
            # actually stores (SAC's prototype has no `truncation`; DDPG/TD3's
            # does) so the pytree structures match at add time.
            proto = agent.state.buffer_state.experience
            transitions = Transition(**{
                f.name: (
                    getattr(transitions, f.name)
                    if getattr(proto, f.name) is not None else None
                )
                for f in dataclasses.fields(Transition)
            })

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

            self.steps = warmup_iters * NUM_ENVS
            epoch_steps = self.steps % self.epoch_steps
            episodes = int(jnp.sum(transitions.terminal))
            steps_since_save = self.steps % self.save_steps
            print(f"  Done: {self.steps:,} steps, {episodes} episodes", flush=True)

        # Precompile the gradient step (buffer is now full of warmup data).
        if hasattr(agent, "update") and hasattr(agent, "steps_before_learning"):
            print("Compiling gradient step...", flush=True)
            t0 = time.time()
            agent_key, warm_key = jax.random.split(agent_key)
            agent.update(steps=agent.steps_before_learning, agent_rng=warm_key)
            jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # --- Training loop ---

        print("Training...", flush=True)
        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, dtype=jnp.int32)
        # Completed-episode accumulators for the current epoch. `scores`/`lengths`
        # above are the *in-flight* per-env counters; these collect the return and
        # length of episodes that actually terminate, so the epoch log reports
        # real episode stats rather than the in-progress counter. Reset each epoch.
        ep_return_sum = jnp.zeros(())
        ep_len_sum = jnp.zeros(())
        ep_count = jnp.zeros(())
        # Sum-of-squares companions to the two accumulators above, used to
        # recover the per-episode std at the epoch boundary via
        # sqrt(E[x^2] - E[x]^2) — without keeping a host-side list of individual
        # episode returns (which would force a device->host sync every step).
        ep_return_sq_sum = jnp.zeros(())
        ep_len_sq_sum = jnp.zeros(())
        # Per-epoch accumulators for the env's own metrics dict (e.g. the mocap
        # env's `reward/pose`, `reward/vel`, ... tracking-reward components). The
        # env populates `env_state.metrics` with a scalar per env each step; we
        # sum the per-step env-mean and divide by the iteration count at dump
        # time to report the epoch-mean of each component. Keys are discovered
        # lazily so this stays generic across envs (empty dict -> nothing logged).
        metric_sums = {}
        metric_iters = 0
        # Exploration-noise accumulator (deterministic agents only). Tracked
        # separately from metric_sums: noise is added only on policy steps (not
        # during the random warmup), so it has its own iteration count.
        noise_abs_sum = jnp.zeros(())
        noise_iters = 0
        bench_t0 = time.time()
        bench_steps = 0

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
                # Mean absolute exploration noise per joint (normalized action
                # units), averaged over envs + joints. Deterministic agents
                # (DDPG/TD3) expose last_noise; others simply don't contribute.
                last_noise = getattr(agent, "last_noise", None)
                if last_noise is not None:
                    noise_abs_sum = noise_abs_sum + jnp.mean(jnp.abs(last_noise))
                    noise_iters += 1
            else:
                actions = _random_actions(action_key)
                agent.last_action = actions

            old_wrapped_states = wrapped_states
            new_wrapped_states, wrapped_states, mining_counts = train_step(
                old_wrapped_states, actions, step_key, reset_pool, mining_counts,
            )

            agent_key, update_key = jax.random.split(agent_key)
            agent.add(old_wrapped_states.env_state, new_wrapped_states.env_state)

            # Keep the obs-normalization running stats tracking the CURRENT
            # policy's state distribution. Stats were previously accumulated only
            # from the random-action warmup and then frozen, so velocity- and
            # ref-delta-scale features stayed normalized to the flailing-policy
            # regime for the whole run. Obs are stored raw in the replay buffer
            # and normalized at sample time, so continuously-updated stats stay
            # consistent for old and new data alike. Pure device op — no host
            # sync, same async-pipeline rules as the metric accumulation below.
            if getattr(agent, "normalize_observations", False):
                agent.state.obs_stats = Agent.update_obs_stats(
                    agent.state.obs_stats, new_wrapped_states.env_state.obs,
                )

            # Let the agent gate its own updates. Every agent.update() decides
            # internally whether to run gradient steps (DDPG/SAC via Python step
            # counters, PPO by draining its rollout buffer). The trainer must NOT
            # read buffer device state here: a Python branch on a device array
            # (e.g. flashbax can_sample) forces a blocking host sync every
            # iteration and serializes the async GPU pipeline.
            gradient_steps, actor_loss, critic_loss = agent.update(
                steps=self.steps, agent_rng=update_key,
            )
            if gradient_steps > 0:
                actor_losses.append(actor_loss)
                critic_losses.append(critic_loss)

            tot_gradient_steps += gradient_steps
            scores += new_wrapped_states.env_state.reward
            lengths += 1
            # Accumulate the env's reward-component metrics as per-env vectors
            # (deferred mean to epoch boundary to avoid one dispatch per
            # metric per step).
            for k, v in new_wrapped_states.env_state.metrics.items():
                metric_sums[k] = metric_sums.get(k, jnp.zeros_like(v)) + v
            metric_iters += 1
            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS
            done = new_wrapped_states.env_state.done.astype(jnp.float32)
            # Compute jnp.sum(done) once — reused for both `episodes` and
            # `ep_count` to avoid a duplicate reduction dispatch each step.
            done_sum = jnp.sum(done)
            episodes = episodes + done_sum
            # Capture return/length of episodes terminating this step. `scores`
            # and `lengths` already include the terminal transition (updated
            # above), and are zeroed for done envs further down.
            ep_returns_done = scores * done
            ep_lengths_done = lengths.astype(jnp.float32) * done
            ep_return_sum = ep_return_sum + jnp.sum(ep_returns_done)
            ep_len_sum = ep_len_sum + jnp.sum(ep_lengths_done)
            # Non-done envs contribute 0 here, so squaring keeps summing only
            # completed episodes' returns/lengths.
            ep_return_sq_sum = ep_return_sq_sum + jnp.sum(ep_returns_done ** 2)
            ep_len_sq_sum = ep_len_sq_sum + jnp.sum(ep_lengths_done ** 2)
            ep_count = ep_count + done_sum

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
                sps = self.steps / (time.time() - start_time)

                # Report stats over episodes that completed this epoch. When none
                # finished (episodes longer than an epoch), fall back to the
                # in-flight counters so the line is never blank.
                ep_n = int(ep_count)
                if ep_n > 0:
                    epoch_score = float(ep_return_sum / ep_count)
                    epoch_length = float(ep_len_sum / ep_count)
                    # Population std across the epoch's completed episodes.
                    # Clamp to guard against tiny negative variances from
                    # float round-off in the E[x^2]-E[x]^2 subtraction.
                    epoch_score_std = float(np.sqrt(max(
                        float(ep_return_sq_sum / ep_count) - epoch_score ** 2, 0.0)))
                    epoch_length_std = float(np.sqrt(max(
                        float(ep_len_sq_sum / ep_count) - epoch_length ** 2, 0.0)))
                else:
                    epoch_score = float(jnp.mean(scores))
                    epoch_length = float(jnp.mean(lengths))
                    epoch_score_std = float(jnp.std(scores))
                    epoch_length_std = float(jnp.std(lengths.astype(jnp.float32)))

                # Feed the epoch stats through the logger so every configured
                # backend (console table, CSV, wandb, ...) records them. Keyed
                # by env steps so the wandb x-axis matches training progress.
                logger.store("epoch", epochs)
                logger.store("steps", self.steps)
                logger.store("episodes/epoch", ep_n)
                logger.store("episodes/total", int(episodes))
                logger.store("time/total_s", time.time() - start_time)
                logger.store("time/epoch_s", time.time() - last_epoch_time)
                logger.store("sps", sps)
                logger.store("score", epoch_score)
                logger.store("score/std", epoch_score_std)
                logger.store("length", epoch_length)
                logger.store("length/std", epoch_length_std)
                logger.store("gradient_steps", tot_gradient_steps)
                logger.store(
                    "loss/actor",
                    float(np.mean(actor_losses)) if actor_losses else 0.0,
                )
                logger.store(
                    "loss/critic",
                    float(np.mean(critic_losses)) if critic_losses else 0.0,
                )
                # Epoch-mean of each env reward component (and any other metric
                # the env exposes), keyed by its own name so the console/CSV/wandb
                # backends nest it (e.g. under `reward/`).
                if metric_iters > 0:
                    for k, total in metric_sums.items():
                        logger.store(k, float(jnp.mean(total) / metric_iters))
                # Average exploration noise added per joint this epoch (post-clip,
                # normalized [-1, 1] action units). Compare against the noise
                # module's scheduled scale to see how much clipping eats.
                if noise_iters > 0:
                    logger.store("noise/per_joint_abs", float(noise_abs_sum / noise_iters))
                # Optional per-agent diagnostics, drained once per epoch. PPO
                # uses this for its trust-region metrics (approx_kl, clip_frac);
                # agents without the hook contribute nothing.
                pop_diagnostics = getattr(agent, "pop_diagnostics", None)
                if pop_diagnostics is not None:
                    for k, v in pop_diagnostics().items():
                        logger.store(k, float(v))
                # Negative-mining health. Logged BEFORE the refresh below, which
                # zeroes the counters. `mining/effective_bins` is the one to
                # watch: if it collapses toward 1 the start distribution has
                # degenerated onto a single region and coverage is being lost.
                if mining_on:
                    for k, v in mining_env.mining_stats(
                        mining_weights, mining_counts
                    ).items():
                        logger.store(k, float(v))
                # GPU telemetry (utilization / temperature / memory / power).
                # Sampled once per epoch; a no-op on hosts without nvidia-smi.
                for k, v in logger.gpu_stats().items():
                    logger.store(k, v)
                # Host-memory watch. Checkpoint saves once OOM-killed the process
                # mid-write (host RAM exhausted); track resident set + the number
                # of live JAX buffers each epoch so any baseline creep is visible
                # in the logs well before it hits the ceiling. /proc is Linux-only;
                # guarded so non-Linux hosts just skip it.
                try:
                    page = os.sysconf("SC_PAGE_SIZE")
                    rss_pages = int(open("/proc/self/statm").read().split()[1])
                    logger.store("mem/rss_gb", rss_pages * page / 1e9)
                    logger.store("mem/live_arrays", len(jax.live_arrays()))
                except (OSError, ValueError):
                    pass
                logger.dump(step=self.steps)

                actor_losses = []
                critic_losses = []
                ep_return_sum = jnp.zeros(())
                ep_len_sum = jnp.zeros(())
                ep_return_sq_sum = jnp.zeros(())
                ep_len_sq_sum = jnp.zeros(())
                ep_count = jnp.zeros(())
                metric_sums = {}
                metric_iters = 0
                noise_abs_sum = jnp.zeros(())
                noise_iters = 0

                # Regenerate the reset pool (fresh random starts for auto-reset)
                # and reshuffle the GPU clip subset if the env supports it. Only a
                # real clip swap invalidates in-progress episodes (their stored
                # clip indices reference the old chunk), so only then do we reset
                # the live envs. Otherwise episodes run continuously across epochs.
                # Fold this epoch's terminations into the start distribution
                # BEFORE rebuilding the pool, so the new pool already reflects
                # them. Once per epoch: the per-step cost is two scatter-adds,
                # the normalization/EMA happens here.
                if mining_on:
                    mining_weights, mining_counts = mining_env.mining_refresh(
                        mining_weights, mining_counts
                    )

                loop_rng, pool_rng = jax.random.split(loop_rng)
                reset_pool = jit_v_reset(jax.random.split(pool_rng, POOL_SIZE), mining_weights)
                swapped = False
                if hasattr(self.environment, 'swap_clips'):
                    swapped = bool(self.environment.swap_clips())
                if swapped:
                    loop_rng, reset_rng = jax.random.split(loop_rng)
                    wrapped_states = jit_v_reset(
                        jax.random.split(reset_rng, NUM_ENVS), mining_weights
                    )
                    scores = jnp.zeros(NUM_ENVS)
                    lengths = jnp.zeros(NUM_ENVS, dtype=jnp.int32)

                last_epoch_time = time.time()

            scores = jnp.where(new_wrapped_states.env_state.done, 0, scores)
            lengths = jnp.where(new_wrapped_states.env_state.done, 0, lengths)

            stop_training = self.steps >= self.max_steps
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(self.output_dir, 'checkpoints')
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                save_path = os.path.join(path, f'step_{self.steps}')
                agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

    def _make_eval_fn(self, v_step, num_tests, max_steps):
        """Build a single compiled eval rollout.

        The whole episode loop runs inside ``jax.lax.while_loop`` so the
        termination check (``~all(dones)``) is evaluated on-device — no per-step
        host sync, full GPU pipelining — unlike a Python ``while`` that drags a
        device array back to the host every step. A static ``max_steps`` cap
        bounds compute and guarantees termination. Actor / obs-stats are passed
        as traced args so the same compiled fn is reused every epoch.
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
                # Deterministic eval: actor output (in [-1, 1]) scaled to env
                # units. No noise module (its stateful update can't be mutated
                # across the while_loop trace level). A stochastic actor (PPO)
                # returns a distribution rather than an action, so the mean is
                # taken here — never a sample, so eval stays deterministic.
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

    def _test(self, rng, v_reset, v_step):
        """Run the held-out eval rollouts. `rng` is accepted for call-site
        symmetry but intentionally unused — see the fixed reset keys below."""
        del rng
        num_tests = int(self.test_episodes)
        max_steps = int(getattr(self.test_environment, "max_episode_steps", 1000))

        # FIXED reset keys, not a fresh draw off `rng`. The eval env pins the
        # start state itself (mocap: frame 0, no noise — see its eval-env note),
        # so for a single-clip run the keys change nothing; they matter for
        # multi-clip runs, where reset samples WHICH clip. Holding them constant
        # means eval scores the same clips every epoch, so a change in
        # test/score is a change in the POLICY rather than a different draw of
        # clips. `max_steps` comes from the eval env's own horizon, which for
        # mocap is the longest clip — eval runs each clip to its end.
        states = v_reset(jax.random.split(jax.random.PRNGKey(_EVAL_SEED), num_tests))

        # How many GENUINELY distinct states the eval batch starts from. The env
        # decides its own reset stochasticity in Python and none of that reaches
        # the logged hydra config, so runs were not self-describing: you could
        # not tell from a run's artifacts what protocol its test/* numbers meant.
        # `test/length/std` cannot stand in for this — 0.00 means a degenerate
        # eval OR a policy saturating the episode cap, which are opposite news.
        # Under the canonical mocap protocol this equals the number of DISTINCT
        # CLIPS being evaluated; 1 on a multi-clip run means the eval keys are
        # not covering the set, and 1 on a single-clip run means every extra
        # `test_episodes` beyond the first is pure wasted compute.
        start_obs = np.asarray(states.env_state.obs).reshape(num_tests, -1)
        logger.store("test/distinct_starts", float(len(np.unique(start_obs, axis=0))))

        if getattr(self, "_eval_fn", None) is None:
            self._eval_fn = self._make_eval_fn(v_step, num_tests, max_steps)

        scores, lengths = self._eval_fn(
            self.agent.state.actor, self.agent.state.obs_stats, states,
        )

        scores_np = np.array(scores)
        lengths_np = np.array(lengths)
        # Stored (not printed) so the eval stats land in the same epoch dump as
        # the training stats and reach every backend. The epoch loop calls
        # _test just before logger.dump().
        logger.store("test/score", float(np.mean(scores_np)))
        logger.store("test/score/std", float(np.std(scores_np)))
        logger.store("test/length", float(np.mean(lengths_np)))
        logger.store("test/length/std", float(np.std(lengths_np)))
        # Episode return is a SUM, so with spread start phases it is bounded by
        # how much clip was left at reset — a policy starting late in a
        # non-cyclic clip cannot score what a frame-0 start can, however well it
        # tracks. The per-step rate divides that out: it is start-phase
        # invariant, and it is the number to compare against runs recorded under
        # the old frame-0-only eval, whose `test/score` is on a different scale.
        logger.store(
            "test/score_per_step",
            float(np.mean(scores_np / np.maximum(lengths_np, 1))),
        )

    # ------------------------------------------------------------------
    # EnvPool (CPU) training path
    # ------------------------------------------------------------------

    def _run_envpool(self, NUM_ENVS, rngs):
        """Training loop for EnvPool (CPU) environments.

        EnvPool handles batching and auto-reset in C++, so no jax.vmap/jit
        wrapping of the env step is needed. The agent (actor, critic, replay
        buffer, gradient updates) still runs in JAX on whatever device is
        active (CPU by default, or GPU if one is present and JAX_PLATFORMS is
        not overridden).
        """
        start_time = last_epoch_time = time.time()
        env = self.environment
        agent = self.agent
        action_size = env.action_size
        action_low = agent.action_low
        action_high = agent.action_high

        loop_rng = rngs.envs()
        agent_key = rngs.agent()

        def _random_actions(key):
            u = jax.random.uniform(key, (NUM_ENVS, action_size))
            return action_low + (action_high - action_low) * u

        # --- Reset and compile agent step ---

        print("Resetting environment...", flush=True)
        t0 = time.time()
        state = env.reset()
        print(f"  {time.time() - t0:.1f}s", flush=True)

        print("Compiling agent step...", flush=True)
        t0 = time.time()
        _ = agent.step(state.env_state.obs, evaluate=False, key=loop_rng)
        jax.block_until_ready(_)
        print(f"  {time.time() - t0:.1f}s", flush=True)

        # --- Warmup: fill replay buffer with random actions ---

        memory_warmup = getattr(agent, "memory_warmup", 0)
        warmup_iters = memory_warmup // NUM_ENVS
        self.steps = 0
        epoch_steps = 0
        epochs = 0
        episodes = 0
        tot_gradient_steps = 0
        steps_since_save = 0
        actor_losses = []
        critic_losses = []

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
            self.steps = warmup_iters * NUM_ENVS
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
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # --- Optional async learner (overlap CPU acting + GPU learning) ---
        # When enabled, a background thread owns agent.state and runs the
        # gradient bursts, while this (main) thread acts from a behaviour-actor
        # snapshot and hands transitions off through a queue. See
        # roxie.utils.async_learner for the ownership/donation contract.
        learner = None
        behavior_actor = None
        behavior_stats = None
        behavior_version = -1
        if self.async_learner and hasattr(agent, "learn"):
            import copy

            from roxie.utils.async_learner import AsyncLearner

            agent_key, learner_key = jax.random.split(agent_key)
            behavior_actor = copy.deepcopy(agent.state.actor)
            behavior_stats = agent.state.obs_stats
            # Trigger any behaviour-path compilation on THIS thread before the
            # learner starts dispatching, so the two threads never race a first
            # compile of the same program.
            _warm_a, _ = agent.select_action(
                behavior_actor, behavior_stats, state.env_state.obs,
                loop_rng, evaluate=False,
            )
            jax.block_until_ready(_warm_a)
            # Precompile the chunk-sized grad program here (the learner submits
            # `learner_chunk` fused steps at a time, a different program from the
            # full-burst precompile above) so the learner thread never pays a
            # cold compile mid-run.
            agent_key, chunk_key = jax.random.split(agent_key)
            agent.learn(chunk_key, n_steps=self.learner_chunk)
            jax.block_until_ready(jax.tree.leaves(nnx.state(agent.state)))
            learner = AsyncLearner(
                agent, learner_key, initial_steps=self.steps,
                chunk=self.learner_chunk,
            )
            learner.start()
            print(
                "Async learner started "
                "(CPU acting overlaps GPU gradient bursts).",
                flush=True,
            )

        # --- Training loop ---

        print("Training...", flush=True)
        scores = np.zeros(NUM_ENVS)
        lengths = np.zeros(NUM_ENVS, dtype=np.int32)
        ep_return_sum = 0.0
        ep_len_sum = 0.0
        # Sum-of-squares companions for the per-episode std (see the jax path).
        ep_return_sq_sum = 0.0
        ep_len_sq_sum = 0.0
        ep_count = 0
        metric_sums = {}
        metric_iters = 0
        noise_abs_sum = 0.0
        noise_iters = 0
        bench_t0 = time.time()
        bench_steps = 0

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
                noise_abs_sum += float(jnp.mean(jnp.abs(last_noise)))
                noise_iters += 1

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

                gradient_steps, actor_loss, critic_loss = agent.update(
                    steps=self.steps, agent_rng=update_key,
                )
                if gradient_steps > 0:
                    actor_losses.append(actor_loss)
                    critic_losses.append(critic_loss)
                tot_gradient_steps += gradient_steps

            done_np = np.array(state.env_state.done)
            reward_np = np.array(state.env_state.reward)
            scores += reward_np
            lengths += 1
            for k, v in state.env_state.metrics.items():
                metric_sums[k] = metric_sums.get(k, 0.0) + float(np.mean(np.array(v)))
            metric_iters += 1

            done_sum = int(np.sum(done_np))
            episodes += done_sum
            ep_returns_done = scores * done_np
            ep_lengths_done = lengths * done_np
            ep_return_sum += float(np.sum(ep_returns_done))
            ep_len_sum += float(np.sum(ep_lengths_done))
            ep_return_sq_sum += float(np.sum(ep_returns_done ** 2))
            ep_len_sq_sum += float(np.sum(ep_lengths_done ** 2))
            ep_count += done_sum

            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS

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
                    # Eval reads agent.state (via agent.step); pause the learner
                    # so it observes a quiescent, consistent state and doesn't
                    # contend for the GPU during scoring.
                    if learner is not None:
                        learner.pause()
                    self._test_envpool()
                    if learner is not None:
                        learner.resume()

                epochs += 1
                epoch_steps = 0
                sps = self.steps / (time.time() - start_time)

                ep_n = ep_count
                if ep_n > 0:
                    epoch_score = ep_return_sum / ep_count
                    epoch_length = ep_len_sum / ep_count
                    # Population std over the epoch's completed episodes,
                    # clamped against float round-off (see the jax path).
                    epoch_score_std = float(np.sqrt(max(
                        ep_return_sq_sum / ep_count - epoch_score ** 2, 0.0)))
                    epoch_length_std = float(np.sqrt(max(
                        ep_len_sq_sum / ep_count - epoch_length ** 2, 0.0)))
                else:
                    epoch_score = float(np.mean(scores))
                    epoch_length = float(np.mean(lengths))
                    epoch_score_std = float(np.std(scores))
                    epoch_length_std = float(np.std(lengths))

                logger.store("epoch", epochs)
                logger.store("steps", self.steps)
                logger.store("episodes/epoch", ep_n)
                logger.store("episodes/total", int(episodes))
                logger.store("time/total_s", time.time() - start_time)
                logger.store("time/epoch_s", time.time() - last_epoch_time)
                logger.store("sps", sps)
                logger.store("score", epoch_score)
                logger.store("score/std", epoch_score_std)
                logger.store("length", epoch_length)
                logger.store("length/std", epoch_length_std)
                logger.store("gradient_steps", tot_gradient_steps)
                logger.store(
                    "loss/actor",
                    float(np.mean([float(x) for x in actor_losses])) if actor_losses else 0.0,
                )
                logger.store(
                    "loss/critic",
                    float(np.mean([float(x) for x in critic_losses])) if critic_losses else 0.0,
                )
                if metric_iters > 0:
                    for k, total in metric_sums.items():
                        logger.store(k, total / metric_iters)
                if noise_iters > 0:
                    logger.store("noise/per_joint_abs", noise_abs_sum / noise_iters)
                # Same optional per-agent diagnostics hook the MJX loop drains
                # (PPO's trust-region metrics, TD3's saturation/value metrics).
                # It was missing here, so the CPU path silently logged none of
                # them; agents without the hook still contribute nothing.
                pop_diagnostics = getattr(agent, "pop_diagnostics", None)
                if pop_diagnostics is not None:
                    for k, v in pop_diagnostics().items():
                        logger.store(k, float(v))
                for k, v in logger.gpu_stats().items():
                    logger.store(k, v)
                logger.dump(step=self.steps)

                actor_losses = []
                critic_losses = []
                ep_return_sum = 0.0
                ep_len_sum = 0.0
                ep_return_sq_sum = 0.0
                ep_len_sq_sum = 0.0
                ep_count = 0
                metric_sums = {}
                metric_iters = 0
                noise_abs_sum = 0.0
                noise_iters = 0
                last_epoch_time = time.time()

            scores = np.where(done_np, 0.0, scores)
            lengths = np.where(done_np, 0, lengths)

            stop_training = self.steps >= self.max_steps
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(self.output_dir, "checkpoints")
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith("step_"):
                            os.remove(os.path.join(path, file))
                save_path = os.path.join(path, f"step_{self.steps}")
                # Checkpointing reads (and momentarily detaches) agent.state;
                # pause the learner so it can't grad-step a half-detached state.
                if learner is not None:
                    learner.pause()
                agent.save(save_path)
                if learner is not None:
                    learner.resume()
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                if learner is not None:
                    learner.stop()
                break

    def _test_envpool(self):
        """Eval loop for EnvPool environments.

        Runs a Python loop (no jax.lax.while_loop) against the test pool until
        all test episodes are done or max_episode_steps is reached.
        """
        test_env = self.test_environment
        agent = self.agent
        max_steps = int(getattr(test_env, "max_episode_steps", 1000))

        state = test_env.reset()
        # Size the eval buffers from the pool the reset actually returns, not
        # from self.test_episodes: the test pool's env count (env.test_episodes)
        # and trainer.test_episodes are separate config keys, and a mismatch
        # would otherwise fail the `reward * active` broadcast below.
        num_tests = int(np.asarray(state.env_state.reward).shape[0])
        scores = np.zeros(num_tests, dtype=np.float32)
        lengths = np.zeros(num_tests, dtype=np.int32)
        dones = np.zeros(num_tests, dtype=bool)

        # Use a fixed key for eval (noise is bypassed when evaluate=True).
        eval_key = jax.random.PRNGKey(0)

        for _ in range(max_steps):
            if np.all(dones):
                break
            actions = agent.step(state.env_state.obs, evaluate=True, key=eval_key)
            state = test_env.step(state, actions)
            new_done = np.array(state.env_state.done)
            reward = np.array(state.env_state.reward)
            active = ~dones
            scores += reward * active
            lengths += active.astype(np.int32)
            dones |= new_done

        logger.store("test/score", float(np.mean(scores)))
        logger.store("test/score/std", float(np.std(scores)))
        logger.store("test/length", float(np.mean(lengths)))
        logger.store("test/length/std", float(np.std(lengths)))
