import os
import time

import numpy as np
import jax
import jax.numpy as jnp
from flax import nnx
from roxie.utils import logger
from roxie.agents.agent import Agent
from roxie.agents.utils import Transition


class Trainer:

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(3e5), save_steps=int(1e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint
        self.output_dir = output_dir

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

        start_time = last_epoch_time = time.time()
        agent = self.agent
        v_reset = jax.vmap(self.environment.reset)
        v_step = jax.vmap(self.environment.step)
        action_size = self.environment.action_size
        action_low = agent.action_low
        action_high = agent.action_high
        POOL_SIZE = NUM_ENVS

        # --- Core: step envs, auto-reset done ones from a pre-built pool ---

        def _step_and_autoreset(states, actions, rng, reset_pool):
            step_key, pool_key = jax.random.split(rng)
            new_states = v_step(states, actions)
            dones = new_states.env_state.done
            idx = jax.random.randint(pool_key, (NUM_ENVS,), 0, POOL_SIZE)

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
            return new_states, auto_states

        def _random_actions(key):
            u = jax.random.uniform(key, (NUM_ENVS, action_size))
            return action_low + (action_high - action_low) * u

        # --- JIT wrappers ---

        @jax.jit
        def train_step(states, actions, rng, reset_pool):
            return _step_and_autoreset(states, actions, rng, reset_pool)

        jit_v_reset = jax.jit(v_reset)

        v_test_reset = jax.jit(jax.vmap(self.test_environment.reset))
        v_test_step = jax.jit(jax.vmap(self.test_environment.step))

        # --- Compile everything upfront ---

        loop_rng = rngs.envs()

        reset_keys = jax.random.split(loop_rng, NUM_ENVS)
        wrapped_states = self._timed(jit_v_reset, reset_keys, label="reset")

        # Build initial reset pool (reused for auto-reset via gather)
        loop_rng, pool_rng = jax.random.split(loop_rng)
        reset_pool = self._timed(
            jit_v_reset, jax.random.split(pool_rng, POOL_SIZE), label="reset pool",
        )

        dummy_actions = jnp.zeros((NUM_ENVS, action_size))
        self._timed(
            train_step, wrapped_states, dummy_actions, loop_rng, reset_pool,
            label="train step",
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
            dummy_t = Transition(
                observation=wrapped_states.env_state.obs,
                action=jnp.zeros((NUM_ENVS, action_size)),
                reward=jnp.zeros(NUM_ENVS),
                terminal=jnp.zeros(NUM_ENVS, dtype=jnp.bool_),
            )
            _ = agent.replay.add(agent.state.buffer_state, dummy_t)
            jax.block_until_ready(jax.tree.leaves(_))
            print(f"  {time.time() - t0:.1f}s", flush=True)

        # Clean reset
        loop_rng, rng = jax.random.split(loop_rng)
        wrapped_states = jit_v_reset(jax.random.split(rng, NUM_ENVS))

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
            t0 = time.time()

            @jax.jit
            def warmup_rollout(state, rng, reset_pool):
                def body(carry, _):
                    state, rng = carry
                    rng, act_key, step_key = jax.random.split(rng, 3)
                    actions = _random_actions(act_key)
                    new_states, auto_states = _step_and_autoreset(
                        state, actions, step_key, reset_pool,
                    )
                    transition = Transition(
                        observation=state.env_state.obs,
                        action=actions,
                        reward=new_states.env_state.reward,
                        terminal=new_states.env_state.info["termination"],
                    )
                    return (auto_states, rng), (transition, new_states.env_state.obs)
                return jax.lax.scan(body, (state, rng), None, length=warmup_iters)

            (wrapped_states, loop_rng), (transitions, next_obs) = warmup_rollout(
                wrapped_states, loop_rng, reset_pool,
            )
            wrapped_states.env_state.obs.block_until_ready()
            print(f"  Rollout: {time.time() - t0:.1f}s", flush=True)

            t0 = time.time()

            @jax.jit
            def batch_add(buffer_state, transitions):
                def add_one(bs, t):
                    return agent.replay.add(bs, t), None
                bs, _ = jax.lax.scan(add_one, buffer_state, transitions)
                return bs

            agent.state.buffer_state = batch_add(agent.state.buffer_state, transitions)
            jax.block_until_ready(jax.tree.leaves(agent.state.buffer_state))
            print(f"  Replay fill: {time.time() - t0:.1f}s", flush=True)

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
            else:
                actions = _random_actions(action_key)
                agent.last_action = actions

            old_wrapped_states = wrapped_states
            new_wrapped_states, wrapped_states = train_step(
                old_wrapped_states, actions, step_key, reset_pool,
            )

            agent_key, update_key = jax.random.split(agent_key)
            agent.add(old_wrapped_states.env_state, new_wrapped_states.env_state)

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
            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS
            bench_steps += NUM_ENVS
            done = new_wrapped_states.env_state.done.astype(jnp.float32)
            episodes = episodes + jnp.sum(done)
            # Capture return/length of episodes terminating this step. `scores`
            # and `lengths` already include the terminal transition (updated
            # above), and are zeroed for done envs further down.
            ep_return_sum = ep_return_sum + jnp.sum(scores * done)
            ep_len_sum = ep_len_sum + jnp.sum(lengths.astype(jnp.float32) * done)
            ep_count = ep_count + jnp.sum(done)

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
                else:
                    epoch_score = float(jnp.mean(scores))
                    epoch_length = float(jnp.mean(lengths))

                print(
                    f"\nEpoch {epochs} | steps {self.steps:,} | "
                    f"episodes {ep_n} (tot {int(episodes)}) | "
                    f"time {time.time() - start_time:.0f}s | "
                    f"epoch {time.time() - last_epoch_time:.1f}s | "
                    f"SPS {sps:.0f} | "
                    f"score {epoch_score:.2f} | "
                    f"length {epoch_length:.1f} | "
                    f"grad_steps {tot_gradient_steps} | "
                    f"a_loss {np.mean(actor_losses) if actor_losses else 0:.4f} | "
                    f"c_loss {np.mean(critic_losses) if critic_losses else 0:.6f}",
                    flush=True,
                )
                actor_losses = []
                critic_losses = []
                ep_return_sum = jnp.zeros(())
                ep_len_sum = jnp.zeros(())
                ep_count = jnp.zeros(())

                # Regenerate the reset pool (fresh random starts for auto-reset)
                # and reshuffle the GPU clip subset if the env supports it. Only a
                # real clip swap invalidates in-progress episodes (their stored
                # clip indices reference the old chunk), so only then do we reset
                # the live envs. Otherwise episodes run continuously across epochs.
                loop_rng, pool_rng = jax.random.split(loop_rng)
                reset_pool = jit_v_reset(jax.random.split(pool_rng, POOL_SIZE))
                swapped = False
                if hasattr(self.environment, 'swap_clips'):
                    swapped = bool(self.environment.swap_clips())
                if swapped:
                    loop_rng, reset_rng = jax.random.split(loop_rng)
                    wrapped_states = jit_v_reset(jax.random.split(reset_rng, NUM_ENVS))
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
                # across the while_loop trace level).
                action = jnp.clip(actor(obs), -1.0, 1.0)
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
        num_tests = int(self.test_episodes)
        max_steps = int(getattr(self.test_environment, "max_episode_steps", 1000))

        rng, keys_rng = jax.random.split(rng, 2)
        states = v_reset(jax.random.split(keys_rng, num_tests))

        if getattr(self, "_eval_fn", None) is None:
            self._eval_fn = self._make_eval_fn(v_step, num_tests, max_steps)

        scores, lengths = self._eval_fn(
            self.agent.state.actor, self.agent.state.obs_stats, states,
        )

        scores_np = np.array(scores)
        lengths_np = np.array(lengths)
        print(
            f"\nTest | score {np.mean(scores_np):.2f} | "
            f"length {np.mean(lengths_np):.1f} | "
            f"scores {scores_np.tolist()}",
            flush=True,
        )
