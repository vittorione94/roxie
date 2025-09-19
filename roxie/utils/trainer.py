import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from roxie.agents.agent import Agent
from roxie.utils import logger
from functools import partial

class Trainer:
    """Trainer used to train and evaluate an agent on an environment."""

    def __init__(
        self,
        output_dir,
        steps=int(1e7),
        epoch_steps=int(3e5),
        save_steps=int(1e5),
        test_episodes=5,
        show_progress=True,
        replace_checkpoint=False,
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

    @partial(jax.jit, static_argnums=(0,))
    def _collect_step(self, carry):
        """
        Collect a single environment step across all vectorized envs.
        
        Args:
            carry: (env_states, agent_state, rng, scores, lengths)
                env_states: [n_envs] wrapped environment states
                agent_state: agent's internal state (networks, optimizers, memory)
                rng: random key
                scores: [n_envs] episode returns
                lengths: [n_envs] episode lengths
        
        Returns:
            Updated carry tuple
        """
        env_states, agent_state, rng, scores, lengths = carry
        
        # Split keys for this iteration
        rng, action_key, reset_key, update_key = jax.random.split(rng, 4)
        
        # Agent selects actions - shape: [n_envs, action_dim]
        actions = self.agent.step(
            env_states.obs,  # [n_envs, obs_dim]
            evaluate=False,
            key=action_key
        )
        
        # Environment step (vectorized) - returns new wrapped states
        next_env_states = self.v_step(env_states, actions)
        
        # Agent processes transition and potentially updates
        # For off-policy: stores in replay buffer, samples and updates if ready
        # For on-policy: accumulates trajectory, updates when buffer full
        gradient_steps, actor_loss, critic_loss = self.agent.update(
            env_states,      # old states
            actions,         # [n_envs, action_dim]
            next_env_states, # new states including rewards/dones
            update_key
        )
        
        # Update metrics
        scores = scores + next_env_states.reward  # [n_envs]
        lengths = lengths + 1
        
        # Reset done environments lazily
        def reset_if_done(env_state, done, key):
            """Reset single env conditionally"""
            return jax.lax.cond(
                done,
                lambda k: self.env.reset(k),
                lambda _: env_state,
                key
            )
        
        reset_keys = jax.random.split(reset_key, self.n_envs)
        env_states = jax.vmap(reset_if_done)(
            next_env_states, 
            next_env_states.done,
            reset_keys
        )
        
        # Reset metrics for done episodes
        scores = jnp.where(next_env_states.done, 0.0, scores)
        lengths = jnp.where(next_env_states.done, 0, lengths)
        
        return (env_states, agent_state, rng, scores, lengths), next_env_states.done
    
    def run(self, NUM_ENVS, rngs):
        """Runs the main training loop."""
        start_time = last_epoch_time = time.time()

        # --- Vectorization and JIT Compilation ---
        # Create a vectorized version of the environment's reset function.
        # jax.vmap will map the reset function over the first axis of its input (a batch of keys).
        self.v_reset = jax.jit(jax.vmap(self.environment.reset))

        # Create a vectorized version of the step function.
        # jax.vmap will map over the first axis of both the states and actions.
        self.v_step = jax.jit(jax.vmap(self.environment.step))

        # JIT the test envs outside the test function (batched over test_episodes)
        self.v_test_reset = jax.jit(jax.vmap(self.test_environment.reset))
        self.v_test_step = jax.jit(jax.vmap(self.test_environment.step))

        print("Successfully created JIT-compiled, vectorized reset and step functions.")

        print("Initializing environments...")
        # Split the master key to get a unique key for each parallel environment.
        loop_rng = rngs.envs()
        reset_keys = jax.random.split(loop_rng, NUM_ENVS)
        wrapped_states = self.v_reset(reset_keys)

        # Add a persistent RNG for agent updates
        agent_key = rngs.agent()

        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, int)
        self.steps, epoch_steps, epochs, episodes, tot_gradient_steps = 0, 0, 0, 0, 0
        steps_since_save = self.save_steps
        actor_losses = []
        critic_losses = []

        #  Determine collection size based on algorithm type
        # On-policy: collect full trajectories before update
        # Off-policy: collect single steps (agent handles replay internally)
        collect_steps = self.agent.collect_steps if hasattr(self.agent, 'collect_steps') else 1

        while True:
            # Collect experience
            carry = (env_states, agent_state, rng, scores, lengths)
            (env_states, agent_state, rng, scores, lengths), dones = jax.lax.scan(
                self._collect_step,
                carry,
                None,
                length=collect_steps
            )

            self.steps += NUM_ENVS * collect_steps

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(self.steps, self.epoch_steps, self.max_steps)

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                if hasattr(self.agent, "noise_module"):
                    self.agent.noise_module.reset_noise()  # reset noise process at epoch end

                # Evaluate the agent on the test environment.
                if self.test_environment and hasattr(self.agent, "state"):
                    self._test(rngs.envs(), self.v_test_reset, self.v_test_step)

                # Log the data.
                epochs += 1
                epoch_steps = 0

                print(
                    "\nEpoch Stats: \n"
                    f"    Epoch: {epochs} \n"
                    f"    Steps: {self.steps} \n"
                    f"    Episodes: {episodes} \n"
                    f"    Time: {time.time() - start_time:.2f} \n"
                    f"    Epoch time: {time.time() - last_epoch_time:.2f} \n"
                    f"    Steps per second: {self.steps / (time.time() - start_time):.2f} \n"
                    #   f"    Warmup: {self.steps < self.agent.memory_warmup} \n"
                    f"    Average score: {jnp.mean(scores):.2f} \n"
                    f"    Average length: {jnp.mean(lengths):.2f} \n"
                    f"    Gradient steps: {tot_gradient_steps} \n"
                    f"    Actor loss: {np.mean(actor_losses):.2f} \n"
                    f"    Critic loss: {np.mean(critic_losses):.2f} \n"
                )
                actor_losses = []
                critic_losses = []

                last_epoch_time = time.time()

            # End of training.
            stop_training = self.steps >= self.max_steps
            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(self.output_dir, "checkpoints")
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith("step_"):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f"step_{self.steps}"
                save_path = os.path.join(path, checkpoint_name)
                self.agent.save(save_path)
                steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

    def _test(self, rng, v_reset, v_step):
        """Vectorized test loop, runs fully on device."""

        num_tests = int(self.test_episodes)

        # Reset all test envs in parallel
        rng, keys_rng = jax.random.split(rng, 2)
        reset_keys = jax.random.split(keys_rng, num_tests)
        states = v_reset(reset_keys)

        # Initial accumulators
        dones = jnp.zeros((num_tests,), dtype=bool)
        scores = jnp.zeros((num_tests,), dtype=jnp.float32)
        lengths = jnp.zeros((num_tests,), dtype=jnp.int32)

        # Accumulators for action stats
        action_sum = jnp.array(0.0, dtype=jnp.float32)
        action_sumsq = jnp.array(0.0, dtype=jnp.float32)
        action_count = jnp.array(0, dtype=jnp.int32)

        eval_key = rng  # disables exploration noise

        def _broadcast_mask(dones_mask, leaf):
            # Make mask shape (B, 1, 1, ..., 1) to match each leaf's rank
            return dones_mask.reshape(
                (dones_mask.shape[0],) + (1,) * max(leaf.ndim - 1, 0)
            )

        # Use a regular Python while loop instead of nnx.while_loop
        # This avoids the trace context error
        while not jnp.all(dones):
            actions = self.agent.step(states.env_state.obs, evaluate=True, key=eval_key)

            next_states = v_step(states, actions)

            not_done = ~dones
            not_done_f32 = not_done.astype(jnp.float32)
            scores = scores + next_states.env_state.reward * not_done_f32
            lengths = lengths + not_done.astype(jnp.int32)

            states = jax.tree.map(
                lambda old, new: (
                    jnp.where(_broadcast_mask(dones, new), old, new)
                    if isinstance(new, jnp.ndarray) and new.shape[:1] == dones.shape
                    else new
                ),
                states,
                next_states,
            )
            dones = jnp.logical_or(dones, next_states.env_state.done)

            action_sum = action_sum + jnp.sum(actions)
            action_sumsq = action_sumsq + jnp.sum(jnp.square(actions))
            action_count = action_count + actions.size

        # Compute stats (host)
        scores_np = np.array(scores)
        lengths_np = np.array(lengths)

        act_mean = float(action_sum / jnp.maximum(1, action_count))
        act_var = jnp.maximum(
            0.0, action_sumsq / jnp.maximum(1, action_count) - act_mean * act_mean
        )
        act_std = float(jnp.sqrt(act_var))

        print(
            "\nTest results: \n"
            f"    Average score: {np.mean(scores_np):.2f} \n"
            f"    Average length: {np.mean(lengths_np):.2f} \n"
            f"    Scores: {scores_np.tolist()} \n"
            f"    Actions mean: {act_mean:.4f} \n"
            f"    Actions std: {act_std:.4f}"
        )
