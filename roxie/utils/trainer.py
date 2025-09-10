import os
import time

import numpy as np
import jax
from roxie.utils import logger
import jax.numpy as jnp
from flax import nnx
from roxie.agents.agent import Agent

class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

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

    def run(self, NUM_ENVS, rngs):
        '''Runs the main training loop.'''
            
        start_time = last_epoch_time = time.time()

        # --- Vectorization and JIT Compilation ---

        # Create a vectorized version of the environment's reset function.
        # jax.vmap will map the reset function over the first axis of its input (a batch of keys).
        v_reset = jax.vmap(self.environment.reset)

        # Create a vectorized version of the step function.
        # jax.vmap will map over the first axis of both the states and actions.
        v_step = jax.vmap(self.environment.step)

        # Apply JIT compilation to the vectorized functions for maximum performance.
        # This compiles the entire batched operation into a single optimized kernel.
        jit_reset = jax.jit(self.environment.reset)  # single-env reset
        jit_v_reset = jax.jit(v_reset)
        jit_v_step = jax.jit(v_step)

        # JIT the test envs outside the test function (batched over test_episodes)
        v_test_reset = jax.jit(jax.vmap(self.test_environment.reset))
        v_test_step = jax.jit(jax.vmap(self.test_environment.step))

        # Helper: lazily reset only the envs that are done (per-env cond, vmapped and jitted)
        def _selective_reset(done, key, state):
            # If this env is done, reset it with key; otherwise keep the new state.
            return jax.lax.cond(
                done,
                lambda k: jit_reset(k),     # reset returns a single-env wrapped state
                lambda _: state,            # keep the provided single-env state
                key,
            )

        v_selective_reset = jax.jit(jax.vmap(_selective_reset))

        print("Successfully created JIT-compiled, vectorized reset and step functions.")

        print("Initializing environments...")
        # Split the master key to get a unique key for each parallel environment.
        loop_rng = rngs.envs()
        reset_keys = jax.random.split(loop_rng, NUM_ENVS)
        wrapped_states = jit_v_reset(reset_keys)

        # Add a persistent RNG for agent updates
        agent_key = rngs.agent()

        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, int)
        self.steps, epoch_steps, epochs, episodes, tot_gradient_steps = 0, 0, 0, 0, 0
        steps_since_save = self.save_steps
        actor_losses = []
        critic_losses = []
        
        while True:
            # Split the main loop's RNG key for each iteration.
            loop_rng, action_key, reset_key_batch = jax.random.split(loop_rng, 3)

            # `wrapped_states` holds the state at time `t`.
            
            # Get actions for the current state (s_t).
            if hasattr(self.agent, "memory_warmup") and self.steps > getattr(self.agent, "memory_warmup", 0):
                actions = self.agent.step(wrapped_states.env_state.obs, evaluate=False, key=action_key)
            else:
                low = self.agent.action_low
                high = self.agent.action_high
                u = jax.random.uniform(action_key, (NUM_ENVS, self.environment.action_size), minval=0.0, maxval=1.0)
                actions = low + (high - low) * u
                self.agent.last_action = actions 
            
            # TODO use chex
            #assert not np.isnan(actions.sum())

            # Store the current state before it's overwritten. This is your "old state".
            old_wrapped_states = wrapped_states
            
            # Perform the step to get the "new state" (s_t+1).
            new_wrapped_states = jit_v_step(old_wrapped_states, actions)
            
            # Pass BOTH the old and new states to the agent for the full transition.
            # Split a fresh key for agent update every loop
            agent_key, update_key = jax.random.split(agent_key)
            gradient_steps, actor_loss, critic_loss = self.agent.update(
                old_wrapped_states.env_state, 
                new_wrapped_states.env_state, 
                steps=self.steps, 
                agent_rng=update_key,
            )
            actor_losses.append(actor_loss)
            critic_losses.append(critic_loss)

            tot_gradient_steps += gradient_steps

            dones = new_wrapped_states.env_state.done
            
            # Generate new keys for the environments that need resetting.
            reset_keys = jax.random.split(reset_key_batch, NUM_ENVS)
            
            # Lazily reset only the envs that are done, keep others as-is.
            # This avoids computing reset() for every env at every step.
            wrapped_states = v_selective_reset(dones, reset_keys, new_wrapped_states)

            scores += new_wrapped_states.env_state.reward
            lengths += 1
            self.steps += NUM_ENVS
            epoch_steps += NUM_ENVS
            steps_since_save += NUM_ENVS

            # Show the progress bar.
            if self.show_progress:
                logger.show_progress(
                    self.steps, self.epoch_steps, self.max_steps)
            
            # Check the finished episodes.
            # Where next_states.done is True, set scores and lengths to 0.
            # Otherwise, keep their original values.
            scores = jnp.where(new_wrapped_states.env_state.done, 0, scores)
            lengths = jnp.where(new_wrapped_states.env_state.done, 0, lengths)
            # Count the number of completed episodes by summing the boolean 'done' array
            # (where True=1, False=0) and add it to the total count.
            episodes = episodes + jnp.sum(new_wrapped_states.env_state.done)

            # End of the epoch.
            if epoch_steps >= self.epoch_steps:
                if hasattr(self.agent, "noise_module"):
                    self.agent.noise_module.reset_noise()  # reset noise process at epoch end

                # Evaluate the agent on the test environment.
                if self.test_environment and hasattr(self.agent, "state"):
                    self._test(rngs.envs(), v_test_reset, v_test_step)

                # Log the data.
                epochs += 1
                epoch_steps = 0

                print("\nEpoch Stats: \n"
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
                      f"    Critic loss: {np.mean(critic_losses):.2f} \n")
                actor_losses = []
                critic_losses = []  

                last_epoch_time = time.time()

            # End of training.
            stop_training = self.steps >= self.max_steps
            # Save a checkpoint.
            if stop_training or steps_since_save >= self.save_steps:
                path = os.path.join(self.output_dir, 'checkpoints')
                if os.path.isdir(path) and self.replace_checkpoint:
                    for file in os.listdir(path):
                        if file.startswith('step_'):
                            os.remove(os.path.join(path, file))
                checkpoint_name = f'step_{self.steps}'
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
            return dones_mask.reshape((dones_mask.shape[0],) + (1,) * max(leaf.ndim - 1, 0))

        # Use a regular Python while loop instead of nnx.while_loop
        # This avoids the trace context error
        while not jnp.all(dones):
            actions = self.agent.step(
                states.env_state.obs,
                evaluate=True,
                key=eval_key
            )

            next_states = v_step(states, actions)

            not_done = ~dones
            not_done_f32 = not_done.astype(jnp.float32)
            scores = scores + next_states.env_state.reward * not_done_f32
            lengths = lengths + not_done.astype(jnp.int32)

            states = jax.tree.map(
                lambda old, new: jnp.where(_broadcast_mask(dones, new), old, new)
                if isinstance(new, jnp.ndarray) and new.shape[:1] == dones.shape
                else new,
                states, next_states
            )
            dones = jnp.logical_or(dones, next_states.env_state.done)

            action_sum = action_sum + jnp.sum(actions)
            action_sumsq = action_sumsq + jnp.sum(jnp.square(actions))
            action_count = action_count + actions.size

        # Compute stats (host)
        scores_np = np.array(scores)
        lengths_np = np.array(lengths)

        act_mean = float(action_sum / jnp.maximum(1, action_count))
        act_var = jnp.maximum(0.0, action_sumsq / jnp.maximum(1, action_count) - act_mean * act_mean)
        act_std = float(jnp.sqrt(act_var))

        print(
            "\nTest results: \n"
            f"    Average score: {np.mean(scores_np):.2f} \n"
            f"    Average length: {np.mean(lengths_np):.2f} \n"
            f"    Scores: {scores_np.tolist()} \n"
            f"    Actions mean: {act_mean:.4f} \n"
            f"    Actions std: {act_std:.4f}"
        )

