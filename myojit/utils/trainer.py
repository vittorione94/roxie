import os
import time

import numpy as np
import jax
from myojit.utils import logger
import jax.numpy as jnp
from myojit.agents.ddpg import DDPG


class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(1e5), save_steps=int(1e5),
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
        jit_v_reset = jax.jit(v_reset)
        jit_v_step = jax.jit(v_step)

        print("Successfully created JIT-compiled, vectorized reset and step functions.")

        # 1. Initialize the environments
        print("Initializing environments...")
        # Split the master key to get a unique key for each parallel environment.
        loop_rng = rngs.envs()
        reset_keys = jax.random.split(loop_rng, NUM_ENVS)

        print("reset_keys shape:", reset_keys.shape)
        # Call the vectorized reset function to get the initial states for all envs.
        wrapped_states = jit_v_reset(reset_keys)

        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0
        
        while True:
            # Split the main loop's RNG key for each iteration.
            loop_rng, action_key, update_key, reset_key_batch = jax.random.split(loop_rng, 4)

            # `wrapped_states` holds the state at time `t`.
            
            # 1. Get actions for the current state (s_t).
            actions = self.agent.step(wrapped_states.env_state.obs, evaluate=False, key=action_key)
            # TODO use chex
            #assert not np.isnan(actions.sum())

            # 2. Store the current state before it's overwritten. This is your "old state".
            old_wrapped_states = wrapped_states
            
            # 3. Perform the step to get the "new state" (s_t+1).
            new_wrapped_states = jit_v_step(old_wrapped_states, actions)
            
            # 4. Pass BOTH the old and new states to the agent for the full transition.
            new_buffer_state = self.agent.update(
                old_wrapped_states.env_state, 
                new_wrapped_states.env_state, 
                steps=self.steps, 
                key=update_key
            )

            dones = new_wrapped_states.env_state.done
            
            # 2. Generate new keys for the environments that need resetting.
            reset_keys = jax.random.split(reset_key_batch, NUM_ENVS)
            
            # 3. Get a batch of *potential* new states by calling the reset function.
            #    We do this for all environments; `where` will select only the needed ones.
            reset_states = jit_v_reset(reset_keys)
            
            # 4. The main event: Use tree_map and where to create the true next state.
            #    For each leaf in the state PyTree, it picks from `reset_states` if done,
            #    otherwise it keeps the state from `new_wrapped_states`.
            final_states = jax.tree.map(
                lambda reset_leaf, next_leaf: jnp.where(
                    dones.reshape((dones.shape[0],) + (1,) * (reset_leaf.ndim - 1)), # Ensure `dones` broadcasts correctly to array shapes
                    reset_leaf,
                    next_leaf
                ),
                reset_states,
                new_wrapped_states
            )
            
            # 5. CRITICAL: Update the main state variable for the next loop iteration.
            wrapped_states = final_states

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

            # # End of the epoch.
            if epoch_steps >= self.epoch_steps:

                # Evaluate the agent on the test environment.
                if self.test_environment:
                    self._test(rngs.envs())

                # Log the data.
                epochs += 1
                epoch_steps = 0

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

    def _test(self, key):
        '''Tests the agent on the test environment.'''
        scores, lengths = [], []

        jit_reset = jax.jit(self.test_environment.reset)
        jit_step = jax.jit(self.test_environment.step)

        # Test loop.
        for _ in range(self.test_episodes):
            score, length = 0, 0
            # CORRECT: Initialize a 'current_state' that will be updated
            current_state = jit_reset(key)

            while True:
                # Select an action.
                actions = self.agent.step(current_state.env_state.obs, evaluate=True, key=key)
                current_state = jit_step(current_state, actions)
                score += current_state.env_state.reward
                length += 1
   
                if current_state.env_state.done:
                    break
            scores.append(score)
            lengths.append(length)

        print(f"Test results: "
              f"Average score: {np.mean(scores):.2f}, "
              f"Average length: {np.mean(lengths):.2f}, "
              f"Scores: {scores}, Lengths: {lengths}")

