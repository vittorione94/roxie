import os
import time

import numpy as np
import jax
from myojit.utils import logger
import jax.numpy as jnp
from myojit.agents.ddpg import DDPG
from mujoco_playground import wrapper

class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, output_dir, steps=int(1e7), epoch_steps=int(5e4), save_steps=int(1e5),
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
        
        self.environment = wrapper.wrap_for_brax_training(environment)
        print("Environment wrapped for Brax training.", self.environment)

        self.test_environment = wrapper.wrap_for_brax_training(test_environment)
        print("Test environment wrapped for Brax training.", self.test_environment)

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
        
        reset_keys = jax.random.split(rngs.envs(), NUM_ENVS)
        local_devices_to_use = 1
        reset_keys = jnp.reshape(
            reset_keys, (local_devices_to_use, -1) + reset_keys.shape[1:]
        )
        # reset_keys shape: (1, 150) -> num GPU devices, num environments per device

        print("reset_keys shape:", reset_keys.shape)
        # Call the vectorized reset function to get the initial states for all envs.
        states = jit_v_reset(reset_keys)

        scores = jnp.zeros(NUM_ENVS)
        lengths = jnp.zeros(NUM_ENVS, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0
        
        while True:

            # Select actions.
            # Pass a key for exploration noise.
            actions = self.agent.step(states.obs, evaluate=False, key=rngs.agent())
            
            # TODO use chex
            #assert not np.isnan(actions.sum())

            # Take a step in the environments.
            next_states = jit_v_step(states, actions)
            new_buffer_state = self.agent.update(states, next_states, steps=self.steps, key=rngs.agent())

            scores += next_states.reward
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
            scores = jnp.where(next_states.done, 0, scores)
            lengths = jnp.where(next_states.done, 0, lengths)
            
            # Count the number of completed episodes by summing the boolean 'done' array
            # (where True=1, False=0) and add it to the total count.
            episodes = episodes + jnp.sum(next_states.done)

            # print(f"Steps: {self.steps}, "
            #       f"Epochs: {epochs}, "
            #       f"Episodes: {episodes}, "
            #       f"Scores: {scores.mean():.2f}, "
            #       f"Lengths: {lengths.mean():.2f}, "
            #       f"Time: {time.time() - start_time:.2f}s")
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
            state = jit_reset(key)

            while True:
                # Select an action.
                actions = self.agent.step(state.obs, evaluate=True, key=key)
                next_state = jit_step(state, actions)
                score += next_state.reward
                length += 1
                
                print(f"Test Episode: {_+1}, "
                      f"Score: {score:.2f}, "
                      f"Length: {length}")
                if next_state.done:
                    break
            scores.append(score)
            lengths.append(length)

        print(f"Test results: "
              f"Average score: {np.mean(scores):.2f}, "
              f"Average length: {np.mean(lengths):.2f}, "
              f"Scores: {scores}, Lengths: {lengths}")

