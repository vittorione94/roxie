import os
import time

import numpy as np
import jax


class Trainer:
    '''Trainer used to train and evaluate an agent on an environment.'''

    def __init__(
        self, steps=int(1e7), epoch_steps=int(2e4), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,
    ):
        self.max_steps = steps
        self.epoch_steps = epoch_steps
        self.save_steps = save_steps
        self.test_episodes = test_episodes
        self.show_progress = show_progress
        self.replace_checkpoint = replace_checkpoint

    def initialize(self, agent, environment, test_environment=None):
        self.agent = agent
        self.environment = environment
        self.test_environment = test_environment

    def run(self, NUM_ENVS):
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
        key = jax.random.PRNGKey(seed=0)
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, NUM_ENVS)
        # Call the vectorized reset function to get the initial states for all envs.
        states = jit_v_reset(reset_keys)

        scores = np.zeros(NUM_ENVS)
        lengths = np.zeros(NUM_ENVS, int)
        self.steps, epoch_steps, epochs, episodes = 0, 0, 0, 0
        steps_since_save = 0

        while True:
            # Select actions.
            actions = self.agent.step(states.obs, evaluate=False)
            
            # TODO use chex
            #assert not np.isnan(actions.sum())

            # Take a step in the environments.
            next_states = jit_v_step(states, actions)
            self.agent.update(next_states, steps=self.steps)

            # scores += infos['rewards']
            # lengths += 1
            # self.steps += num_workers
            # epoch_steps += num_workers
            # steps_since_save += num_workers

            # # Show the progress bar.
            # if self.show_progress:
            #     logger.show_progress(
            #         self.steps, self.epoch_steps, self.max_steps)

            # # Check the finished episodes.
            # for i in range(num_workers):
            #     if infos['resets'][i]:
            #         scores[i] = 0
            #         lengths[i] = 0
            #         episodes += 1

            # # End of the epoch.
            # if epoch_steps >= self.epoch_steps:
            #     # Evaluate the agent on the test environment.
            #     if self.test_environment:
            #         self._test()

            #     # Log the data.
            #     epochs += 1
            #     epoch_steps = 0

            # End of training.
            stop_training = self.steps >= self.max_steps
            print("setps", self.steps)
            # Save a checkpoint.
            # if stop_training or steps_since_save >= self.save_steps:
            #     path = os.path.join(logger.get_path(), 'checkpoints')
            #     if os.path.isdir(path) and self.replace_checkpoint:
            #         for file in os.listdir(path):
            #             if file.startswith('step_'):
            #                 os.remove(os.path.join(path, file))
            #     checkpoint_name = f'step_{self.steps}'
            #     save_path = os.path.join(path, checkpoint_name)
            #     self.agent.save(save_path)
            #     steps_since_save = self.steps % self.save_steps

            if stop_training:
                break

    # def _test(self):
    #     '''Tests the agent on the test environment.'''

    #     # Start the environment.
    #     if not hasattr(self, 'test_observations'):
    #         self.test_observations = self.test_environment.start()
    #         assert len(self.test_observations) == 1

    #     # Test loop.
    #     for _ in range(self.test_episodes):
    #         score, length = 0, 0

    #         while True:
    #             # Select an action.
    #             actions = self.agent.test_step(
    #                 self.test_observations, self.steps)
    #             assert not np.isnan(actions.sum())
    #             logger.store('test/action', actions, stats=True)

    #             # Take a step in the environment.
    #             self.test_observations, infos = self.test_environment.step(
    #                 actions)
    #             self.agent.test_update(**infos, steps=self.steps)

    #             score += infos['rewards'][0]
    #             length += 1

    #             if infos['resets'][0]:
    #                 break

    #         # Log the data.
    #         logger.store('test/episode_score', score, stats=True)
    #         logger.store('test/episode_length', length, stats=True)