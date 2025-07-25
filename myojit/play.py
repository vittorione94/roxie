import mujoco
import mujoco.viewer
import numpy as np
import random
import time
import argparse
import jax
from mujoco import mjx
from myojit.environment.loader import load_playground_env
import hydra
from omegaconf import DictConfig
from myojit.agents import agents


@hydra.main(version_base=None, config_path="configs", config_name="myojit")
def main(cfg: DictConfig):
    """
    Launches an interactive MuJoCo viewer with a random policy in a given environment.
    """

    # 1. Create a JAX random key
    key = jax.random.PRNGKey(seed=0)
    # Load the environment
    env, env_cfg = load_playground_env(cfg.env.env_name)
    
    # Get the standard MuJoCo model and data from the MJX-based environment
    model = env.mj_model
    data = mujoco.MjData(model)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # agent = agents[cfg.agent.name](**cfg.agent.params)
    agent = agents[cfg.agent.name]()

    agent.initialize(env.observation_size, env.action_size)

    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the environment to get the initial state
        key, reset_key = jax.random.split(key)
        state = jit_reset(rng=reset_key)
        mujoco.mj_forward(model, data)

        # Run the simulation loop
        while viewer.is_running():
            step_start = time.time()

            # Take a random action
            action = agent.step([state.obs], 1)

            # Step the environment
            state = jit_step(state, action)

            new_data = mjx.get_data(model, state.data)

            data.qpos = new_data.qpos
            data.qvel = new_data.qvel
            data.ctrl = action
            mujoco.mj_forward(model, data)

            # Sync the viewer with the new data
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == '__main__':
    main()