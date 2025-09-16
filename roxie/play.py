import os
import time

import click
import hydra
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from roxie.agents import agents
from roxie.environment.loader import load_playground_env
from roxie.utils.trainer import Trainer


@click.command()
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint file.")
def main(checkpoint_path):
    """
    Launches an interactive MuJoCo viewer with a random policy in a given environment.
    """

    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)

    # 1. Create a JAX random key
    key = jax.random.PRNGKey(seed=0)

    # Load the environment
    env, env_cfg = load_playground_env(cfg.env.env_name)

    agent_args = {}
    if "actor" in cfg.agent:
        agent_args["actor_config"] = cfg.agent.actor
    if "critic" in cfg.agent:
        agent_args["critic_config"] = cfg.agent.critic
    if "memory" in cfg.agent:
        agent_args["memory_config"] = cfg.agent.memory
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = agents[cfg.agent.name].load(path=checkpoint_path, **agent_args)

    # Get the standard MuJoCo model and data from the MJX-based environment
    model = env.mj_model
    data = mujoco.MjData(model)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # Launch the interactive viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset the environment to get the initial state
        key, reset_key = jax.random.split(key)
        wrapped_state = jit_reset(key=reset_key)
        mujoco.mj_forward(model, data)

        score = 0.0
        actions = []

        # Run the simulation loop
        while viewer.is_running():
            step_start = time.time()

            # Take a policy action
            obs_b = jnp.expand_dims(wrapped_state.env_state.obs, axis=0)
            action = agent.step(obs_b, evaluate=True, key=key)  # shape (1, act_dim)

            # Step the environment
            wrapped_state = jit_step(wrapped_state, action[0])

            # new_data = mjx.get_data(model, wrapped_state.env_state.data)

            data.qpos = wrapped_state.env_state.data.qpos
            data.qvel = wrapped_state.env_state.data.qvel
            data.ctrl = action
            mujoco.mj_forward(model, data)

            score += wrapped_state.env_state.reward
            actions.append(action)

            if wrapped_state.env_state.done:
                print(f"Total score: {score}")
                print(f"Actions mean: {jnp.mean(jnp.array(actions)):.2f}")
                print(f"Actions std: {jnp.std(jnp.array(actions)):.2f}")
                key, reset_key = jax.random.split(reset_key)
                wrapped_state = jit_reset(key=reset_key)
                mujoco.mj_resetData(model, data)
                score = 0.0
                actions = []

            # Sync the viewer with the new data
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
