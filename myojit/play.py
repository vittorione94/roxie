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
from omegaconf import OmegaConf
from myojit.agents import agents
import os 
import click
import hydra
from omegaconf import DictConfig
from myojit.agents import agents
from myojit.environment.loader import load_playground_env
from myojit.replays.buffer import JaxReplayBuffer
from flax import nnx
from myojit.utils.trainer import Trainer
import jax.numpy as jnp
from myojit.replays.buffer import Transition
from hydra.core.hydra_config import HydraConfig


@click.command()
@click.option('--checkpoint_path', type=str, help='Path to the checkpoint file.')
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
    
    prototype = Transition(
        observation=jnp.zeros(env.observation_size, dtype=jnp.float32),
        action=jnp.zeros(env.action_size, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_observation=jnp.zeros(env.observation_size, dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_)
    )

    replay = hydra.utils.instantiate(
        cfg.agent.memory
    )
    buffer_state = replay.init(prototype)
    
    action_dim = env.action_size
    actor_rngs = nnx.Rngs(params=0, dropout=1)
    critic_rngs = nnx.Rngs(params=0, dropout=1)

    # A more direct check:
    actor = hydra.utils.instantiate(
            cfg.model.actor,
            in_features=env.observation_size,
            action_dim=action_dim,
            rngs=actor_rngs)
    
    critic = hydra.utils.instantiate(
        cfg.model.critic,
        in_features=env.observation_size + action_dim,
        rngs=critic_rngs
    )
    noise_module = hydra.utils.instantiate(
        cfg.noise,
        action_shape=(action_dim,),
    )


    # Get the standard MuJoCo model and data from the MJX-based environment
    model = env.mj_model
    data = mujoco.MjData(model)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # When loading the agent, ensure it's on CPU
    # with jax.default_device(jax.devices('cpu')[0]):
    agent = agents[cfg.agent.name].load(checkpoint_path, actor=actor, critic=critic, replay=replay, noise_module=noise_module)

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
            action = agent.step(obs_b, evaluate=True, key=key)   # shape (1, act_dim)

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

        

if __name__ == '__main__':
    main()