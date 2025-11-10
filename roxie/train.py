import copy

import hydra
import jax
import jax.numpy as jnp
from flax import nnx
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from roxie.agents import agents
from roxie.environment.loader import load_playground_env
from roxie.utils.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="roxie")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())

    env, env_cfg = load_playground_env(cfg.env.env_name)
    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    # Create RNGs for agent initialization
    training_rngs = nnx.Rngs(envs=cfg.env.seed, agent=3)  # Use your seed from cfg.seed

    # Agent handles all component instantiation internally
    ctrl_range = jnp.array(env.mj_model.actuator_ctrlrange)  # shape (action_dim, 2)
    action_low = ctrl_range[:, 0]
    action_high = ctrl_range[:, 1]

    agent_args = {
        "env_obs_size": env.observation_size,
        "env_action_size": env.action_size,
        "action_low": action_low,
        "action_high": action_high,
        **cfg.agent.args,
    }
    if "actor" in cfg.agent:
        agent_args["actor_config"] = cfg.agent.actor
    if "critic" in cfg.agent:
        agent_args["critic_config"] = cfg.agent.critic
    if "memory" in cfg.agent:
        agent_args["memory_config"] = cfg.agent.memory
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = agents[cfg.agent.name](**agent_args)

    trainer = Trainer(
        output_dir=output_dir,
        steps=int(1e9),
        epoch_steps=int(1e5),
        save_steps=int(5e5),
        test_episodes=5,
        show_progress=True,
        replace_checkpoint=False,
    )
    trainer.initialize(
        agent=agent, environment=env, test_environment=copy.deepcopy(env)
    )
    trainer.run(cfg.env.parallel_envs, training_rngs)

    return


if __name__ == "__main__":
    main()
