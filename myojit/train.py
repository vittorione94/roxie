import hydra
from omegaconf import DictConfig
from myojit.agents import agents
from myojit.environment.loader import load_playground_env
from myojit.replays.buffer import JaxReplayBuffer
from flax import nnx
from myojit.utils.trainer import Trainer
import jax.numpy as jnp
import jax
from myojit.replays.buffer import Transition
from hydra.core.hydra_config import HydraConfig
import copy

@hydra.main(version_base=None, config_path="configs", config_name="myojit")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    print("JAX devices:", jax.devices())
    print("JAX platform:", jax.default_backend())
    
    # Test array placement
    test_array = jnp.ones(3)
    print("Test array device:", test_array.devices())


    env, env_cfg = load_playground_env(cfg.env.env_name)

    print("Environment configuration:", env_cfg)

    output_dir = HydraConfig.get().runtime.output_dir

    prototype = Transition(
        observation=jnp.zeros(env.observation_size, dtype=jnp.float32),
        action=jnp.zeros(env.action_size, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_observation=jnp.zeros(env.observation_size, dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_)
    )

    replay = hydra.utils.instantiate(
        cfg.agent.memory,
        capacity=cfg.agent.memory.capacity,
        batch_size=cfg.agent.memory.batch_size
    )
    buffer_state = replay.init(prototype)
    
    action_dim = env.action_size
    rngs = nnx.Rngs(params=0, dropout=1, envs=2, agent=3) # Use your seed from cfg.seed

    # A more direct check:
    actor = hydra.utils.instantiate(
            cfg.model.actor,
            in_features=env.observation_size,
            action_dim=action_dim,
            rngs=rngs)
    critic = hydra.utils.instantiate(
        cfg.model.critic,
        in_features=env.observation_size + action_dim,
        rngs=rngs
    )
    
    ctrl_range = jnp.array(env.mj_model.actuator_ctrlrange)  # shape (action_dim, 2)
    action_low = ctrl_range[:, 0]
    action_high = ctrl_range[:, 1]
    agent = agents[cfg.agent.name](actor, critic, replay, buffer_state, \
                                   action_low=action_low, action_high=action_high, **cfg.agent.args)

    trainer = Trainer(output_dir=output_dir, steps=int(1e7), epoch_steps=int(1e5), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,)
    trainer.initialize(agent=agent, environment=env, test_environment=copy.deepcopy(env))
    trainer.run(cfg.parallel_envs, rngs) 

    return

if __name__ == '__main__':
    main()