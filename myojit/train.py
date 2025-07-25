import hydra
from omegaconf import DictConfig
from myojit.agents import agents
from myojit.environment.loader import load_playground_env
from myojit.replays.buffer import JaxReplayBuffer
from flax import nnx
from myojit.utils.trainer import Trainer


@hydra.main(version_base=None, config_path="configs", config_name="myojit")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    env, env_cfg = load_playground_env(cfg.env.env_name)

    prototype = Transition(
        observation=jnp.zeros(OBS_DIM, dtype=jnp.float32),
        action=jnp.zeros(ACTION_DIM, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        next_observation=jnp.zeros(OBS_DIM, dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_)
    )

    replay = JaxReplayBuffer(
        capacity=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size
    )

    replay.init(prototype)
    

    action_dim = env.action_size
    #TODO: study properly this
    rngs = nnx.Rngs(params=0, dropout=1) # Use your seed from cfg.seed

    # A more direct check:
    actor = hydra.utils.instantiate(
            cfg.model.actor, 
            action_dim=action_dim,
            rngs=rngs)
    # critic = hydra.utils.instantiate(
    #     cfg.model.critic
    # )
    
    
    agent = agents[cfg.agent.name](actor, None, replay)

    trainer = Trainer(steps=int(1e7), epoch_steps=int(2e4), save_steps=int(5e5),
        test_episodes=5, show_progress=True, replace_checkpoint=False,)
    trainer.initialize(agent=agent, environment=env, test_environment=None)
    trainer.run(num_workers)

    return

if __name__ == '__main__':
    main()