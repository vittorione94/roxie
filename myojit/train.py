import hydra
from omegaconf import DictConfig
from myojit.agents import agents
from myojit.environment.loader import load_playground_env
from myojit.replays.buffer import JaxReplayBuffer
from flax import nnx

@hydra.main(version_base=None, config_path="configs", config_name="myojit")
def main(cfg: DictConfig):
    print(cfg.agent.name)

    env, env_cfg = load_playground_env(cfg.env.env_name)
    action_dim = env.action_size
    rngs = nnx.Rngs(params=0, dropout=1) # Use your seed from cfg.seed

    # A more direct check:
    actor = hydra.utils.instantiate(
            cfg.model.actor, 
            action_dim=action_dim,
            rngs=rngs)
    # critic = hydra.utils.instantiate(
    #     cfg.model.critic
    # )
    
    replay = JaxReplayBuffer(
        capacity=cfg.agent.buffer_size,
        batch_size=cfg.agent.batch_size
    )


    agent = agents[cfg.agent.name](actor, None, replay)

    return

if __name__ == '__main__':
    main()