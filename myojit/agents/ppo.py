from myojit.agents.agent import Agent
import jax.numpy as jnp


class PPO(Agent):
    '''Proximal Policy Optimization agent.'''

    def __init__(self, 
                 env_obs_size: int,
                 env_action_size: int,
                 action_low: jnp.ndarray,
                 action_high: jnp.ndarray,
                 actor_config: dict,
                 critic_config: dict,
                 memory_config: dict,):
        pass

    def step(self):
        pass

    def update(self):
        pass

    def _export_hyperparams(self):
        return super()._export_hyperparams()
