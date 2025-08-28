
import jax.numpy as jnp
from myojit.agents.agent import Agent
from flax import nnx

@nnx.jit
def ddpg_actor_loss_fn(actor_model, critic_model, samples, obs_mean, obs_std, obs_clip, action_low, action_high):
    """Calculates the loss for the actor (aims to maximize Q-value)."""
    obs = Agent.normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)                                  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)   # [low, high]
    q_values = critic_model(obs, actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss

