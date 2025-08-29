
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

# @nnx.jit
def ppo_loss_fn(actor_model, critic_model, samples, obs_mean, obs_std, obs_clip, action_low, action_high, clip_epsilon=0.2):
    """Calculates the loss for the actor using PPO clipped objective."""
    obs = Agent.normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)                                  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)
    q_values = critic_model(obs, actions)
    ratios = jnp.exp(q_values - jnp.mean(q_values))
    ppo_loss = -jnp.mean(ratios * jnp.clip(ratios / (1 + clip_epsilon), 0, 1))
    return ppo_loss