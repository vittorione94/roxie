import jax
import jax.numpy as jnp
from roxie.agents.agent import Agent
from flax import nnx

@nnx.jit
def ddpg_critic_loss_fn(critic_model, target_actor_model, target_critic_model, samples, \
                    gamma, noise_key, target_policy_noise, target_noise_clip,  action_low, action_high, \
                    obs_mean, obs_std, obs_clip):
    """Calculates the MSE loss for the critic."""
    # Normalize observations
    obs = Agent.normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(samples['next_observations'], obs_mean, obs_std, obs_clip)

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)                 # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target smoothing noise in env units
    act_span = (action_high - action_low)
    noise = jax.random.normal(noise_key, next_actions.shape) * (target_policy_noise * act_span)
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)

    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    next_q = target_critic_model(next_obs, next_actions)
    
    term = samples['terminals'].astype(jnp.float32)
    reward = jnp.squeeze(samples['rewards'])
    next_q = jnp.squeeze(next_q)
    target_q = reward + gamma * (1.0 - term) * next_q
    target_q = jax.lax.stop_gradient(target_q)

    current_q = critic_model(obs, samples['actions'])
    
    critic_loss = jnp.mean((jnp.squeeze(current_q) - target_q)**2)
    return critic_loss

@nnx.jit
def ppo_critic_loss_fn(critic_model, samples, \
                    gamma, action_low, action_high, \
                    obs_mean, obs_std, obs_clip):
    """Calculates the MSE loss for the critic using PPO."""
    
    obs = samples['observations']
    nxt_obs = samples['next_observations']
    reward = jnp.squeeze(samples['rewards'])
    term = samples['terminals'].astype(jnp.float32)
    

    v = critic_model(obs).squeeze(-1)         # [N]
    v_next = critic_model(nxt_obs).squeeze(-1)# [N]
