
import jax.numpy as jnp
from roxie.agents.agent import Agent
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

@nnx.jit
def ppo_loss_fn(actor_model, critic_model, samples, obs_mean, obs_std, obs_clip, action_low, action_high, clip_epsilon=0.2):
    """Calculates the loss for the actor using PPO clipped objective."""
    obs = Agent.normalize_obs(samples['observations'], obs_mean, obs_std, obs_clip)
    actions, logp_new = Agent.stochastic_step_fn(actor_model, obs)
    actions = Agent.scale_to_env(actions, action_low, action_high)
    
    logp_old = samples['log_probs']
    ratio = jnp.exp(logp_new - logp_old)

    clipped_ratio = jnp.clip(ratio, 1 - clip_epsilon, 1 + clip_epsilon)

    # unclipped surrogate
    surr1 = ratio * advantages

    # clipped surrogate
    surr2 = clipped_ratio * advantages

    return -jnp.mean(jnp.maximum(surr1, surr2))
