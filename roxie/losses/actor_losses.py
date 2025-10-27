import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent
import rlax

@nnx.jit
def ddpg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
):
    """Calculates the loss for the actor (aims to maximize Q-value)."""
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    q_values = critic_model(obs, actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


@nnx.jit
def ppo_loss_fn(
    actor_model,
    observations,
    old_log_probs,
    action_low,
    action_high,
    advantages,
    clip_epsilon,
    entropy_coef,
):
    """Calculates the loss for the actor using PPO clipped objective."""
    actions, logp_new, entropy = Agent.stochastic_step_fn(actor_model, observations)
    actions = Agent.scale_to_env(actions, action_low, action_high)

    ratio = jnp.exp(logp_new - old_log_probs)

    pg_loss = rlax.clipped_surrogate_pg_loss(ratio, advantages, clip_epsilon)

    return pg_loss - entropy_coef * entropy
