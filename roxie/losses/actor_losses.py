import jax.numpy as jnp
from flax import nnx

from roxie.agents.agent import Agent

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
    actions_buf,
    old_log_probs,
    action_low,
    action_high,
    advantages,
    clip_epsilon,
    entropy_coef,
    key
):
    """Calculates the loss for the actor using PPO clipped objective."""

    distribution = actor_model(observations)
    logp_new = distribution.log_prob(actions_buf)       # (N, T)

    # Analytical entropy from the distribution
    entropy_t = distribution.entropy()[:, :-1]

    # Compute importance ratio aligned with advantages (skip last next-frame entry)
    ratio = jnp.exp(logp_new[:, :-1] - old_log_probs[:, :-1])

    # advantages --> (NUM_ENVS, BATCH_SIZE -1)
    # ratio --> (NUM_ENVS, BATCH_SIZE -1)
    # actions --> (NUM_ENVS, BATCH_SIZE, ACT_SIZE)
    # logp_new --> (NUM_ENVS, BATCH_SIZE)
    # old_log_probs --> (NUM_ENVS, BATCH_SIZE)

    # Use compatibility wrapper which handles arbitrary leading batch dims
    # and applies rlax per-time-step along the final axis.
    # pg_loss = rlax_compat.clipped_surrogate_pg_loss(ratio, advantages, clip_epsilon)

    if ratio.shape != advantages.shape:
        raise ValueError(f"ratio and advantages shapes must match; got {ratio.shape} vs {advantages.shape}")

    # Ensure there is a time axis
    if ratio.ndim < 1:
        raise ValueError("ratio must have at least 1 dimension (time axis)")

    # Compute clipped surrogate objective elementwise so the output keeps
    # the same shape as the inputs (i.e. per-timestep losses). rlax's
    # implementation may return a reduced scalar for a 1-D input, so
    # implement the per-step formula directly here to avoid ambiguity.
    #
    # surrogate = min(ratio * adv, clip(ratio, 1-eps, 1+eps) * adv)
    # loss = -surrogate  (we minimize loss; maximizing surrogate)

    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    surrogate = jnp.minimum(surrogate1, surrogate2)
    pg_loss = -surrogate

    # Per-step loss minus entropy term, then average across all dims
    per_step_loss = pg_loss - entropy_coef * entropy_t
    return jnp.mean(per_step_loss)
