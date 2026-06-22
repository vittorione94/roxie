import jax
import jax.numpy as jnp
import rlax
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
def td3_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
):
    """Deterministic policy gradient through the first critic head (TD3)."""
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    q1, _ = twin_critic(obs, actions)
    actor_loss = -jnp.mean(q1)
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


@nnx.jit
def sac_actor_loss_fn(
    actor_model,
    twin_critic,
    alpha,
    samples,
    key,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
):
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    distribution = actor_model(obs)
    u = distribution.sample(seed=key)
    actions = jnp.tanh(u)
    log_probs = distribution.log_prob(u) - jnp.sum(
        jnp.log(1.0 - actions ** 2 + 1e-6), axis=-1
    )
    actions_scaled = Agent.scale_to_env(actions, action_low, action_high)
    q1, q2 = twin_critic(obs, actions_scaled)
    min_q = jnp.minimum(jnp.squeeze(q1), jnp.squeeze(q2))
    actor_loss = jnp.mean(alpha * log_probs - min_q)
    return actor_loss, log_probs


def mpo_actor_loss_fn(
    actor_model,
    dual_params,
    target_actor_model,
    critic_model,
    samples,
    key,
    num_action_samples,
    epsilon,
    epsilon_mean,
    epsilon_stddev,
    action_low,
    action_high,
    obs_mean,
    obs_std,
    obs_clip,
):
    """MPO policy improvement loss (E-step + M-step).

    E-step: build a nonparametric improved policy by reweighting actions sampled
    from the target (old) policy with ``softmax(Q / temperature)``, where the
    temperature is found by minimizing the convex dual of a hard KL bound
    (``epsilon``).

    M-step: fit the parametric policy to those weighted samples by weighted
    maximum likelihood, subject to a *decoupled* KL trust region — the mean and
    the covariance of the Gaussian are constrained separately (``epsilon_mean``,
    ``epsilon_stddev``), each with its own per-dimension Lagrange multiplier.

    The dual variables (temperature and the two KL multipliers) live in
    ``dual_params`` and are optimized jointly with the actor by this same loss.
    Gradients are taken w.r.t. ``actor_model`` and ``dual_params``.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)

    # Sample N actions per state from the target (old) policy: [S, B, A].
    target_dist = target_actor_model(obs)
    actions = target_dist.sample(seed=key, sample_shape=(num_action_samples,))

    # Evaluate the online critic on those actions (env scale). No gradient flows
    # into the policy through Q; it only shapes the E-step weights / temperature.
    actions_scaled = Agent.scale_to_env(
        jnp.clip(actions, -1.0, 1.0), action_low, action_high
    )
    obs_tiled = jnp.broadcast_to(obs, (num_action_samples,) + obs.shape)
    q_values = critic_model(obs_tiled, actions_scaled)  # [S, B, 1]
    q_values = jax.lax.stop_gradient(jnp.squeeze(q_values, axis=-1))  # [S, B]

    # M-step likelihood of the sampled actions under the *online* policy.
    online_dist = actor_model(obs)
    log_probs = online_dist.log_prob(actions)  # [S, B]

    # Decoupled, per-dimension KL between the target and online Gaussians.
    # KL_mean varies the mean while holding the target std fixed; KL_stddev
    # varies the std while holding the target mean fixed.
    mu_t = jax.lax.stop_gradient(target_dist.loc)
    sig_t = jax.lax.stop_gradient(target_dist.scale_diag)
    mu_o = online_dist.loc
    sig_o = online_dist.scale_diag

    kl_mean = jnp.square(mu_t - mu_o) / (2.0 * jnp.square(sig_t))  # [B, A]
    kl_stddev = (
        jnp.log(sig_o / sig_t)
        + jnp.square(sig_t) / (2.0 * jnp.square(sig_o))
        - 0.5
    )  # [B, A]

    # Project the raw dual params into the positive range.
    temperature = jax.nn.softplus(dual_params.log_temperature.value)
    alpha_mean = jax.nn.softplus(dual_params.log_alpha_mean.value)  # [A]
    alpha_stddev = jax.nn.softplus(dual_params.log_alpha_stddev.value)  # [A]

    loss, outputs = rlax.mpo_loss(
        sample_log_probs=log_probs,
        sample_q_values=q_values,
        temperature_constraint=rlax.LagrangePenalty(
            alpha=temperature, epsilon=jnp.asarray(epsilon, jnp.float32)
        ),
        kl_constraints=[
            (
                kl_mean,
                rlax.LagrangePenalty(
                    alpha=alpha_mean,
                    epsilon=jnp.asarray(epsilon_mean, jnp.float32),
                    per_dimension=True,
                ),
            ),
            (
                kl_stddev,
                rlax.LagrangePenalty(
                    alpha=alpha_stddev,
                    epsilon=jnp.asarray(epsilon_stddev, jnp.float32),
                    per_dimension=True,
                ),
            ),
        ],
        sample_axis=0,
    )

    aux = {
        "policy_loss": jnp.mean(outputs.policy_loss),
        "temperature_loss": jnp.mean(outputs.temperature_loss),
        "kl_loss": jnp.mean(outputs.kl_loss),
        "temperature": temperature,
        "kl_mean": jnp.mean(jnp.sum(kl_mean, axis=-1)),
        "kl_stddev": jnp.mean(jnp.sum(kl_stddev, axis=-1)),
    }
    return jnp.mean(loss), aux


@nnx.jit
def sac_alpha_loss_fn(log_alpha_module, log_probs, target_entropy):
    alpha = jnp.exp(log_alpha_module.log_alpha.value)
    return -jnp.mean(alpha * (jax.lax.stop_gradient(log_probs) + target_entropy))
