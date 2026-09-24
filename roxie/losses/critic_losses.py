"""Critic loss functions for reinforcement learning algorithms."""

import jax
import jax.numpy as jnp
import rlax

from roxie.utils.math import scale_to_env


def _smoothed_target_actions(
    target_actor_model,
    next_obs,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
):
    """Computes target actions with target policy smoothing noise in environment scale.

    Args:
        target_actor_model: Target actor network.
        next_obs: Next observation batch.
        noise_key: PRNG key for Gaussian noise generation.
        target_policy_noise: Standard deviation scale of smoothing noise.
        target_noise_clip: Absolute clipping threshold for smoothing noise.
        action_low: Lower bound array of the environment action space.
        action_high: Upper bound array of the environment action space.

    Returns:
        A tuple of `(scaled_next_actions, smooth_clip_frac)`.
    """
    next_actions = target_actor_model(next_obs)

    noise = (
        jax.random.normal(noise_key, next_actions.shape, dtype=next_actions.dtype)
        * target_policy_noise
    )
    clipped_noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    smooth_clip_frac = jnp.mean(
        (jnp.abs(noise) > target_noise_clip).astype(jnp.float32)
    )

    next_actions = jnp.clip(next_actions + clipped_noise, -1.0, 1.0)
    return scale_to_env(next_actions, action_low, action_high), smooth_clip_frac


def categorical_mean(logits, atoms) -> jnp.ndarray:
    """Computes the expected value of a categorical distribution over fixed support atoms."""
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * atoms, axis=-1)


def value_aux(q_buffer, target_q, td_error, **extra) -> dict:
    """Computes standard value-health diagnostic metrics across critic losses."""
    return {
        "q_buffer": jnp.mean(q_buffer),
        "q_target": jnp.mean(target_q),
        "td_abs": jnp.mean(jnp.abs(td_error)),
        **extra,
    }


def action_rail_frac(actions, action_low, action_high) -> jnp.ndarray:
    """Calculates the fraction of actions sitting at the environment boundary limits."""
    tol = 1e-3 * (action_high - action_low)
    at_rail = (actions <= action_low + tol) | (actions >= action_high - tol)
    return jnp.mean(at_rail.astype(jnp.float32))


def ddpg_critic_loss_fn(
    critic_model,
    target_actor_model,
    target_critic_model,
    samples,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
):
    """Computes the mean squared error critic loss for DDPG."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )
    next_q = jnp.squeeze(target_critic_model(next_obs, next_actions))
    current_q = jnp.squeeze(critic_model(obs, samples["actions"]))

    reward = jnp.squeeze(samples["rewards"])
    td_error = jax.vmap(rlax.td_learning)(
        current_q, reward, samples["bootstrap"], next_q
    )

    critic_loss = jnp.mean(jnp.square(td_error))

    aux = value_aux(
        current_q,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q),
        td_error,
        target_smooth_clip_frac=smooth_clip_frac,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return critic_loss, aux


def soft_clipped_double_q(q1, q2, weight):
    """Blends a twin bootstrap between `min` (weight 1.0) and `max` (weight 0.0)."""
    return weight * jnp.minimum(q1, q2) + (1.0 - weight) * jnp.maximum(q1, q2)


def td3_critic_loss_fn(
    twin_critic,
    target_actor_model,
    target_twin_critic,
    samples,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
    twin_q_weight,
):
    """Computes the twin critic MSE loss with clipped double-Q target evaluation for TD3."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_q1, target_q2 = target_twin_critic(next_obs, next_actions)
    next_q = soft_clipped_double_q(
        jnp.squeeze(target_q1), jnp.squeeze(target_q2), twin_q_weight
    )

    q1, q2 = twin_critic(obs, samples["actions"])
    q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)

    reward = jnp.squeeze(samples["rewards"])
    td_error1 = jax.vmap(rlax.td_learning)(q1, reward, samples["bootstrap"], next_q)
    td_error2 = jax.vmap(rlax.td_learning)(q2, reward, samples["bootstrap"], next_q)
    critic_loss = jnp.mean(jnp.square(td_error1)) + jnp.mean(jnp.square(td_error2))

    aux = value_aux(
        q1,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q),
        td_error1,
        twin_gap=jnp.mean(jnp.abs(q1 - q2)),
        target_smooth_clip_frac=smooth_clip_frac,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return critic_loss, aux


def d4pg_critic_loss_fn(
    critic_model,
    target_actor_model,
    target_critic_model,
    samples,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
):
    """Computes the categorical distributional temporal-difference loss for D4PG."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_logits = target_critic_model(next_obs, next_actions)
    logits = critic_model(obs, samples["actions"])
    reward = jnp.squeeze(samples["rewards"])

    atoms = critic_model.atoms
    losses = jax.vmap(rlax.categorical_td_learning, in_axes=(None, 0, 0, 0, None, 0))(
        atoms, logits, reward, samples["bootstrap"], atoms, target_logits
    )

    q_buffer = categorical_mean(logits, atoms)
    target_q = jax.lax.stop_gradient(
        reward + samples["bootstrap"] * categorical_mean(target_logits, atoms)
    )
    aux = value_aux(
        q_buffer,
        target_q,
        q_buffer - target_q,
        target_smooth_clip_frac=smooth_clip_frac,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return jnp.mean(losses), aux


def wasserstein_blend_logits(logits_lo, logits_hi, weight, atoms):
    """Blends two categorical value distributions along their quantile functions.

    Mixing the densities instead would add a second mode and inflate the spread,
    which compounds over Bellman backups until the support smears out.

    Args:
        logits_lo: Logits of the distribution taken at `weight` 1.0.
        logits_hi: Logits of the other head.
        weight: 1.0 returns `logits_lo`, 0.0 returns `logits_hi`.
        atoms: The shared, fixed support both are defined on.

    Returns:
        Logits of the blend, projected back onto `atoms`.
    """
    n = atoms.shape[0]
    levels = (jnp.arange(n, dtype=atoms.dtype) + 0.5) / n

    def quantiles(logits):
        cdf = jnp.cumsum(jax.nn.softmax(logits, axis=-1), axis=-1)
        idx = jnp.sum(
            (cdf[..., None, :] < levels[:, None]).astype(jnp.int32), axis=-1
        )
        return atoms[jnp.clip(idx, 0, n - 1)]

    q = weight * quantiles(logits_lo) + (1.0 - weight) * quantiles(logits_hi)

    dz = (atoms[-1] - atoms[0]) / (n - 1)
    b = jnp.clip((q - atoms[0]) / dz, 0.0, n - 1.0)
    lower = jnp.floor(b)
    frac = b - lower
    lower_i = lower.astype(jnp.int32)
    upper_i = jnp.minimum(lower_i + 1, n - 1)
    probs = (
        jnp.sum(
            jax.nn.one_hot(lower_i, n, dtype=atoms.dtype) * (1.0 - frac)[..., None],
            axis=-2,
        )
        + jnp.sum(
            jax.nn.one_hot(upper_i, n, dtype=atoms.dtype) * frac[..., None], axis=-2
        )
    ) / n
    return jnp.log(probs + 1e-12)


def td4_critic_loss_fn(
    twin_critic,
    target_actor_model,
    target_twin_critic,
    samples,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
    twin_q_weight,
):
    """Computes the categorical distributional loss for twin critics with double-Q targets."""
    atoms = twin_critic.critic1.atoms
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_logits1, target_logits2 = target_twin_critic(next_obs, next_actions)
    target_q1 = categorical_mean(target_logits1, atoms)
    target_q2 = categorical_mean(target_logits2, atoms)
    take_first = (target_q1 <= target_q2)[:, None]
    lower = jnp.where(take_first, target_logits1, target_logits2)
    if twin_q_weight >= 1.0:
        target_logits = lower
    else:
        upper = jnp.where(take_first, target_logits2, target_logits1)
        target_logits = wasserstein_blend_logits(
            lower, upper, twin_q_weight, atoms
        )

    logits1, logits2 = twin_critic(obs, samples["actions"])
    reward = jnp.squeeze(samples["rewards"])

    categorical_td = jax.vmap(
        rlax.categorical_td_learning, in_axes=(None, 0, 0, 0, None, 0)
    )
    critic_loss = jnp.mean(
        categorical_td(atoms, logits1, reward, samples["bootstrap"], atoms, target_logits)
    ) + jnp.mean(
        categorical_td(atoms, logits2, reward, samples["bootstrap"], atoms, target_logits)
    )

    q1 = categorical_mean(logits1, atoms)
    target_q = jax.lax.stop_gradient(
        reward + samples["bootstrap"] * soft_clipped_double_q(
            target_q1, target_q2, twin_q_weight
        )
    )
    aux = value_aux(
        q1,
        target_q,
        q1 - target_q,
        twin_gap=jnp.mean(jnp.abs(q1 - categorical_mean(logits2, atoms))),
        target_smooth_clip_frac=smooth_clip_frac,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return critic_loss, aux


def ppo_critic_loss_fn(
    critic_model,
    observations,
    returns,
):
    """Computes the state-value MSE critic loss for PPO."""
    v_t = critic_model(observations)[..., 0]
    return 0.5 * jnp.mean((v_t - returns) ** 2)


def mpo_critic_loss_fn(
    critic_model,
    target_actor_model,
    target_critic_model,
    samples,
    key,
    num_action_samples,
    action_low,
    action_high,
):
    """Computes the MPO state-action value loss averaged over action samples."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_dist = target_actor_model(next_obs)
    next_actions = next_dist.sample(seed=key, sample_shape=(num_action_samples,))
    next_actions = scale_to_env(next_actions, action_low, action_high)

    next_obs_tiled = jnp.broadcast_to(next_obs, (num_action_samples,) + next_obs.shape)
    next_q = target_critic_model(next_obs_tiled, next_actions)
    next_q = jnp.mean(jnp.squeeze(next_q, axis=-1), axis=0)

    reward = jnp.squeeze(samples["rewards"])
    current_q = jnp.squeeze(critic_model(obs, samples["actions"]))
    # `rlax.td_learning` is defined per scalar transition.
    td_error = jax.vmap(rlax.td_learning)(
        current_q, reward, samples["bootstrap"], next_q
    )

    critic_loss = jnp.mean(jnp.square(td_error))
    aux = value_aux(
        current_q,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q),
        td_error,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return critic_loss, aux


def sac_critic_loss_fn(
    twin_critic,
    actor_model,
    target_twin_critic,
    samples,
    alpha,
    key,
    action_low,
    action_high,
    twin_q_weight,
):
    """Computes the Soft Actor-Critic twin value loss with entropy-augmented targets."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_dist = actor_model(next_obs)
    next_actions, next_pre = next_dist.sample_from_pre(seed=key)
    next_log_probs = next_dist.log_prob_from_pre(next_pre)

    next_actions_scaled = scale_to_env(next_actions, action_low, action_high)

    target_q1, target_q2 = target_twin_critic(next_obs, next_actions_scaled)
    target_q = soft_clipped_double_q(
        jnp.squeeze(target_q1), jnp.squeeze(target_q2), twin_q_weight
    )
    target_q = target_q - alpha * next_log_probs

    q1, q2 = twin_critic(obs, samples["actions"])
    q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)

    reward = jnp.squeeze(samples["rewards"])
    td_error1 = jax.vmap(rlax.td_learning)(q1, reward, samples["bootstrap"], target_q)
    td_error2 = jax.vmap(rlax.td_learning)(q2, reward, samples["bootstrap"], target_q)
    critic_loss = jnp.mean(rlax.l2_loss(td_error1)) + jnp.mean(rlax.l2_loss(td_error2))

    aux = value_aux(
        q1,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * target_q),
        td_error1,
        twin_gap=jnp.mean(jnp.abs(q1 - q2)),
        target_act_rail_frac=action_rail_frac(
            next_actions_scaled, action_low, action_high
        ),
    )
    return critic_loss, aux