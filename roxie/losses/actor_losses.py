"""Actor loss functions for policy optimization algorithms."""

import jax
import jax.numpy as jnp
import rlax

from roxie.losses.critic_losses import soft_clipped_double_q
from roxie.utils.math import scale_to_env

PRE_ACTIVATION_THRESHOLD = 1.0
_MAX_LOG_RATIO = 20.0
_DUAL_EPSILON = 1e-10


def _project_dual(alpha: jnp.ndarray) -> jnp.ndarray:
    """Projects Lagrange multipliers to stay strictly positive."""
    return jnp.clip(alpha, _DUAL_EPSILON)


def pre_activation_penalty(pre_activation: jnp.ndarray) -> jnp.ndarray:
    """Computes a one-sided quadratic penalty on pre-tanh actor logits.

    Args:
        pre_activation: Unsquashed policy pre-activation logits.

    Returns:
        Scalar penalty value averaged over the batch.
    """
    excess = jax.nn.relu(jnp.abs(pre_activation) - PRE_ACTIVATION_THRESHOLD)
    return jnp.mean(jnp.sum(jnp.square(excess), axis=-1))


ACTOR_DIAGNOSTIC_KEYS = (
    "pre_act_abs",
    "pre_act_max",
    "sat_frac",
    "tanh_grad",
    "actor_q",
)


def skipped_actor_aux(dtype, *extra_keys) -> dict:
    """Generates a zero-filled diagnostic dictionary for skipped policy update steps."""
    zero = jnp.array(0.0, dtype=dtype)
    return {key: zero for key in (*ACTOR_DIAGNOSTIC_KEYS, *extra_keys)}


def actor_aux(actions, pre_activation, actor_q, **extra) -> dict:
    """Computes policy saturation and Q-value diagnostics for actor logging."""
    abs_u = jnp.abs(pre_activation)
    return {
        "pre_act_abs": jnp.mean(abs_u),
        "pre_act_max": jnp.max(abs_u),
        "sat_frac": jnp.mean((jnp.abs(actions) > 0.99).astype(jnp.float32)),
        "tanh_grad": jnp.mean(1.0 - jnp.square(actions)),
        "actor_q": actor_q,
        **extra,
    }


def ddpg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    action_low,
    action_high,
    pre_activation_coef,
):
    """Computes the DDPG deterministic policy gradient loss with pre-activation penalty."""
    obs = samples["observations"]
    actions, pre_activation = actor_model.forward(obs)
    scaled_actions = scale_to_env(actions, action_low, action_high)
    q_values = critic_model(obs, scaled_actions)
    actor_q = jnp.mean(q_values)
    penalty = pre_activation_penalty(pre_activation)
    aux = actor_aux(actions, pre_activation, actor_q, pre_act_penalty=penalty)
    return -actor_q + pre_activation_coef * penalty, aux


def d4pg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    action_low,
    action_high,
    pre_activation_coef,
):
    """Computes the D4PG actor loss by maximizing expected return over categorical Q-atoms."""
    obs = samples["observations"]
    actions, pre_activation = actor_model.forward(obs)
    scaled_actions = scale_to_env(actions, action_low, action_high)
    logits = critic_model(obs, scaled_actions)
    q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * critic_model.atoms, axis=-1)
    actor_q = jnp.mean(q_values)
    penalty = pre_activation_penalty(pre_activation)
    aux = actor_aux(actions, pre_activation, actor_q, pre_act_penalty=penalty)
    return -actor_q + pre_activation_coef * penalty, aux


def td3_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    action_low,
    action_high,
    pre_activation_coef,
):
    """Computes the TD3 delayed actor loss against the first critic head."""
    obs = samples["observations"]
    actions, pre_activation = actor_model.forward(obs)
    scaled_actions = scale_to_env(actions, action_low, action_high)
    q1, _ = twin_critic(obs, scaled_actions)
    actor_q = jnp.mean(q1)
    penalty = pre_activation_penalty(pre_activation)
    aux = actor_aux(actions, pre_activation, actor_q, pre_act_penalty=penalty)
    return -actor_q + pre_activation_coef * penalty, aux


def td4_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    action_low,
    action_high,
    pre_activation_coef,
):
    """Computes the TD4 actor loss over the first head of a twin distributional critic."""
    obs = samples["observations"]
    actions, pre_activation = actor_model.forward(obs)
    scaled_actions = scale_to_env(actions, action_low, action_high)
    logits1, _ = twin_critic(obs, scaled_actions)
    q_values = jnp.sum(
        jax.nn.softmax(logits1, axis=-1) * twin_critic.critic1.atoms, axis=-1
    )
    actor_q = jnp.mean(q_values)
    penalty = pre_activation_penalty(pre_activation)
    aux = actor_aux(actions, pre_activation, actor_q, pre_act_penalty=penalty)
    return -actor_q + pre_activation_coef * penalty, aux


def ppo_loss_fn(
    actor_model,
    observations,
    pre_actions,
    old_log_probs,
    advantages,
    clip_epsilon,
    entropy_coef,
    key
):
    """Computes the PPO clipped surrogate actor loss and diagnostic metrics."""
    # Every argument arrives flat over `(env, time)` and already trimmed to the
    # GAE horizon — `_prepare_rollout` slices and flattens once, so that a single
    # permutation can shuffle transitions rather than whole trajectories.
    distribution = actor_model(observations)
    logp_new = distribution.log_prob_from_pre(pre_actions)

    entropy_t = distribution.entropy(seed=key)

    log_ratio = jnp.clip(
        logp_new - old_log_probs,
        -_MAX_LOG_RATIO,
        _MAX_LOG_RATIO,
    )
    ratio = jnp.exp(log_ratio)

    if ratio.shape != advantages.shape:
        raise ValueError(f"ratio and advantages shapes must match; got {ratio.shape} vs {advantages.shape}")

    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    surrogate = jnp.minimum(surrogate1, surrogate2)
    pg_loss = -surrogate

    approx_kl = jnp.mean((ratio - 1.0) - log_ratio)
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32))

    mode = jnp.tanh(distribution.loc)
    diagnostics = {
        "approx_kl": approx_kl,
        "clip_frac": clip_frac,
        "entropy": jnp.mean(entropy_t),
        "policy_std": jnp.mean(distribution.scale_diag),
        "sat_frac": jnp.mean((jnp.abs(mode) > 0.99).astype(jnp.float32)),
        "tanh_grad": jnp.mean(1.0 - jnp.square(mode)),
    }

    per_step_loss = pg_loss - entropy_coef * entropy_t
    return jnp.mean(per_step_loss), diagnostics


def sac_actor_loss_fn(
    actor_model,
    twin_critic,
    alpha,
    samples,
    key,
    action_low,
    action_high,
    twin_q_weight,
):
    """Computes the Soft Actor-Critic reparameterized policy loss."""
    obs = samples["observations"]
    distribution = actor_model(obs)
    actions, u = distribution.sample_from_pre(seed=key)
    log_probs = distribution.log_prob_from_pre(u)
    actions_scaled = scale_to_env(actions, action_low, action_high)
    q1, q2 = twin_critic(obs, actions_scaled)
    min_q = soft_clipped_double_q(
        jnp.squeeze(q1), jnp.squeeze(q2), twin_q_weight
    )
    actor_loss = jnp.mean(alpha * log_probs - min_q)
    aux = actor_aux(
        actions, u, jnp.mean(min_q),
        alpha=alpha,
        entropy=-jnp.mean(log_probs),
    )
    return actor_loss, (log_probs, aux)


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
):
    """Computes the MPO policy improvement loss (E-step and M-step)."""
    obs = samples["observations"]

    target_dist = target_actor_model(obs)
    actions, pre_actions = target_dist.sample_from_pre(
        seed=key, sample_shape=(num_action_samples,)
    )

    actions_scaled = scale_to_env(actions, action_low, action_high)
    obs_tiled = jnp.broadcast_to(obs, (num_action_samples,) + obs.shape)
    q_values = critic_model(obs_tiled, actions_scaled)
    q_values = jax.lax.stop_gradient(jnp.squeeze(q_values, axis=-1))

    online_dist = actor_model(obs)
    log_probs = online_dist.log_prob_from_pre(pre_actions)

    mu_t = jax.lax.stop_gradient(target_dist.loc)
    sig_t = jax.lax.stop_gradient(target_dist.scale_diag)
    mu_o = online_dist.loc
    sig_o = online_dist.scale_diag

    kl_mean = jnp.square(mu_t - mu_o) / (2.0 * jnp.square(sig_t))
    # Symmetrized; KL(target||online) alone grows only as log(sig_o / sig_t).
    ratio_sq = jnp.square(sig_o / sig_t)
    kl_stddev = 0.5 * (ratio_sq + 1.0 / ratio_sq) - 1.0

    temperature = jax.nn.softplus(dual_params.log_temperature[...])
    alpha_mean = jax.nn.softplus(dual_params.log_alpha_mean[...])
    alpha_stddev = jax.nn.softplus(dual_params.log_alpha_stddev[...])

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
        projection_operator=_project_dual,
        sample_axis=0,
    )

    aux = actor_aux(
        jnp.tanh(mu_o),
        mu_o,
        jnp.mean(q_values),
        policy_loss=jnp.mean(outputs.policy_loss),
        temperature_loss=jnp.mean(outputs.temperature_loss),
        kl_loss=jnp.mean(outputs.kl_loss),
        temperature=temperature,
        kl_mean=jnp.mean(jnp.sum(kl_mean, axis=-1)),
        kl_stddev=jnp.mean(jnp.sum(kl_stddev, axis=-1)),
    )
    return jnp.mean(loss), aux


def sac_alpha_loss_fn(log_alpha_module, log_probs, target_entropy):
    """Computes the loss for tuning the Soft Actor-Critic entropy temperature alpha."""
    alpha = jnp.exp(log_alpha_module.log_alpha[...])
    return -jnp.mean(alpha * (jax.lax.stop_gradient(log_probs) + target_entropy))