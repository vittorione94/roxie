"""Actor losses.

Same contract as `critic_losses`: the observations in `samples` arrive already
normalized, so no loss here normalizes on its own, and nothing here is jitted —
the agents' `_grad_steps` bursts are the compilation unit. See that module's
docstring.
"""

import jax
import jax.numpy as jnp
import rlax

from roxie.agents.agent import Agent
from roxie.models.actors import distribution_entropy

# Pre-tanh magnitude past which the saturation penalty starts charging.
# tanh(1) = 0.76, so the policy keeps the useful range of the action space for
# free and is pushed back only once it heads for the rails. A two-sided u^2
# penalty would instead bias every action toward zero.
PRE_ACTIVATION_THRESHOLD = 1.0

# rlax's MPO ops default `projection_operator` to a `jnp.clip(a_min=...)`
# partial, but JAX dropped the `a_min`/`a_max` aliases, so `rlax.mpo_loss`
# raises a TypeError as soon as it projects the duals. This is rlax's own
# `_EPSILON`, so the numerics are unchanged.
_DUAL_EPSILON = 1e-10


def _project_dual(alpha: jnp.ndarray) -> jnp.ndarray:
    """Project a Lagrange multiplier into the strictly positive range."""
    return jnp.clip(alpha, _DUAL_EPSILON)


def pre_activation_penalty(pre_activation: jnp.ndarray) -> jnp.ndarray:
    """One-sided hinge on the actor's pre-tanh logits: sum_j relu(|u_j| - 1)^2,
    averaged over the batch.

    Counterweight to the deterministic policy gradient, which pushes the logits
    outward without bound. Its gradient grows linearly in the overshoot, so it
    still bites at |u| ~ 10 where the tanh derivative has underflowed to zero and
    -dQ/du can no longer pull the policy back on its own.

    Summed over the action dimension, meaned over the batch — deliberately not a
    plain `mean` over both. The DPG term it counterbalances is `-mean_batch(Q)`,
    and a single Q depends on the whole action vector, so its gradient w.r.t. one
    logit carries no 1/action_dim factor. Averaging here too would divide
    `pre_activation_coef` by action_dim, making a tuned coefficient meaningless
    across embodiments.
    """
    excess = jax.nn.relu(jnp.abs(pre_activation) - PRE_ACTIVATION_THRESHOLD)
    return jnp.mean(jnp.sum(jnp.square(excess), axis=-1))


def ddpg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    action_low,
    action_high,
):
    """Calculates the loss for the actor (aims to maximize Q-value)."""
    obs = samples["observations"]
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    q_values = critic_model(obs, actions)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


def d4pg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    action_low,
    action_high,
    atoms,
):
    """DPG through the distributional critic: maximize the categorical's mean."""
    obs = samples["observations"]
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    logits = critic_model(obs, actions)  # (B, num_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * atoms, axis=-1)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


def td3_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    action_low,
    action_high,
    pre_activation_coef,
):
    """Deterministic policy gradient through the first critic head (TD3), plus
    a one-sided penalty on the actor's pre-tanh logits.

    Without the penalty the DPG objective drives the logits out until tanh
    saturates and the actor gradient underflows to zero, freezing the policy as a
    bang-bang controller. `pre_activation_coef` of 0.0 recovers the textbook TD3
    actor loss.

    Returns ``(loss, aux)``; `aux` carries the saturation diagnostics the agent
    logs under `td3/`, read off this forward pass rather than a second one.
    """
    obs = samples["observations"]
    actions, pre_activation = actor_model.forward(obs)  # [-1, 1], pre-tanh
    scaled_actions = Agent.scale_to_env(actions, action_low, action_high)
    q1, _ = twin_critic(obs, scaled_actions)
    actor_q = jnp.mean(q1)
    penalty = pre_activation_penalty(pre_activation)

    abs_u = jnp.abs(pre_activation)
    # d(tanh u)/du: the factor the DPG gradient is multiplied by before it
    # reaches the weights, and so the leading indicator of saturation collapse.
    tanh_grad = 1.0 - jnp.square(actions)
    aux = {
        "pre_act_abs": jnp.mean(abs_u),
        "pre_act_max": jnp.max(abs_u),
        "pre_act_penalty": penalty,
        "sat_frac": jnp.mean((jnp.abs(actions) > 0.99).astype(jnp.float32)),
        "tanh_grad": jnp.mean(tanh_grad),
        "actor_q": actor_q,
    }
    return -actor_q + pre_activation_coef * penalty, aux


def td4_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    action_low,
    action_high,
    atoms,
):
    """DPG through the first distributional head's expected value (TD4).

    The TD3 convention: the actor follows critic 1 only, so the pessimistic
    min stays confined to the critic's bootstrap target.
    """
    obs = samples["observations"]
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    logits1, _ = twin_critic(obs, actions)  # (B, num_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits1, axis=-1) * atoms, axis=-1)
    actor_loss = -jnp.mean(q_values)
    return actor_loss


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
    """Calculates the loss for the actor using PPO clipped objective.

    `observations` are already normalized, under the frozen behaviour-policy
    statistics — that freeze is what keeps the ratio exactly 1 on the first pass.
    """

    distribution = actor_model(observations)
    logp_new = distribution.log_prob(actions_buf)       # (N, T)

    # Analytic for a plain Normal; for a squashed policy it is a single-sample
    # estimate, which is what `key` is for.
    entropy_t = distribution_entropy(distribution, key)[:, :-1]

    # The last entry is the next-frame bootstrap, dropped so the ratio lines up
    # with the advantages: (NUM_ENVS, BATCH_SIZE - 1).
    ratio = jnp.exp(logp_new[:, :-1] - old_log_probs[:, :-1])

    if ratio.shape != advantages.shape:
        raise ValueError(f"ratio and advantages shapes must match; got {ratio.shape} vs {advantages.shape}")

    if ratio.ndim < 1:
        raise ValueError("ratio must have at least 1 dimension (time axis)")

    # Written out elementwise so the result keeps the inputs' shape (per-timestep
    # losses) rather than being reduced to a scalar.
    clipped_ratio = jnp.clip(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon)
    surrogate1 = ratio * advantages
    surrogate2 = clipped_ratio * advantages
    surrogate = jnp.minimum(surrogate1, surrogate2)
    pg_loss = -surrogate

    # `approx_kl` is Schulman's low-variance estimator of KL(pi_old || pi_new):
    # >= 0, and 0 iff the ratio is 1 everywhere. Samples outside the clip range
    # take the constant branch of the min() above, whose policy gradient is zero
    # — near clip_frac 1 the entropy term alone drives the update.
    approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32))

    per_step_loss = pg_loss - entropy_coef * entropy_t
    return jnp.mean(per_step_loss), (approx_kl, clip_frac)


def sac_actor_loss_fn(
    actor_model,
    twin_critic,
    alpha,
    samples,
    key,
    action_low,
    action_high,
):
    obs = samples["observations"]
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
):
    """MPO policy improvement loss (E-step + M-step).

    E-step: build a nonparametric improved policy by reweighting actions sampled
    from the target (old) policy with ``softmax(Q / temperature)``, where the
    temperature is found by minimizing the convex dual of a hard KL bound
    (``epsilon``).

    M-step: fit the parametric policy to those weighted samples by weighted
    maximum likelihood, subject to a *decoupled* KL trust region — mean and
    covariance are constrained separately (``epsilon_mean``, ``epsilon_stddev``),
    each with its own per-dimension Lagrange multiplier.

    The dual variables live in ``dual_params`` and are optimized jointly with the
    actor by this same loss.
    """
    obs = samples["observations"]

    # N actions per state from the target (old) policy: [S, B, A].
    target_dist = target_actor_model(obs)
    actions = target_dist.sample(seed=key, sample_shape=(num_action_samples,))

    # No gradient flows into the policy through Q; it only shapes the E-step
    # weights and the temperature.
    actions_scaled = Agent.scale_to_env(
        jnp.clip(actions, -1.0, 1.0), action_low, action_high
    )
    obs_tiled = jnp.broadcast_to(obs, (num_action_samples,) + obs.shape)
    q_values = critic_model(obs_tiled, actions_scaled)  # [S, B, 1]
    q_values = jax.lax.stop_gradient(jnp.squeeze(q_values, axis=-1))  # [S, B]

    # M-step likelihood of the sampled actions under the *online* policy.
    online_dist = actor_model(obs)
    log_probs = online_dist.log_prob(actions)  # [S, B]

    # `kl_mean` varies the mean while holding the target std fixed; `kl_stddev`
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
        projection_operator=_project_dual,
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


def sac_alpha_loss_fn(log_alpha_module, log_probs, target_entropy):
    alpha = jnp.exp(log_alpha_module.log_alpha.value)
    return -jnp.mean(alpha * (jax.lax.stop_gradient(log_probs) + target_entropy))
