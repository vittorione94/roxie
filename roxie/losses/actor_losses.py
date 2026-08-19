import jax
import jax.numpy as jnp
import rlax
from flax import nnx

from roxie.agents.agent import Agent
from roxie.models.actors import distribution_entropy

# Pre-tanh magnitude past which the saturation penalty starts charging.
# tanh(1) = 0.76, so the policy keeps the whole useful range of the action
# space for free and is only pushed back once it heads for the rails, where
# d(tanh u)/du collapses and the DPG gradient dies. A two-sided u^2 penalty
# would instead bias every action toward zero, which fights tasks (like mocap
# position servos) that legitimately need targets near the joint limits.
PRE_ACTIVATION_THRESHOLD = 1.0


def pre_activation_penalty(pre_activation: jnp.ndarray) -> jnp.ndarray:
    """One-sided hinge on the actor's pre-tanh logits: sum_j relu(|u_j| - 1)^2,
    averaged over the batch.

    Counterweight to the deterministic policy gradient, which pushes the logits
    outward without bound (see `DeterministicActor`'s saturation note). Its
    gradient grows linearly in the overshoot, so it still bites at |u| ~ 10
    where the tanh derivative has already underflowed to zero and -dQ/du can no
    longer pull the policy back on its own.

    REDUCTION: sum over the action dimension, mean over the batch — deliberately
    *not* a plain `mean` over both. The DPG term it counterbalances is
    `-mean_batch(Q)`, and a single Q depends on the whole action vector, so its
    gradient w.r.t. one logit carries no 1/action_dim factor. Averaging the
    penalty over action dims too would silently divide `pre_activation_coef` by
    action_dim (56 for the CMU humanoid — two orders of magnitude), making the
    hinge inert exactly where it is needed and making a tuned coefficient
    meaningless across embodiments. Summing keeps the two terms commensurate:
    per logit, the penalty gradient is `2 * coef * excess` against the DPG
    term's `dQ/da_j * (1 - tanh^2 u_j)`.
    """
    excess = jax.nn.relu(jnp.abs(pre_activation) - PRE_ACTIVATION_THRESHOLD)
    return jnp.mean(jnp.sum(jnp.square(excess), axis=-1))


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
def d4pg_actor_loss_fn(
    actor_model,
    critic_model,
    samples,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
    atoms,
):
    """DPG through the distributional critic: maximize the categorical's mean."""
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    logits = critic_model(obs, actions)  # (B, num_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits, axis=-1) * atoms, axis=-1)
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
    pre_activation_coef,
):
    """Deterministic policy gradient through the first critic head (TD3), plus
    a one-sided penalty on the actor's pre-tanh logits.

    Without the penalty term the DPG objective drives the logits out until tanh
    saturates and the actor gradient underflows to zero, freezing the policy as
    a bang-bang controller (see `DeterministicActor`). `pre_activation_coef`
    trades Q against that; 0.0 recovers the textbook TD3 actor loss.

    Returns ``(loss, aux)``; `aux` carries the saturation diagnostics the agent
    logs under `td3/` (see `TD3.pop_diagnostics`). They are read off this very
    forward pass rather than a second one, so the instrumentation is free.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions, pre_activation = actor_model.forward(obs)  # [-1, 1], pre-tanh
    scaled_actions = Agent.scale_to_env(actions, action_low, action_high)
    q1, _ = twin_critic(obs, scaled_actions)
    actor_q = jnp.mean(q1)
    penalty = pre_activation_penalty(pre_activation)

    abs_u = jnp.abs(pre_activation)
    # d(tanh u)/du: the factor the DPG gradient is multiplied by before it ever
    # reaches the weights. As it goes to zero the actor stops being trainable,
    # so this is the leading indicator of the saturation collapse — it moves
    # well before the eval score does.
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


@nnx.jit
def td4_actor_loss_fn(
    actor_model,
    twin_critic,
    samples,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
    atoms,
):
    """DPG through the first distributional head's expected value (TD4).

    The TD3 convention: the actor follows critic 1 only, so the pessimistic
    min stays confined to the critic's bootstrap target.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions = actor_model(obs)  # [-1, 1]
    actions = Agent.scale_to_env(actions, action_low, action_high)  # [low, high]
    logits1, _ = twin_critic(obs, actions)  # (B, num_atoms)
    q_values = jnp.sum(jax.nn.softmax(logits1, axis=-1) * atoms, axis=-1)
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

    # Entropy of the distribution. Analytic for a plain Normal; for a squashed
    # policy it is a single-sample estimate and genuinely needs `key` (which was
    # previously accepted and unused here).
    entropy_t = distribution_entropy(distribution, key)[:, :-1]

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

    # Trust-region diagnostics, returned as aux so they cost no extra forward
    # pass. `approx_kl` is Schulman's low-variance estimator of
    # KL(pi_old || pi_new); it is >= 0 and equals 0 iff the ratio is 1 everywhere.
    # `clip_frac` is the share of the batch that has left the clip range -- those
    # samples select the constant branch of the min() above, whose gradient
    # w.r.t. the policy is ZERO, so they stop pulling the policy back. A clip_frac
    # near 1 means the surrogate is saturated and the update is being driven by
    # whatever is left (in practice the entropy term).
    approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
    clip_frac = jnp.mean((jnp.abs(ratio - 1.0) > clip_epsilon).astype(jnp.float32))

    # Per-step loss minus entropy term, then average across all dims
    per_step_loss = pg_loss - entropy_coef * entropy_t
    return jnp.mean(per_step_loss), (approx_kl, clip_frac)


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
