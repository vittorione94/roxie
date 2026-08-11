import jax
import jax.numpy as jnp
import rlax
from flax import nnx

from roxie.agents.agent import Agent


@nnx.jit
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
    obs_mean,
    obs_std,
    obs_clip,
):
    """Calculates the MSE loss for the critic."""
    # Normalize observations
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)  # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target smoothing noise in env units
    act_span = action_high - action_low
    noise = jax.random.normal(noise_key, next_actions.shape) * (
        target_policy_noise * act_span
    )
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)

    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    next_q = target_critic_model(next_obs, next_actions)

    # `rewards` is the (n-step) return and `bootstrap` the per-sample
    # coefficient gamma^b * (0 if terminal in window) — both precomputed by
    # repack_samples, which owns all gamma/terminal/truncation handling.
    reward = jnp.squeeze(samples["rewards"])
    next_q = jnp.squeeze(next_q)
    target_q = reward + samples["bootstrap"] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    current_q = critic_model(obs, samples["actions"])

    critic_loss = jnp.mean((jnp.squeeze(current_q) - target_q) ** 2)
    return critic_loss


@nnx.jit
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
    obs_mean,
    obs_std,
    obs_clip,
):
    """MSE loss for the twin critic using clipped double-Q targets (TD3).

    Identical to DDPG's target construction but takes the elementwise minimum
    of the two target critics to curb the overestimation bias that makes
    single-critic DDPG diverge.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)  # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target policy smoothing noise in env units
    act_span = action_high - action_low
    noise = jax.random.normal(noise_key, next_actions.shape) * (
        target_policy_noise * act_span
    )
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    # Clipped double-Q: take the minimum of the two target critics
    target_q1, target_q2 = target_twin_critic(next_obs, next_actions)
    next_q = jnp.minimum(jnp.squeeze(target_q1), jnp.squeeze(target_q2))

    # `rewards` is the (n-step) return and `bootstrap` the per-sample
    # coefficient gamma^b * (0 if terminal in window) — both precomputed by
    # repack_samples, which owns all gamma/terminal/truncation handling.
    reward = jnp.squeeze(samples["rewards"])
    target_q = reward + samples["bootstrap"] * next_q
    target_q = jax.lax.stop_gradient(target_q)

    q1, q2 = twin_critic(obs, samples["actions"])
    critic_loss = jnp.mean((jnp.squeeze(q1) - target_q) ** 2) + jnp.mean(
        (jnp.squeeze(q2) - target_q) ** 2
    )
    return critic_loss


@nnx.jit
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
    obs_mean,
    obs_std,
    obs_clip,
    atoms,
):
    """Categorical distributional critic loss for D4PG (Barth-Maron et al. 2018).

    The critic outputs logits over a fixed support `atoms`. The target
    distribution is the target critic's categorical at (s', pi'(s')) with its
    support shifted per sample to `rewards + bootstrap * atoms` (rewards is the
    n-step return and bootstrap the precomputed gamma^b coefficient, 0 at
    terminals — so a terminal collapses the target to a delta at the return),
    then L2-projected back onto `atoms`. Loss is the cross-entropy between the
    projected target and the online critic's categorical.

    Target policy smoothing is kept for signature parity with the DDPG/TD3
    losses; the paper doesn't smooth, so configs set the noise to 0.
    """
    # Normalize observations
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)  # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target smoothing noise in env units (no-op at the paper's noise = 0)
    act_span = action_high - action_low
    noise = jax.random.normal(noise_key, next_actions.shape) * (
        target_policy_noise * act_span
    )
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    # Target categorical over the shifted support, projected onto `atoms`
    target_logits = target_critic_model(next_obs, next_actions)  # (B, num_atoms)
    target_probs = jax.nn.softmax(target_logits, axis=-1)

    reward = jnp.squeeze(samples["rewards"])
    target_z = reward[:, None] + samples["bootstrap"][:, None] * atoms[None, :]

    projected = jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))(
        target_z, target_probs, atoms
    )
    projected = jax.lax.stop_gradient(projected)

    # Cross-entropy between projected target and online categorical
    logits = critic_model(obs, samples["actions"])
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    critic_loss = -jnp.mean(jnp.sum(projected * log_probs, axis=-1))
    return critic_loss


@nnx.jit
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
    obs_mean,
    obs_std,
    obs_clip,
    atoms,
):
    """Categorical distributional loss for a twin critic with clipped double-Q.

    Same construction as `d4pg_critic_loss_fn`, but the bootstrap distribution
    comes from *one* of the two target critics — the pessimistic one. Taking an
    elementwise minimum over the two atom probability vectors would not yield a
    distribution (it doesn't sum to 1), so the selection is per sample: whichever
    target head has the lower expected value contributes its *whole* categorical.
    That keeps the target a valid distribution while preserving TD3's
    underestimation bias.

    Both online heads are then trained by cross-entropy against the same
    projected target, and the two losses are summed (as in `td3_critic_loss_fn`).
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Target actions in env scale
    next_actions = target_actor_model(next_obs)  # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target policy smoothing noise in env units
    act_span = action_high - action_low
    noise = jax.random.normal(noise_key, next_actions.shape) * (
        target_policy_noise * act_span
    )
    noise_clip = target_noise_clip * act_span
    noise = jnp.clip(noise, -noise_clip, noise_clip)
    next_actions = jnp.clip(next_actions + noise, action_low, action_high)

    # Both target categoricals at (s', pi'(s')), then pick the pessimistic head
    # per sample by comparing their expected values.
    target_logits1, target_logits2 = target_twin_critic(next_obs, next_actions)
    target_probs1 = jax.nn.softmax(target_logits1, axis=-1)  # (B, num_atoms)
    target_probs2 = jax.nn.softmax(target_logits2, axis=-1)

    target_q1 = jnp.sum(target_probs1 * atoms[None, :], axis=-1)  # (B,)
    target_q2 = jnp.sum(target_probs2 * atoms[None, :], axis=-1)
    take_first = (target_q1 <= target_q2)[:, None]
    target_probs = jnp.where(take_first, target_probs1, target_probs2)

    # Shift the support by the n-step return / bootstrap coefficient (0 at
    # terminals → the target collapses to a delta at the return) and L2-project
    # it back onto the fixed `atoms`.
    reward = jnp.squeeze(samples["rewards"])
    target_z = reward[:, None] + samples["bootstrap"][:, None] * atoms[None, :]

    projected = jax.vmap(rlax.categorical_l2_project, in_axes=(0, 0, None))(
        target_z, target_probs, atoms
    )
    projected = jax.lax.stop_gradient(projected)

    # Cross-entropy of both online heads against the shared projected target
    logits1, logits2 = twin_critic(obs, samples["actions"])
    log_probs1 = jax.nn.log_softmax(logits1, axis=-1)
    log_probs2 = jax.nn.log_softmax(logits2, axis=-1)
    critic_loss = -jnp.mean(jnp.sum(projected * log_probs1, axis=-1)) - jnp.mean(
        jnp.sum(projected * log_probs2, axis=-1)
    )
    return critic_loss


@nnx.jit
def ppo_critic_loss_fn(
    critic_model, 
    observations,
    values, 
    advantages, 
):
    """Calculates the MSE loss for the critic using PPO."""
    v_t = critic_model(observations)[:, :-1, 0] # shape (NUM_ENVS, BATCH_SIZE, 1)
    
    # values --> (NUM_ENVS, BATCH_SIZE)
    # advantages --> (NUM_ENVS, BATCH_SIZE -1 )

    target_values = values[:, :-1] + advantages

    critic_loss = jnp.mean((v_t - target_values) ** 2)
    return critic_loss


def mpo_critic_loss_fn(
    critic_model,
    target_actor_model,
    target_critic_model,
    samples,
    gamma,
    key,
    num_action_samples,
    action_low,
    action_high,
    obs_mean,
    obs_std,
    obs_clip,
):
    """MSE critic loss for MPO (policy evaluation).

    The bootstrap value is the expected Q of the *target* (old) policy at the
    next state, estimated by averaging the target critic over a handful of
    actions sampled from the target policy. This is the standard MPO policy
    evaluation step; we keep a single Q critic (as in the original paper) rather
    than a distributional one.

    `num_action_samples` is static at trace time (it sets the sample shape), so
    this is called from inside the already-jitted grad step rather than jitted on
    its own.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Sample N next actions from the target policy: [S, B, A] (Gaussian, no
    # squashing — bounded by clipping, consistent with action selection).
    next_dist = target_actor_model(next_obs)
    next_actions = next_dist.sample(seed=key, sample_shape=(num_action_samples,))
    next_actions = jnp.clip(next_actions, -1.0, 1.0)
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    # Broadcast next_obs over the sample axis to match [S, B, A].
    next_obs_tiled = jnp.broadcast_to(next_obs, (num_action_samples,) + next_obs.shape)
    next_q = target_critic_model(next_obs_tiled, next_actions)  # [S, B, 1]
    next_q = jnp.mean(jnp.squeeze(next_q, axis=-1), axis=0)  # [B]

    reward = jnp.squeeze(samples["rewards"])
    term = samples["terminals"].astype(jnp.float32)
    target_q = reward + gamma * (1.0 - term) * next_q
    target_q = jax.lax.stop_gradient(target_q)

    # Stored actions are already in env scale (see DDPG/SAC `add`).
    current_q = critic_model(obs, samples["actions"])
    critic_loss = jnp.mean((jnp.squeeze(current_q) - target_q) ** 2)
    return critic_loss


@nnx.jit
def sac_critic_loss_fn(
    twin_critic,
    actor_model,
    target_twin_critic,
    samples,
    gamma,
    alpha,
    key,
    action_low,
    action_high,
    obs_mean,
    obs_std,
    obs_clip,
):
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    next_obs = Agent.normalize_obs(
        samples["next_observations"], obs_mean, obs_std, obs_clip
    )

    # Sample next actions from current policy with tanh squashing
    next_dist = actor_model(next_obs)
    next_u = next_dist.sample(seed=key)
    next_actions = jnp.tanh(next_u)
    next_log_probs = next_dist.log_prob(next_u) - jnp.sum(
        jnp.log(1.0 - next_actions ** 2 + 1e-6), axis=-1
    )

    next_actions_scaled = Agent.scale_to_env(next_actions, action_low, action_high)

    # Target Q values with entropy regularization
    target_q1, target_q2 = target_twin_critic(next_obs, next_actions_scaled)
    target_q = jnp.minimum(jnp.squeeze(target_q1), jnp.squeeze(target_q2))
    target_q = target_q - alpha * next_log_probs

    # Bellman target
    reward = jnp.squeeze(samples["rewards"])
    terminal = samples["terminals"].astype(jnp.float32)
    target = reward + gamma * (1.0 - terminal) * target_q
    target = jax.lax.stop_gradient(target)

    # Current Q values from both critics
    q1, q2 = twin_critic(obs, samples["actions"])
    critic_loss = 0.5 * (
        jnp.mean((jnp.squeeze(q1) - target) ** 2)
        + jnp.mean((jnp.squeeze(q2) - target) ** 2)
    )
    return critic_loss
