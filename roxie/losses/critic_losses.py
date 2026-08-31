"""Critic losses.

Uniform contract across this module and `actor_losses`: the observations in
`samples` arrive already normalized, by `Agent.normalize_samples` once per
gradient step (or, on-policy, once per rollout in `PPO._prepare_rollout`). No
loss normalizes on its own or carries `obs_mean` / `obs_std` / `obs_clip`.

Likewise `rewards` is the (n-step) return and `bootstrap` the per-sample
coefficient gamma^b * (0 if terminal in window), both precomputed by
`repack_samples` — so no loss here sees gamma, terminals or truncations, and
each works unchanged for 1-step and n-step returns.

Nothing here is jitted. Every loss is called from inside an agent's jitted
`_grad_steps` burst, so the enclosing trace compiles it; a decorator here would
only add a nested `pjit` for XLA to inline, and would force each argument to be
traced — ruling out the shape-valued statics MPO needs.
"""

import jax
import jax.numpy as jnp
import rlax

from roxie.agents.agent import Agent


def _smoothed_target_actions(
    target_actor_model,
    next_obs,
    noise_key,
    target_policy_noise,
    target_noise_clip,
    action_low,
    action_high,
):
    """Target actions at `next_obs`, in env scale, with TD3 policy smoothing.

    Shared by every off-policy critic loss that bootstraps through a
    deterministic target actor. `target_policy_noise` and `target_noise_clip`
    are both fractions of the action span. D4PG/TD4 keep this call for signature
    parity even though the papers do not smooth; their configs set the noise to
    0, making it a no-op.

    Returns ``(next_actions, smooth_clip_frac)``, the latter being the share of
    smoothing samples the clip bit: near 0 means `target_noise_clip` is inert,
    near 1 that it has flattened the Gaussian into a two-point distribution.
    """
    next_actions = target_actor_model(next_obs)  # [-1, 1]
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

    act_span = action_high - action_low
    noise = jax.random.normal(noise_key, next_actions.shape) * (
        target_policy_noise * act_span
    )
    noise_clip = target_noise_clip * act_span
    clipped_noise = jnp.clip(noise, -noise_clip, noise_clip)
    smooth_clip_frac = jnp.mean((jnp.abs(noise) > noise_clip).astype(jnp.float32))

    next_actions = jnp.clip(next_actions + clipped_noise, action_low, action_high)
    return next_actions, smooth_clip_frac


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
    """Calculates the MSE loss for the critic."""
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, _ = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )
    next_q = jnp.squeeze(target_critic_model(next_obs, next_actions))
    current_q = jnp.squeeze(critic_model(obs, samples["actions"]))

    # `rlax.td_learning` owns the stop_gradient on the target and is defined per
    # scalar transition, hence the vmap over the batch.
    reward = jnp.squeeze(samples["rewards"])
    td_error = jax.vmap(rlax.td_learning)(
        current_q, reward, samples["bootstrap"], next_q
    )

    # NOT `rlax.l2_loss`, which carries a 0.5 factor: this loss is the unhalved
    # MSE, and halving it would halve the critic's effective learning rate.
    critic_loss = jnp.mean(jnp.square(td_error))
    return critic_loss


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
):
    """MSE loss for the twin critic using clipped double-Q targets (TD3).

    Identical to DDPG's target construction but takes the elementwise minimum
    of the two target critics to curb overestimation bias.

    Returns ``(loss, aux)``; `aux` carries the value-health diagnostics the
    agent logs under `td3/`, read off this forward pass so the instrumentation
    costs nothing extra.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_q1, target_q2 = target_twin_critic(next_obs, next_actions)
    next_q = jnp.minimum(jnp.squeeze(target_q1), jnp.squeeze(target_q2))

    q1, q2 = twin_critic(obs, samples["actions"])
    q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)

    # See `ddpg_critic_loss_fn` on why this is `rlax.td_learning` but not
    # `rlax.l2_loss`.
    reward = jnp.squeeze(samples["rewards"])
    td_error1 = jax.vmap(rlax.td_learning)(q1, reward, samples["bootstrap"], next_q)
    td_error2 = jax.vmap(rlax.td_learning)(q2, reward, samples["bootstrap"], next_q)
    critic_loss = jnp.mean(jnp.square(td_error1)) + jnp.mean(jnp.square(td_error2))

    # Restates the target `td_learning` builds internally; XLA folds it into the
    # same subexpression, so it costs nothing.
    target_q = jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q)

    # A high rail fraction means the target actor has saturated too, so the
    # bootstrap is evaluated at action-space corners where the critic has the
    # least data — the classic overestimation setup.
    rail_tol = 1e-3 * (action_high - action_low)
    at_rail = (next_actions <= action_low + rail_tol) | (
        next_actions >= action_high - rail_tol
    )
    aux = {
        "q_buffer": jnp.mean(q1),
        "q_target": jnp.mean(target_q),
        "td_abs": jnp.mean(jnp.abs(td_error1)),
        # Should stay small relative to |Q|; a widening gap means the two heads
        # are extrapolating differently and the min is doing heavy lifting.
        "twin_gap": jnp.mean(jnp.abs(q1 - q2)),
        "target_smooth_clip_frac": smooth_clip_frac,
        "target_act_rail_frac": jnp.mean(at_rail.astype(jnp.float32)),
    }
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
    atoms,
):
    """Categorical distributional critic loss for D4PG (Barth-Maron et al. 2018).

    The critic outputs logits over a fixed support `atoms`. The target is the
    target critic's categorical at (s', pi'(s')) with its support shifted to
    `rewards + bootstrap * atoms`, projected back onto `atoms`, then
    cross-entropy against the online critic. That whole chain is
    `rlax.categorical_td_learning`, which is defined for one transition, hence
    the vmap over the batch with `atoms` broadcast.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, _ = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_logits = target_critic_model(next_obs, next_actions)  # (B, num_atoms)
    logits = critic_model(obs, samples["actions"])               # (B, num_atoms)
    reward = jnp.squeeze(samples["rewards"])

    losses = jax.vmap(rlax.categorical_td_learning, in_axes=(None, 0, 0, 0, None, 0))(
        atoms, logits, reward, samples["bootstrap"], atoms, target_logits
    )
    return jnp.mean(losses)


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
    atoms,
):
    """Categorical distributional loss for a twin critic with clipped double-Q.

    Same construction as `d4pg_critic_loss_fn`, but the bootstrap distribution
    comes from *one* of the two target critics. An elementwise minimum over the
    two atom probability vectors would not sum to 1, so the selection is per
    sample: whichever head has the lower expected value contributes its whole
    categorical, keeping the target a valid distribution while preserving TD3's
    underestimation bias. Both online heads are then trained against it.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, _ = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    # Selected on the logits rather than the probabilities so the result can go
    # straight to `categorical_td_learning`, which softmaxes internally; the two
    # agree because whole rows are selected and softmax is row-wise.
    target_logits1, target_logits2 = target_twin_critic(next_obs, next_actions)
    target_q1 = jnp.sum(jax.nn.softmax(target_logits1, axis=-1) * atoms, axis=-1)  # (B,)
    target_q2 = jnp.sum(jax.nn.softmax(target_logits2, axis=-1) * atoms, axis=-1)
    take_first = (target_q1 <= target_q2)[:, None]
    target_logits = jnp.where(take_first, target_logits1, target_logits2)

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
    return critic_loss


def ppo_critic_loss_fn(
    critic_model,
    observations,
    returns,
):
    """Calculates the MSE loss for the critic using PPO.

    `returns` is the GAE return target `A_raw + V_old`, built by the caller from
    the *unnormalized* advantage. Do not rebuild it here from the advantage the
    actor consumes: that one is standardized to zero mean / unit variance, which
    makes an unfittable regression target — the critic would be chasing its own
    previous output plus unit-variance noise.
    """
    # Both operands are (NUM_ENVS, BATCH_SIZE - 1): the last step has no return.
    v_t = critic_model(observations)[:, :-1, 0]
    critic_loss = jnp.mean((v_t - returns) ** 2)
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
):
    """MSE critic loss for MPO (policy evaluation).

    The bootstrap value is the expected Q of the target (old) policy at the next
    state, estimated by averaging the target critic over a handful of actions
    sampled from it. A single Q critic, as in the original paper.

    `num_action_samples` sets a sample shape, so it must stay a Python int: the
    agent declares it static on `_mpo_grad_steps`.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    # [S, B, A]. Gaussian, no squashing — bounded by clipping, consistent with
    # action selection.
    next_dist = target_actor_model(next_obs)
    next_actions = next_dist.sample(seed=key, sample_shape=(num_action_samples,))
    next_actions = jnp.clip(next_actions, -1.0, 1.0)
    next_actions = Agent.scale_to_env(next_actions, action_low, action_high)

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


def sac_critic_loss_fn(
    twin_critic,
    actor_model,
    target_twin_critic,
    samples,
    alpha,
    key,
    action_low,
    action_high,
):
    """Soft Bellman MSE for the twin critic (SAC).

    Same clipped double-Q target as `td3_critic_loss_fn`, minus target policy
    smoothing (the stochastic policy already smooths) and plus the entropy bonus
    -alpha * log pi(a'|s').

    The entropy bonus is applied only at the bootstrap state. At n_step 1 that is
    exactly the textbook soft target; at n > 1 the intermediate-step bonuses are
    dropped, the usual n-step SAC approximation — the stored actions came from an
    older policy, so their log-probs are not the current pi's anyway.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_dist = actor_model(next_obs)
    next_u = next_dist.sample(seed=key)
    next_actions = jnp.tanh(next_u)
    next_log_probs = next_dist.log_prob(next_u) - jnp.sum(
        jnp.log(1.0 - next_actions ** 2 + 1e-6), axis=-1
    )

    next_actions_scaled = Agent.scale_to_env(next_actions, action_low, action_high)

    target_q1, target_q2 = target_twin_critic(next_obs, next_actions_scaled)
    target_q = jnp.minimum(jnp.squeeze(target_q1), jnp.squeeze(target_q2))
    target_q = target_q - alpha * next_log_probs

    q1, q2 = twin_critic(obs, samples["actions"])
    q1, q2 = jnp.squeeze(q1), jnp.squeeze(q2)

    # Unlike DDPG/TD3 this loss IS the halved MSE, so `rlax.l2_loss`'s built-in
    # 0.5 is exactly the factor that belongs here.
    reward = jnp.squeeze(samples["rewards"])
    td_error1 = jax.vmap(rlax.td_learning)(q1, reward, samples["bootstrap"], target_q)
    td_error2 = jax.vmap(rlax.td_learning)(q2, reward, samples["bootstrap"], target_q)
    critic_loss = jnp.mean(rlax.l2_loss(td_error1)) + jnp.mean(rlax.l2_loss(td_error2))
    return critic_loss
