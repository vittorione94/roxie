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
    """Target actions at `next_obs`, in env scale, with TD3 policy smoothing.

    Shared by every off-policy critic loss that bootstraps through a
    deterministic target actor. D4PG/TD4 keep this call for signature parity
    even though the papers do not smooth; their configs set the noise to 0,
    making it a no-op.

    UNITS: `target_policy_noise` and `target_noise_clip` are in the actor's own
    [-1, 1] output space, which is where the TD3 paper's 0.2 / 0.5 are defined
    and where `NoiseModule.add_noise` applies the exploration noise. The
    smoothing is therefore added *before* `scale_to_env`, exactly as
    `Agent.deterministic_step_fn` does it — so the two noise sources are
    directly comparable and neither depends on the env's action span. Scaling
    these by the span instead (as this did until the AcrobotSwingup TD3 vs DDPG
    regression) doubles them on any [-1, 1] env, putting the smoothing kernel at
    4x the exploration noise.

    Returns ``(next_actions, smooth_clip_frac)``, the latter being the share of
    smoothing samples the clip bit: near 0 means `target_noise_clip` is inert,
    near 1 that it has flattened the Gaussian into a two-point distribution.
    """
    next_actions = target_actor_model(next_obs)  # [-1, 1]

    noise = jax.random.normal(noise_key, next_actions.shape) * target_policy_noise
    clipped_noise = jnp.clip(noise, -target_noise_clip, target_noise_clip)
    smooth_clip_frac = jnp.mean(
        (jnp.abs(noise) > target_noise_clip).astype(jnp.float32)
    )

    # `scale_to_env` is an increasing affine map, so clipping in [-1, 1] first
    # gives the same set as clipping to [action_low, action_high].
    next_actions = jnp.clip(next_actions + clipped_noise, -1.0, 1.0)
    return scale_to_env(next_actions, action_low, action_high), smooth_clip_frac


def categorical_mean(logits, atoms) -> jnp.ndarray:
    """Expected value of a categorical critic's output over its fixed support."""
    return jnp.sum(jax.nn.softmax(logits, axis=-1) * atoms, axis=-1)


def value_aux(q_buffer, target_q, td_error, **extra) -> dict:
    """The value-health diagnostics every critic loss reports.

    Shared for the same reason `actor_aux` is: the failure mode is shared. A
    `q_buffer` that climbs away from `q_target` is the overestimation spiral the
    clipped double-Q exists to curb, and `td_abs` says whether the critic is
    tracking its own target at all. Read off the loss's own forward pass, so the
    instrumentation costs no extra compute.

    `extra` carries what only some critics have: the twin heads' disagreement,
    the target-smoothing clip rate, the target actions' rail fraction.
    """
    return {
        "q_buffer": jnp.mean(q_buffer),
        "q_target": jnp.mean(target_q),
        "td_abs": jnp.mean(jnp.abs(td_error)),
        **extra,
    }


def action_rail_frac(actions, action_low, action_high) -> jnp.ndarray:
    """Share of `actions` (env scale) sitting on the action-space bounds.

    High at the bootstrap state means the target actor has saturated too, so the
    bootstrap is being evaluated exactly where the critic has the least data.
    """
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
    """MSE loss for the critic. Returns ``(loss, aux)``, as every loss here does.

    Single-headed, so `aux` carries no `twin_gap` — the epoch reduction takes
    whatever keys the agent reports, so a metric that has no meaning for an
    agent is simply absent rather than logged as a zero.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )
    next_q = jnp.squeeze(target_critic_model(next_obs, next_actions))
    current_q = jnp.squeeze(critic_model(obs, samples["actions"]))

    # `rlax.td_learning` owns the stop_gradient on the target and is defined
    # per scalar transition, hence the vmap over the batch.
    reward = jnp.squeeze(samples["rewards"])
    td_error = jax.vmap(rlax.td_learning)(
        current_q, reward, samples["bootstrap"], next_q
    )

    # NOT `rlax.l2_loss`, which carries a 0.5 factor: this loss is the unhalved
    # MSE, and halving it would halve the critic's effective learning rate.
    critic_loss = jnp.mean(jnp.square(td_error))

    # Restates the target `td_learning` builds internally; XLA folds it into the
    # same subexpression.
    aux = value_aux(
        current_q,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q),
        td_error,
        target_smooth_clip_frac=smooth_clip_frac,
        target_act_rail_frac=action_rail_frac(next_actions, action_low, action_high),
    )
    return critic_loss, aux


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

    Returns ``(loss, aux)``, as every loss here does.
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

    # Restates the target `td_learning` builds internally; XLA folds it into
    # the same subexpression.
    aux = value_aux(
        q1,
        jax.lax.stop_gradient(reward + samples["bootstrap"] * next_q),
        td_error1,
        # Should stay small relative to |Q|; a widening gap means the min is
        # doing heavy lifting.
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
    atoms,
):
    """Categorical distributional critic loss for D4PG (Barth-Maron et al. 2018).

    The critic outputs logits over a fixed support `atoms`. The target is the
    target critic's categorical at (s', pi'(s')) with its support shifted to
    `rewards + bootstrap * atoms`, projected back onto `atoms`, then
    cross-entropy against the online critic. That whole chain is
    `rlax.categorical_td_learning`, which is defined for one transition, hence
    the vmap over the batch with `atoms` broadcast.

    Returns ``(loss, aux)``, as every loss here does. The `aux` values are the
    categoricals' EXPECTED values, which is what makes them comparable against
    the scalar-critic agents' — the loss itself is a cross-entropy, so `td_abs`
    here is a diagnostic of the same quantity but not a term in it.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    target_logits = target_critic_model(next_obs, next_actions)  # (B, num_atoms)
    logits = critic_model(obs, samples["actions"])               # (B, num_atoms)
    reward = jnp.squeeze(samples["rewards"])

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

    Returns ``(loss, aux)``, as every loss here does; see `d4pg_critic_loss_fn`
    on what the expected values in it mean for a distributional critic.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_actions, smooth_clip_frac = _smoothed_target_actions(
        target_actor_model, next_obs, noise_key,
        target_policy_noise, target_noise_clip, action_low, action_high,
    )

    # Selected on the logits, not the probabilities, so the result can go
    # straight to `categorical_td_learning`, which softmaxes internally.
    target_logits1, target_logits2 = target_twin_critic(next_obs, next_actions)
    target_q1 = categorical_mean(target_logits1, atoms)  # (B,)
    target_q2 = categorical_mean(target_logits2, atoms)
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

    q1 = categorical_mean(logits1, atoms)
    target_q = jax.lax.stop_gradient(
        reward + samples["bootstrap"] * jnp.minimum(target_q1, target_q2)
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

    # [S, B, A]. Tanh-squashed Gaussian, so the draws are already in (-1, 1)
    # and nothing is clipped on top — as in action selection.
    next_dist = target_actor_model(next_obs)
    next_actions = next_dist.sample(seed=key, sample_shape=(num_action_samples,))
    next_actions = scale_to_env(next_actions, action_low, action_high)

    next_obs_tiled = jnp.broadcast_to(next_obs, (num_action_samples,) + next_obs.shape)
    next_q = target_critic_model(next_obs_tiled, next_actions)  # [S, B, 1]
    next_q = jnp.mean(jnp.squeeze(next_q, axis=-1), axis=0)  # [B]

    reward = jnp.squeeze(samples["rewards"])
    term = samples["terminals"].astype(jnp.float32)
    target_q = reward + gamma * (1.0 - term) * next_q
    target_q = jax.lax.stop_gradient(target_q)

    # Stored actions are already in env scale.
    current_q = jnp.squeeze(critic_model(obs, samples["actions"]))
    critic_loss = jnp.mean((current_q - target_q) ** 2)
    aux = value_aux(
        current_q,
        target_q,
        current_q - target_q,
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
):
    """Soft Bellman MSE for the twin critic (SAC).

    Same clipped double-Q target as `td3_critic_loss_fn`, minus target policy
    smoothing (the stochastic policy already smooths) and plus the entropy bonus
    -alpha * log pi(a'|s').

    The entropy bonus is applied only at the bootstrap state. At n_step 1 that is
    exactly the textbook soft target; at n > 1 the intermediate-step bonuses are
    dropped, the usual n-step SAC approximation — the stored actions came from an
    older policy, so their log-probs are not the current pi's anyway.

    Returns ``(loss, aux)``, as every loss here does. `q_target` includes the
    entropy bonus, i.e. it is the SOFT target the critic actually regresses on.
    """
    obs = samples["observations"]
    next_obs = samples["next_observations"]

    next_dist = actor_model(next_obs)
    next_actions, next_pre = next_dist.sample_from_pre(seed=key)
    next_log_probs = next_dist.log_prob_from_pre(next_pre)

    next_actions_scaled = scale_to_env(next_actions, action_low, action_high)

    target_q1, target_q2 = target_twin_critic(next_obs, next_actions_scaled)
    target_q = jnp.minimum(jnp.squeeze(target_q1), jnp.squeeze(target_q2))
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
