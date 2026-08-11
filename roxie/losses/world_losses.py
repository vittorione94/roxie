"""Losses for TD-MPC's latent world model (Hansen et al. 2022).

Two losses, applied with two separate optimizers:

* `tdmpc_model_loss_fn` trains encoder + dynamics + reward + twin Q jointly, by
  unrolling the latent dynamics from a single encoded observation and scoring
  every imagined step against the real trajectory.
* `tdmpc_policy_loss_fn` trains the policy prior pi to maximize Q on those
  (detached) latents. Pi is only a proposal distribution for the planner, so it
  never feeds gradients back into the model.

Neither is decorated with `nnx.jit`: both are called from inside the agent's
already-jitted fused gradient step, following the `mpo_*_loss_fn` precedent.

Action-scale convention matches the rest of the repo — the buffer stores actions
in env units, so the world model consumes env-unit actions and the policy's
[-1, 1] output is scaled before it reaches the model.
"""

import jax
import jax.numpy as jnp

from roxie.agents.agent import Agent


def _weighted_mean(per_step: jnp.ndarray, weights: jnp.ndarray) -> jnp.ndarray:
    """Mean of `per_step` weighted by `weights` (rho^t times a validity mask).

    A weighted *mean* rather than the paper's weighted sum: with masking, a sum
    would make the gradient magnitude depend on how many valid steps happened to
    land in the batch — a batch full of episode boundaries would quietly train
    with a smaller effective learning rate. Normalizing decouples the two. The
    two differ by a roughly constant factor, so the loss coefficients keep their
    usual meaning relative to each other.
    """
    return jnp.sum(weights * per_step) / jnp.maximum(jnp.sum(weights), 1e-6)


def tdmpc_model_loss_fn(
    model,
    target_model,
    policy_model,
    samples,
    obs_mean,
    obs_std,
    obs_clip,
    action_low,
    action_high,
    rho: float,
    reward_coef: float,
    value_coef: float,
    consistency_coef: float,
):
    """Joint TOLD objective over an imagined rollout.

    From z_0 = h(o_0) the latent dynamics are rolled forward using the *real*
    actions taken, never re-encoding intermediate observations. Each imagined
    step t contributes three terms, discounted by rho^t so early (more reliable)
    steps dominate:

    * reward:      (R(z_t, a_t) - r_t)^2
    * value:       (Q_i(z_t, a_t) - [r_t + gamma * Q'_min(z'_{t+1}, pi(z'_{t+1}))])^2
    * consistency: ||d(z_t, a_t) - sg(h'(o_{t+1}))||^2

    where primed nets are the EMA target model. The consistency term is what
    makes the latent task-oriented: nothing reconstructs the observation, so the
    encoder is free to discard everything the reward and value heads don't need.

    Gradients flow into every component of `model` through the rolled latent
    chain; `target_model` and `policy_model` are held fixed.

    Returns `(loss, aux)`; `aux["latents"]` carries the detached rollout for
    `tdmpc_policy_loss_fn`, so pi's update doesn't re-run the dynamics.
    """
    obs = Agent.normalize_obs(samples["observations"], obs_mean, obs_std, obs_clip)
    actions = samples["actions"]  # (B, H, A), env units

    # --- Bellman targets, computed once for the whole window ---------------
    # The target encoder sees every real successor o_1..o_H at once; z'_{t+1}
    # is both the TD bootstrap state and the consistency regression target.
    next_z_target = jax.lax.stop_gradient(target_model.encode(obs[:, 1:]))  # (B,H,L)
    next_actions = Agent.scale_to_env(
        policy_model(next_z_target), action_low, action_high
    )
    tq1, tq2 = target_model.q(next_z_target, next_actions)
    next_q = jnp.minimum(jnp.squeeze(tq1, -1), jnp.squeeze(tq2, -1))  # (B, H)
    td_target = jax.lax.stop_gradient(
        samples["rewards"] + samples["bootstrap"] * next_q
    )  # (B, H)

    # --- Imagined rollout, time-major so it can be scanned ------------------
    def body(z, xs):
        a_t, r_t, td_t, z_next_target = xs

        q1, q2 = model.q(z, a_t)
        r_hat = model.predict_reward(z, a_t)
        z_next = model.next(z, a_t)

        reward_err = jnp.square(jnp.squeeze(r_hat, -1) - r_t)  # (B,)
        value_err = jnp.square(jnp.squeeze(q1, -1) - td_t) + jnp.square(
            jnp.squeeze(q2, -1) - td_t
        )
        consistency_err = jnp.mean(jnp.square(z_next - z_next_target), axis=-1)

        return z_next, (reward_err, value_err, consistency_err, z)

    z0 = model.encode(obs[:, 0])
    _, (reward_err, value_err, consistency_err, latents) = jax.lax.scan(
        body,
        z0,
        (
            jnp.swapaxes(actions, 0, 1),
            jnp.swapaxes(samples["rewards"], 0, 1),
            jnp.swapaxes(td_target, 0, 1),
            jnp.swapaxes(next_z_target, 0, 1),
        ),
    )  # each (H, B, ...); `latents` is (H, B, L) holding z_0..z_{H-1}

    # --- Weighted, masked reduction ----------------------------------------
    horizon = actions.shape[1]
    rho_t = (rho ** jnp.arange(horizon, dtype=jnp.float32))[:, None]  # (H, 1)
    reward_w = rho_t * jnp.swapaxes(samples["reward_mask"], 0, 1)
    value_w = rho_t * jnp.swapaxes(samples["value_mask"], 0, 1)
    consistency_w = rho_t * jnp.swapaxes(samples["consistency_mask"], 0, 1)

    reward_loss = _weighted_mean(reward_err, reward_w)
    value_loss = _weighted_mean(value_err, value_w)
    consistency_loss = _weighted_mean(consistency_err, consistency_w)

    loss = (
        reward_coef * reward_loss
        + value_coef * value_loss
        + consistency_coef * consistency_loss
    )

    aux = {
        "loss/reward": reward_loss,
        "loss/value": value_loss,
        "loss/consistency": consistency_loss,
        # Detached so pi's update can reuse the rollout without threading
        # gradients back into the model.
        "latents": jax.lax.stop_gradient(latents),
        "policy_weights": reward_w,
    }
    return loss, aux


def tdmpc_policy_loss_fn(
    policy_model,
    model,
    latents,
    weights,
    action_low,
    action_high,
):
    """Train the policy prior pi to maximize Q on the model's own latents.

    `latents` (H, B, L) and `weights` (H, B) come straight from the model loss's
    aux — already detached, so this differentiates w.r.t. `policy_model` only and
    the model is a fixed critic here (same arrangement as `td3_actor_loss_fn`,
    where the critic is a non-differentiated argument).

    Pi does not act on its own: the planner uses it to seed a fraction of its
    candidate trajectories, so a merely decent proposal is enough.
    """
    actions = Agent.scale_to_env(policy_model(latents), action_low, action_high)
    q1, q2 = model.q(latents, actions)
    q = jnp.minimum(jnp.squeeze(q1, -1), jnp.squeeze(q2, -1))  # (H, B)
    return -_weighted_mean(q, weights)
