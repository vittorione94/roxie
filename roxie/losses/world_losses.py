"""World-model loss functions, fit on logged windows of real transitions."""

import jax
import jax.numpy as jnp
import optax


def window_masks(terminal, truncation):
    """Which steps of a `(B, k)` window are still the episode it started in.

    Returns:
        `(step_valid, state_valid)`, both float `(B, k)`. `step_valid[:, j]`
        marks a reward and termination label at step `j` as belonging to the
        window's episode: nothing before it ended. `state_valid[:, j]` marks the
        observation step `j` lands on, which additionally needs step `j` itself
        not to end — the pool auto-resets, so the next stored observation after
        a done is the first of a NEW episode, not a successor of this one.
    """
    done = (terminal | truncation).astype(jnp.float32)
    survived = jnp.cumprod(1.0 - done, axis=1)
    step_valid = jnp.concatenate(
        [jnp.ones_like(survived[:, :1]), survived[:, :-1]], axis=1
    )
    return step_valid, survived


def _masked_mean(x, mask):
    return jnp.sum(x * mask) / jnp.maximum(jnp.sum(mask), 1.0)


def world_rollout_loss_fn(
    dynamics,
    window,
    obs_mean,
    obs_std,
    reward_coef,
    termination_coef,
):
    """The k-step autoregressive fit: seeded at the logged start, never reset.

    The model is rolled forward on its OWN predictions and the logged actions,
    and is scored at every step against what really happened. Fitting one step
    at a time would only ever show it true inputs; this shows it its own errors,
    which is what lets an imagined rollout run past `k` without drifting off.

    Args:
        dynamics: the `DynamicsModel` being fit.
        window: a `Transition` sampled with a `(B, k + 1)` prefix. The last
          step contributes only its observation, as the target of step `k - 1`.
        obs_mean: the world's normalizer, constant for the pass.
        obs_std: likewise.
        reward_coef: weight of the reward term against the state term.
        termination_coef: weight of the termination term.

    Returns:
        `(loss, aux)`. The state term is measured in normalized units, so every
        observation dimension counts equally whatever its physical scale.
    """
    norm = (window.observation - obs_mean) / obs_std          # (B, k+1, D)
    actions = window.action[:, :-1]                           # (B, k, A)
    rewards = window.reward[:, :-1]
    terminal = window.terminal[:, :-1]
    step_valid, state_valid = window_masks(terminal, window.truncation[:, :-1])

    def body(z, action):
        delta, reward, logit = dynamics(z, action)
        z = z + delta
        return z, (z, reward, logit)

    # Time-major for the scan, then back. BPTT runs through all `k` steps.
    _, (pred, reward_hat, logit) = jax.lax.scan(
        body, norm[:, 0], jnp.swapaxes(actions, 0, 1)
    )
    pred = jnp.swapaxes(pred, 0, 1)                           # (B, k, D)
    reward_hat = jnp.swapaxes(reward_hat, 0, 1)               # (B, k)
    logit = jnp.swapaxes(logit, 0, 1)

    state_err = jnp.mean(jnp.square(pred - norm[:, 1:]), axis=-1)
    state_loss = _masked_mean(state_err, state_valid)
    reward_loss = _masked_mean(jnp.square(reward_hat - rewards), step_valid)
    term_loss = _masked_mean(
        optax.sigmoid_binary_cross_entropy(logit, terminal.astype(logit.dtype)),
        step_valid,
    )

    loss = state_loss + reward_coef * reward_loss + termination_coef * term_loss
    aux = {
        "state_loss": state_loss,
        "reward_loss": reward_loss,
        "term_loss": term_loss,
        # The paper's headline accuracy number: error at the window's far end.
        "state_err_last": _masked_mean(state_err[:, -1], state_valid[:, -1]),
        "state_err_first": _masked_mean(state_err[:, 0], state_valid[:, 0]),
        "valid_frac": jnp.mean(state_valid),
    }
    return loss, aux
