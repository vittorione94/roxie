from typing import Optional

import flax.struct as struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminal: jnp.ndarray
    log_probs: Optional[jnp.ndarray] = None    # Optional, used in some algorithms
    value: Optional[jnp.ndarray] = None        # Optional, used in some algorithms
    truncation: Optional[jnp.ndarray] = None   # Off-policy n-step masking + PPO GAE


def repack_samples(samples, gamma: float, n_step: int) -> dict:
    """Repack replay samples into the loss-fn dict, computing Bellman-target
    ingredients here so the critic losses stay 1-line: target = rewards +
    bootstrap * Q'(next_observations, pi'(next_observations)).

    Two buffer layouts:

    * Flat pair buffer (legacy, `samples.experience.first/.second`): plain
      1-step target — rewards = r_0, bootstrap = gamma * (1 - terminal).
    * Trajectory buffer (leaves (B, n_step+1, ...)): masked n-step return.
      The stream is contiguous per env row, and episodes are separated only by
      the stored `terminal`/`truncation` flags — the item AFTER a done is the
      next episode's reset state, so the window must never read past the first
      done. Per sample, with e_j = terminal, u_j = truncation (terminal wins
      when both fire on one step):
        - rewards  = sum_j gamma^j * alive_j * (1 - u_j) * r_j, where alive_j
          masks everything after the first done (inclusive of that step).
        - terminal at j: r_j counts, no bootstrap (exact episode return tail).
        - truncation at j: r_j does NOT count; bootstrap gamma^j * Q at o_j —
          the truncated step's own state. Its stored successor is a reset obs,
          so Q(o_j, pi(o_j)) stands in for r_j + gamma * V(o_{j+1}).
        - no done: bootstrap gamma^n at o_n.
      `bootstrap` carries the whole per-sample coefficient (0 for terminals),
      and `next_observations` is the selected bootstrap state.
    """
    exp = samples.experience
    if hasattr(exp, "first"):
        term = exp.first.terminal.astype(jnp.float32)
        return {
            "observations": exp.first.observation,
            "actions": exp.first.action,
            "rewards": exp.first.reward,
            "next_observations": exp.second.observation,
            "bootstrap": gamma * (1.0 - term),
        }

    n = int(n_step)
    obs_seq = exp.observation                                    # (B, n+1, D)
    r = exp.reward[:, :n].astype(jnp.float32)                    # (B, n)
    e = exp.terminal[:, :n].astype(jnp.float32)
    u = exp.truncation[:, :n].astype(jnp.float32) * (1.0 - e)
    stop = (1.0 - e) * (1.0 - u)
    # alive_j = prod_{i<j} stop_i: 1 up to AND INCLUDING the first done step.
    survived = jnp.cumprod(stop, axis=1)                         # (B, n)
    alive = jnp.concatenate([jnp.ones_like(stop[:, :1]), survived[:, :-1]], axis=1)

    disc = gamma ** jnp.arange(n, dtype=jnp.float32)             # (n,)
    returns = jnp.sum(disc * alive * (1.0 - u) * r, axis=1)      # (B,)

    # Bootstrap position one-hot over 0..n: first truncation -> its own obs;
    # clean window -> o_n; terminal anywhere -> all-zero row (no bootstrap).
    sel = jnp.concatenate([alive * u, survived[:, -1:]], axis=1)  # (B, n+1)
    disc_full = gamma ** jnp.arange(n + 1, dtype=jnp.float32)
    bootstrap = jnp.sum(sel * disc_full, axis=1)                 # (B,)
    boot_obs = jnp.einsum("bt,btd->bd", sel, obs_seq)            # (B, D)

    return {
        "observations": obs_seq[:, 0],
        "actions": exp.action[:, 0],
        "rewards": returns,
        "next_observations": boot_obs,
        "bootstrap": bootstrap,
    }


def unpack_sequence(samples, gamma: float, horizon: int) -> dict:
    """Unpack trajectory-buffer samples into a *sequence* dict for model-based
    agents (TD-MPC), keeping the time axis that `repack_samples` collapses.

    Both read the same trajectory buffer, but for opposite purposes:
    `repack_samples` folds a window into a single n-step Bellman target;
    here every step of the window is a supervised training target in its own
    right (predicted reward, TD target, next latent), so the window is returned
    intact alongside the masks that say which steps are trainable.

    The masking follows the same rule as `repack_samples`: the item AFTER a done
    is the next episode's reset state, so nothing may read past the first done.
    With e_j = terminal, u_j = truncation (terminal wins when both fire) and
    alive_j = 1 up to AND INCLUDING the first done, three masks fall out —
    they differ only in how much of step j they trust:

    * `reward_mask` = alive. The transition (o_j, a_j, r_j) is genuine even on
      the step that ends the episode, so the reward head trains on it.
    * `value_mask` = alive * (1 - u). The TD target needs o_{j+1}; at a
      truncation the stored successor is a reset state and the true one is
      gone, so that step is dropped rather than bootstrapped from garbage. At a
      terminal the step is kept — `bootstrap` is 0 there, so the target is just
      r_j and the (meaningless) successor is never read.
    * `consistency_mask` = alive * (1 - e) * (1 - u). The latent consistency
      term regresses the *predicted* next latent onto the encoding of the real
      successor, so it needs o_{j+1} to genuinely follow o_j — true only
      strictly before the first done.

    Returns leaves with the time axis intact: observations (B, H+1, D),
    actions/rewards/masks (B, H, ...).
    """
    exp = samples.experience
    if hasattr(exp, "first"):
        raise ValueError(
            "unpack_sequence needs a trajectory buffer (leaves shaped "
            "(B, T, ...)); got a flat pair buffer. Sequence-model agents must "
            "configure a trajectory buffer with sample_sequence_length "
            "= horizon + 1."
        )

    h = int(horizon)
    available = exp.reward.shape[1]
    if available < h + 1:
        raise ValueError(
            f"horizon {h} needs sample_sequence_length >= {h + 1}, "
            f"but the buffer samples windows of {available}."
        )

    r = exp.reward[:, :h].astype(jnp.float32)                    # (B, H)
    e = exp.terminal[:, :h].astype(jnp.float32)
    u = exp.truncation[:, :h].astype(jnp.float32) * (1.0 - e)
    stop = (1.0 - e) * (1.0 - u)
    # alive_j = prod_{i<j} stop_i: 1 up to AND INCLUDING the first done step.
    survived = jnp.cumprod(stop, axis=1)                         # (B, H)
    alive = jnp.concatenate([jnp.ones_like(stop[:, :1]), survived[:, :-1]], axis=1)

    return {
        "observations": exp.observation[:, : h + 1],             # (B, H+1, D)
        "actions": exp.action[:, :h],                            # (B, H, A)
        "rewards": r,
        "reward_mask": alive,
        "value_mask": alive * (1.0 - u),
        "consistency_mask": alive * stop,
        # Per-step bootstrap coefficient for the 1-step TD target at j:
        # target = r_j + bootstrap_j * Q(o_{j+1}, pi(o_{j+1})), zero at terminals.
        "bootstrap": gamma * (1.0 - e),
    }


# Helpers to serialize/deserialize bounds minimally
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        return (
            float(x) if x.shape == () else np.asarray(x, dtype=np.float32).tolist()[0]
        )  # TODO: check if this index is correct
    if hasattr(x, "item"):
        return x.item()
    return float(x)
