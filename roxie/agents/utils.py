from typing import Optional

import flax.struct as struct
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx


def build_optimizer(config, *, learning_rate: float, max_grad_norm: float = None):
    """Build one network's optax transform from a hydra `_target_` config.

    `config` names the optimizer family and its own hyperparameters (betas, eps,
    weight decay, ...); `learning_rate` comes from the agent's
    `<net>_learning_rate` arg so it stays a first-class swept/logged/checkpointed
    hyperparameter rather than hiding inside the optimizer block. A block that
    declares its own `learning_rate` wins — that is how a schedule is passed:

        actor_optimizer_config:
          _target_: optax.adamw
          weight_decay: 1e-4
          learning_rate:
            _target_: optax.cosine_decay_schedule
            init_value: 3e-4
            decay_steps: 1_000_000

    `config=None` falls back to plain Adam, for agents constructed directly from
    Python or loaded from a checkpoint that predates the block. Clipping stays
    outside the block: `max_grad_norm` is a top-level agent arg, and null/0
    disables it entirely.
    """
    if config is None:
        tx = optax.adam(learning_rate)
    else:
        overrides = {} if "learning_rate" in config else {"learning_rate": learning_rate}
        tx = hydra.utils.instantiate(config, **overrides)

    if max_grad_norm:
        # Clip first, then adapt, so the optimizer's moment estimates see the
        # already-clipped gradient.
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx


def network_rngs(seed: int, offset: int = 0) -> nnx.Rngs:
    """Parameter-init RNGs for one network, derived from the agent's `seed`.

    `offset` separates the networks of a single agent so twin critics never start
    identical (a clipped double-Q min over two identical heads is worthless).
    The convention is actor=0, critic=2, second critic=4.
    """
    return nnx.Rngs(params=seed + offset, dropout=seed + offset + 1)


@struct.dataclass
class Transition:
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminal: jnp.ndarray
    log_probs: Optional[jnp.ndarray] = None    # on-policy agents only (PPO)
    value: Optional[jnp.ndarray] = None        # on-policy agents only (PPO)
    truncation: Optional[jnp.ndarray] = None   # off-policy n-step masking + PPO GAE


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


# Action bounds -> JSON-friendly scalars/lists for the hyperparameter block written
# into checkpoints and logs. Per-actuator bounds stay a full list, so an env whose
# actuators have different ranges is not recorded as just the first one's.
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        if x.shape == ():
            return float(x)
        return np.asarray(x, dtype=np.float32).tolist()
    # A list/tuple arrives when the agent was built FROM a checkpoint: this
    # function wrote per-actuator bounds out as a list, and `Agent.load` feeds
    # that list straight back into the constructor, which re-serializes it on
    # its way to the hyperparameter banner. Without this branch the round trip
    # dies in `float([...])` — train fine, crash on every `play.py`.
    if isinstance(x, (list, tuple)):
        return [serialize_bound(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return float(x)
