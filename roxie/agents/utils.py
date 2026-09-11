from typing import Optional

import flashbax
import flax.struct as struct
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import nnx
from omegaconf import OmegaConf


# `device` has to be applied before the first `jax.*` call, long before an
# agent exists, so `train.py` reads it and `build_agent` drops it. The agent
# block's counterpart to `loader.TRAINER_ENV_KEYS`.
AGENT_PLACEMENT_KEYS = frozenset({"device"})


def build_agent(config, **kwargs):
    """Instantiate an agent from its config block, as `train.py` does.

    `_recursive_=False` keeps the nested `*_config` blocks unbuilt: the agent
    instantiates its own actor/critic/memory/optimizers, injecting shapes
    (in_features, action_dim, rngs, num_atoms) the caller cannot know.

    The block is resolved against the composed config FIRST, because
    `hydra.utils.instantiate` re-creates whatever it is handed as a fresh,
    parentless node — `${env.parallel_envs}` inside `memory_config` has nothing
    left to resolve against once that copy exists.
    """
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True,
        )
    else:
        config = dict(config)
    for key in AGENT_PLACEMENT_KEYS:
        config.pop(key, None)
    return hydra.utils.instantiate(config, _recursive_=False, **kwargs)


def build_optimizer(config, *, learning_rate: float, max_grad_norm: float = None):
    """Build one network's optax transform from a hydra `_target_` config.

    `config` names the optimizer family and its own hyperparameters;
    `learning_rate` comes from the agent's `<net>_learning_rate` arg so it stays
    a first-class swept/logged/checkpointed hyperparameter rather than hiding
    inside the optimizer block. A block that declares its own `learning_rate`
    wins — that is how a schedule is passed:

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
        # Clip first, then adapt, so the moment estimates see the clipped
        # gradient.
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx


def make_optimizer(
    module: nnx.Module, config, *, learning_rate: float, max_grad_norm: float = None
) -> nnx.Optimizer:
    """Bind one network to its optax transform.

    `wrt=nnx.Param` everywhere: the gradients handed to `.update()` are taken
    with `nnx.value_and_grad`, which differentiates w.r.t. the params alone, so
    the optimizer's slots must be keyed the same way. `max_grad_norm` is left
    unset for the scalar duals (SAC's temperature, MPO's Lagrange multipliers) —
    a global-norm clip over a one-element tree only rescales the step and fights
    the dual's own convergence.
    """
    return nnx.Optimizer(
        module,
        build_optimizer(
            config, learning_rate=learning_rate, max_grad_norm=max_grad_norm
        ),
        wrt=nnx.Param,
    )


def soft_update(target: nnx.Module, source: nnx.Module, tau: float) -> None:
    """Polyak-average `source`'s parameters into `target`, IN PLACE.

    `target <- (1 - tau) * target + tau * source`, the soft target update shared
    by every agent that keeps target networks (DDPG, D4PG, TD3, TD4, SAC, MPO).
    Mutates the live module rather than returning a new one, so it composes with
    the in-place style the fused gradient steps are written in — inside
    `fused_grad_steps` the carry IS these same objects after the body has run.
    """
    nnx.update(
        target,
        optax.incremental_update(
            new_tensors=nnx.state(source, nnx.Param),
            old_tensors=nnx.state(target, nnx.Param),
            step_size=tau,
        ),
    )


def fused_grad_steps(state, key: jax.Array, n_steps: int, step_fn, extras=()):
    """Run `step_fn` `n_steps` times on-device as one compiled program.

    A `lax.scan` over the train state, which is a pytree, so the state itself is
    the carry — nothing is split or merged per step. The scan (rather than a
    Python loop, which would unroll) keeps compile time and HLO size flat as
    `n_steps` grows, and the single dispatch costs one host round-trip instead
    of `n_steps`.

    Call it from inside the agent's own jitted entry point, which is where the
    loop-constant arguments (`gamma`, `tau`, the sampler, the normalization
    params) get hoisted and closed over.

    Args:
      state: the pytree to carry. A tuple of them works too and is how an
        agent's side modules (SAC's temperature, MPO's duals, and their
        optimizers) keep updating across the fused steps instead of resetting to
        their entry values on every one.
      key: split into one key per step and scanned over.
      n_steps: static; it is the scan length.
      step_fn: `(state, step_key, *extras_t) -> per_step_outputs`. Mutates the
        state in place; the carry is that same object afterwards, so it should
        not rebuild it.
      extras: additional per-step `xs`, stacked on the leading axis, e.g. TD3's
        delayed-update mask. A `None` entry is an empty pytree node and arrives
        as `None`, which is how `policy_delay == 1` skips the branch rather than
        tracing an all-True mask.

    Returns `(state, outputs)` — the state after the last step, and the per-step
    outputs stacked on a leading `n_steps` axis.
    """
    def body(carry, step_inputs):
        return carry, step_fn(carry, *step_inputs)

    keys = jax.random.split(key, n_steps)
    return jax.lax.scan(body, state, (keys, *extras))


# Diagnostic keys reduced with `max` rather than a mean — over the steps of one
# fused burst (`reduce_diagnostics`) and over the bursts of one epoch
# (`Agent.pop_diagnostics`) alike. Listed once here rather than per agent
# because the rule belongs to the metric, not to whoever emits it: every
# deterministic arm publishes `pre_act_max`, and a mean of a per-batch maximum
# in any of them would wash out the outlier the metric exists to expose.
DIAGNOSTIC_MAX_KEYS = frozenset({"pre_act_max"})
# Counts rather than levels: totalled across an epoch instead of averaged. Only
# the epoch reduction sees these — a burst records its own count already summed.
DIAGNOSTIC_SUM_KEYS = frozenset({"kl_early_stops"})


def reduce_diagnostics(per_step: dict, denom) -> dict:
    """Collapse one burst's stacked per-step diagnostics to a scalar per key.

    `per_step` is what `fused_grad_steps` stacked on the leading step axis;
    `denom` is what the mean divides by — `n_steps` for critic-side keys, the
    number of *actor update* steps for actor-side ones, which under a
    `policy_delay` is the smaller of the two and is why the two sides are
    reduced by separate calls.

    Reduced on-device so the host sees one scalar per metric per burst, never a
    per-step array.
    """
    return {
        key: (
            jnp.max(value)
            if key in DIAGNOSTIC_MAX_KEYS
            else jnp.sum(value) / denom
        )
        for key, value in per_step.items()
    }


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
    # The pre-tanh `u` behind `action`. On-policy only, and only because the
    # squash is not invertible in float32: see `transition_prototype`.
    pre_action: Optional[jnp.ndarray] = None
    truncation: Optional[jnp.ndarray] = None   # off-policy n-step masking + PPO GAE


def transition_prototype(
    env_obs_size: int,
    env_action_size: int,
    *,
    truncation: bool = True,
    on_policy: bool = False,
) -> Transition:
    """One zero-filled transition, the schema `replay.init` allocates against.

    Every leaf of the buffer is sized and typed from here, so a single factory
    rather than a literal per agent. A field left off is `None`, an empty pytree
    node, and costs no memory.

    `truncation` is stored alongside `terminal` (never folded into one `done`)
    because the two mean different things to a Bellman target: a genuine
    termination zeroes the bootstrap, while a time-limit / clip-end truncation
    must still bootstrap the next-state value. Off-policy agents additionally
    need it to stop n-step windows at episode boundaries the terminal flag does
    not mark; MPO, whose 1-step target reads only `terminal`, omits it.

    `on_policy` adds the behaviour log-prob and value estimate PPO stores at
    acting time, plus the pre-tanh `u` behind each stored action. That last is
    not redundant with `action`: tanh is not invertible in float32 — `tanh(u)`
    rounds to exactly 1.0 for |u| >= 8, so recovering `u` with an arctanh pins
    every saturated draw to the same rail (~7.25) while the policy mean walks
    past it, and the ratio explodes until `target_kl` fires on arithmetic rather
    than policy drift.
    """
    extra = {}
    if truncation:
        extra["truncation"] = jnp.zeros((), dtype=jnp.bool_)
    if on_policy:
        extra["log_probs"] = jnp.zeros((), dtype=jnp.float32)
        extra["value"] = jnp.zeros((), dtype=jnp.float32)
        extra["pre_action"] = jnp.zeros(env_action_size, dtype=jnp.float32)
    return Transition(
        observation=jnp.zeros(env_obs_size, dtype=jnp.float32),
        action=jnp.zeros(env_action_size, dtype=jnp.float32),
        reward=jnp.zeros((), dtype=jnp.float32),
        terminal=jnp.zeros((), dtype=jnp.bool_),
        **extra,
    )


def build_replay(memory_config, n_step: int):
    """The replay buffer for an off-policy agent, given its TD horizon.

    n-step targets need `n_step + 1` consecutive items per sample, which the flat
    pair buffer cannot serve, so anything past 1-step switches to a trajectory
    buffer with `period=1` (windows at every offset). The yaml keeps the
    flat-buffer schema either way — `max_length` / `min_length` are TOTAL
    transitions — so the lengths are converted here to flashbax's per-row
    time-axis ones rather than in every config.
    """
    if n_step <= 1:
        return hydra.utils.instantiate(memory_config)

    add_batch = int(memory_config.add_batch_size)
    return flashbax.make_trajectory_buffer(
        add_batch_size=add_batch,
        sample_batch_size=int(memory_config.sample_batch_size),
        sample_sequence_length=n_step + 1,
        period=1,
        min_length_time_axis=max(
            n_step + 1, int(memory_config.min_length) // add_batch
        ),
        max_length_time_axis=int(memory_config.max_length) // add_batch,
    )


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


# Per-actuator bounds stay a full list, so an env whose actuators have
# different ranges is not recorded as just the first one's.
def serialize_bound(x):
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        if x.shape == ():
            return float(x)
        return np.asarray(x, dtype=np.float32).tolist()
    # A list arrives when the agent was built from a checkpoint: `Agent.load`
    # feeds the serialized bounds back through the constructor. Without this
    # branch that round trip dies in `float([...])`.
    if isinstance(x, (list, tuple)):
        return [serialize_bound(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return float(x)
