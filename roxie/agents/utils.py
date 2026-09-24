"""Agent instantiation, optimizer creation, and replay buffer utilities."""

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

from roxie.utils.diagnostics import DIAGNOSTIC_MAX_KEYS


def build_agent(config, **kwargs):
    """Instantiates an agent from a Hydra configuration block.

    Uses `_recursive_=False` to leave nested sub-blocks unbuilt so the agent
    can process and instantiate them internally.

    Args:
        config: The raw or composed OmegaConf/dict configuration.
        **kwargs: Additional keyword arguments forwarded to `hydra.utils.instantiate`.

    Returns:
        The instantiated agent.
    """
    if OmegaConf.is_config(config):
        config = OmegaConf.to_container(
            config, resolve=True, throw_on_missing=True,
        )
    else:
        config = dict(config)

    return hydra.utils.instantiate(config, _recursive_=False, **kwargs)


def build_optimizer(config, *, learning_rate: float, max_grad_norm: float = None):
    """Builds an Optax gradient transformation from a configuration dictionary.

    Args:
        config: Hydra config naming the optimizer family and parameters, or None for Adam.
        learning_rate: Base learning rate override.
        max_grad_norm: Optional threshold for global gradient norm clipping.

    Returns:
        An Optax `GradientTransformation`.
    """
    if config is None:
        tx = optax.adam(learning_rate)
    else:
        overrides = {} if "learning_rate" in config else {"learning_rate": learning_rate}
        tx = hydra.utils.instantiate(config, **overrides)

    if max_grad_norm:
        tx = optax.chain(optax.clip_by_global_norm(max_grad_norm), tx)
    return tx


def make_optimizer(
    module: nnx.Module, config, *, learning_rate: float, max_grad_norm: float = None
) -> nnx.Optimizer:
    """Binds a Flax NNX module to an Optax optimizer over its parameters.

    Args:
        module: The NNX module whose parameters will be optimized.
        config: Optimizer configuration block.
        learning_rate: Base learning rate.
        max_grad_norm: Optional threshold for global gradient norm clipping.

    Returns:
        An `nnx.Optimizer` bound to `nnx.Param`.
    """
    return nnx.Optimizer(
        module,
        build_optimizer(
            config, learning_rate=learning_rate, max_grad_norm=max_grad_norm
        ),
        wrt=nnx.Param,
    )


def soft_update(target: nnx.Module, source: nnx.Module, tau: float) -> None:
    """Performs an in-place Polyak soft update from `source` into `target`.

    Computes `target <- (1 - tau) * target + tau * source`.

    Args:
        target: Target NNX module to mutate in place.
        source: Source NNX module containing updated parameters.
        tau: Interpolation coefficient.
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
    """Executes `n_steps` of a gradient step function on-device via `jax.lax.scan`.

    Args:
        state: The pytree carry (e.g., agent `TrainState` or tuple of modules).
        key: PRNG key split across the step iterations.
        n_steps: Static scan length.
        step_fn: Step function matching `(state, step_key, *extras) -> per_step_out`.
        extras: Additional per-step array inputs stacked on the leading axis.

    Returns:
        A tuple of `(final_state, stacked_outputs)`.
    """
    def body(carry, step_inputs):
        return carry, step_fn(carry, *step_inputs)

    keys = jax.random.split(key, n_steps)
    return jax.lax.scan(body, state, (keys, *extras))


def reduce_diagnostics(per_step: dict, denom) -> dict:
    """Reduces stacked per-step diagnostic arrays into scalar values.

    Keys listed in `DIAGNOSTIC_MAX_KEYS` are reduced via max; all others are 
    averaged by dividing their sum by `denom`.

    Args:
        per_step: Dictionary mapping metric names to stacked step arrays.
        denom: Scalar denominator for mean reductions.

    Returns:
        Dictionary mapping metric names to on-device scalar values.
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
    """Generates parameter initialization and dropout RNGs derived from a base seed.

    Args:
        seed: Base seed value.
        offset: Integer offset to separate distinct network streams.

    Returns:
        An `nnx.Rngs` instance configured for `params` and `dropout`.
    """
    return nnx.Rngs(params=seed + offset, dropout=seed + offset + 1)


@struct.dataclass
class Transition:
    """A single environment step transition pytree."""
    observation: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    terminal: jnp.ndarray
    log_probs: Optional[jnp.ndarray] = None
    value: Optional[jnp.ndarray] = None
    pre_action: Optional[jnp.ndarray] = None
    truncation: Optional[jnp.ndarray] = None


def transition_prototype(
    env_obs_size: int,
    env_action_size: int,
    *,
    truncation: bool = True,
    on_policy: bool = False,
) -> Transition:
    """Constructs a zero-filled `Transition` template for buffer allocation.

    Args:
        env_obs_size: Observation vector dimension.
        env_action_size: Action vector dimension.
        truncation: Whether to allocate the boolean truncation field.
        on_policy: Whether to allocate on-policy fields (`log_probs`, `value`, `pre_action`).

    Returns:
        A prototype `Transition` pytree instance.
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
    """Instantiates an off-policy replay buffer based on TD step horizon.

    Args:
        memory_config: Hydra memory configuration block.
        n_step: Temporal difference step horizon.

    Returns:
        A Flashbax flat or trajectory replay buffer instance.
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
    """Repacks Flashbax replay samples into a dictionary for loss functions.

    Computes $n$-step discounted returns and identifies bootstrap target states
    accounting for terminations and truncations.

    Args:
        samples: Sample batch from a Flashbax replay buffer.
        gamma: Discount factor.
        n_step: Temporal difference step horizon.

    Returns:
        A dictionary containing `observations`, `actions`, `rewards`,
        `next_observations`, and `bootstrap` discount multipliers.
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

    survived = jnp.cumprod(stop, axis=1)                         # (B, n)
    alive = jnp.concatenate([jnp.ones_like(stop[:, :1]), survived[:, :-1]], axis=1)

    disc = gamma ** jnp.arange(n, dtype=jnp.float32)             # (n,)
    returns = jnp.sum(disc * alive * (1.0 - u) * r, axis=1)      # (B,)

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


def serialize_bound(x):
    """Serializes array or scalar action bounds to JSON/YAML compatible floats or lists.

    Args:
        x: Action bound array, list, or scalar.

    Returns:
        A Python float or nested list of floats.
    """
    x = jax.device_get(x)
    if isinstance(x, (jnp.ndarray, np.ndarray)):
        if x.shape == ():
            return float(x)
        return np.asarray(x, dtype=np.float32).tolist()
    if isinstance(x, (list, tuple)):
        return [serialize_bound(v) for v in x]
    if hasattr(x, "item"):
        return x.item()
    return float(x)