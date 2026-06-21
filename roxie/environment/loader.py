from typing import Any, Callable, Optional

import jax.numpy as jnp
from flax import struct
from mujoco_playground import registry, wrapper
from mujoco_playground._src import mjx_env


@struct.dataclass
class WrapperState:
    """A dataclass to hold the state for the jittable wrapper.

    This makes state explicit, containing the original environment state
    and the wrapper's step counter.
    """

    env_state: mjx_env.State
    step_count: jnp.ndarray


class TerminationWrapper(wrapper.Wrapper):
    """
    Non-invasive wrapper that extends mujoco_playground's base Wrapper.

    Key principle: Never modify the original playground state.
    Instead, we return a tuple of (original_state, wrapper_info) or use
    a separate tracking mechanism.
    """

    def __init__(
        self,
        env: Any,
        max_episode_steps: int = 1000,
    ):
        super().__init__(env)

        self.max_episode_steps = max_episode_steps

        # External tracking (not part of state)
        self._current_step_count = 0
        self._episode_active = False

    def reset(self, key: jnp.ndarray) -> WrapperState:
        """Resets the environment and the wrapper's state."""
        # Reset the base environment to get the initial mjx_env.State
        initial_env_state = super().reset(key)

        # Normalize flags to 0-d jnp.bool_ for JAX consistency
        false_b = jnp.array(False, dtype=jnp.bool_)

        # Update info and explicitly set done=False with consistent dtype
        new_info = initial_env_state.info | {
            "truncation": false_b,
            "termination": false_b,
        }
        initial_env_state = initial_env_state.replace(done=false_b, info=new_info)

        # Return the initial WrapperState, starting the step count at 0
        return WrapperState(
            env_state=initial_env_state,
            step_count=jnp.zeros((), dtype=jnp.int32),
        )

    def step(self, state: WrapperState, action: jnp.ndarray) -> WrapperState:
        """
        Performs a step in the environment. This function is now pure
        and can be safely jitted.
        """
        # Step the underlying environment using its state
        next_env_state = super().step(state.env_state, action)

        # Increment the step count from the input state
        new_step_count = state.step_count + 1

        # Determine truncation based on the new step count
        truncated = new_step_count >= self.max_episode_steps

        # The episode is done if the base environment terminates OR if it's truncated.
        # next_env_state.done is the termination signal from the base env.
        done = jnp.logical_or(next_env_state.done, truncated)

        # Normalize flags to 0-d jnp.bool_
        trunc_b = jnp.asarray(truncated, dtype=jnp.bool_)
        term_b = jnp.asarray(next_env_state.done, dtype=jnp.bool_)
        done_b = jnp.asarray(done, dtype=jnp.bool_)

        # Update the info dictionary for observation purposes
        new_info = next_env_state.info | {
            "truncation": trunc_b,
            "termination": term_b,
        }

        # Create the final environment state with the updated done flag and info
        final_env_state = next_env_state.replace(done=done_b, info=new_info)

        # Return the new WrapperState containing the new env state and step count
        return WrapperState(env_state=final_env_state, step_count=new_step_count)


def resolve_loaded_impl(env: Any) -> str:
    """Read the physics backend actually used by a (possibly wrapped) env.

    Reads the value off the constructed env rather than echoing config, so the
    banner reflects what really loaded. The mocap env stores ``_impl``; the
    playground envs store it on ``_config.impl``.
    """
    base = env
    while hasattr(base, "env"):
        base = base.env
    impl = getattr(base, "_impl", None)
    if impl is None:
        cfg = getattr(base, "_config", None)
        if cfg is not None and "impl" in cfg:
            impl = cfg["impl"]
    return impl or "unknown"


def log_loaded_backend(env: Any, requested_impl: str | None = None) -> None:
    """Print a loud banner reporting the loaded physics backend."""
    actual = resolve_loaded_impl(env)
    mismatch = requested_impl is not None and actual != requested_impl

    bar = "=" * 70
    lines = [
        "",
        bar,
        f"  PHYSICS BACKEND LOADED:  impl = {actual.upper()}",
    ]
    if requested_impl is not None:
        if mismatch:
            lines.append(
                f"  !!! MISMATCH !!!  requested {requested_impl!r} "
                f"but env loaded {actual!r}"
            )
        else:
            lines.append(f"  OK - matches requested impl = {requested_impl!r}")
    lines += [bar, ""]
    print("\n".join(lines), flush=True)


def load_playground_env(
    env_name: str,
    impl: str | None = None,
    naconmax: int | None = None,
    njmax: int | None = None,
):
    """Load a mujoco_playground env, optionally forcing the physics backend.

    The playground env configs expose ``impl`` (``"jax"``/``"warp"``) plus the
    Warp contact/constraint budgets (``naconmax``/``njmax``). We override them
    only when explicitly provided so each env keeps its upstream-tuned default
    otherwise (e.g. Warp's per-env ``naconmax``).
    """
    env_cfg = registry.get_default_config(env_name)
    if impl is not None and "impl" in env_cfg:
        env_cfg.impl = impl
    if naconmax is not None and "naconmax" in env_cfg:
        env_cfg.naconmax = naconmax
    if njmax is not None and "njmax" in env_cfg:
        env_cfg.njmax = njmax

    env = registry.load(env_name, config=env_cfg)
    wrapped_env = TerminationWrapper(env)
    return wrapped_env, env_cfg
