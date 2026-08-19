from typing import Any, Callable, NamedTuple, Optional

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

        self._current_step_count = 0
        self._episode_active = False

    def reset(self, key: jnp.ndarray, *reset_args) -> WrapperState:
        """Resets the environment and the wrapper's state.

        `*reset_args` is forwarded verbatim to the wrapped env, so envs with a
        richer reset signature (the mocap env takes negative-mining weights as a
        traced second argument) work through the wrapper without it needing to
        know what those arguments mean.
        """
        initial_env_state = self.env.reset(key, *reset_args)

        false_b = jnp.array(False, dtype=jnp.bool_)

        new_info = initial_env_state.info | {
            "truncation": false_b,
            "termination": false_b,
        }
        initial_env_state = initial_env_state.replace(done=false_b, info=new_info)

        return WrapperState(
            env_state=initial_env_state,
            step_count=jnp.zeros((), dtype=jnp.int32),
        )

    def step(self, state: WrapperState, action: jnp.ndarray) -> WrapperState:
        """Performs a step in the environment. Pure, so it can be jitted."""
        next_env_state = super().step(state.env_state, action)

        new_step_count = state.step_count + 1

        # Wrapper-level truncation: the fixed step-limit time-out.
        step_truncated = new_step_count >= self.max_episode_steps

        # The base env may fold its own truncation into `done` (e.g. a clip-end
        # cutoff, surfaced via info["truncation"]). Pull it back out so it does NOT
        # count as termination: a truncated transition must still bootstrap the
        # next-state value in the Bellman target, whereas marking it terminal zeroes
        # the bootstrap and collapses Q at the cutoff. Envs that don't distinguish
        # truncation default to False, leaving this unchanged.
        env_truncation = next_env_state.info.get(
            "truncation", jnp.array(False, dtype=jnp.bool_)
        )

        # Genuine termination = base env done with any env-internal truncation
        # removed. Truncation = step-limit OR env-internal. Done = either, so
        # the trainer still auto-resets at the cutoff.
        terminated = jnp.logical_and(
            next_env_state.done, jnp.logical_not(env_truncation)
        )
        truncated = jnp.logical_or(step_truncated, env_truncation)
        done = jnp.logical_or(next_env_state.done, step_truncated)

        trunc_b = jnp.asarray(truncated, dtype=jnp.bool_)
        term_b = jnp.asarray(terminated, dtype=jnp.bool_)
        done_b = jnp.asarray(done, dtype=jnp.bool_)

        new_info = next_env_state.info | {
            "truncation": trunc_b,
            "termination": term_b,
        }

        final_env_state = next_env_state.replace(done=done_b, info=new_info)

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


class EnvBundle(NamedTuple):
    """Normalized result of an env builder.

    ``test_env`` is the env used for evaluation rollouts (None means "reuse the
    training env"); ``env_cfg`` is the resolved env config when one exists (the
    playground registry returns one; bespoke example envs may not).
    """

    env: Any
    test_env: Any
    env_cfg: Any


def build_playground_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for mujoco_playground envs.

    This is the default builder: experiments that don't set ``env.builder`` get
    a playground env loaded by ``env.env_name``. ``mode`` is part of the builder
    protocol (train.py passes "train", play.py "play") but playground envs load
    identically for both. Override semantics for ``naconmax``/``njmax`` match
    ``load_playground_env``: ``None`` leaves the env's upstream Warp budget.
    """
    env, env_cfg = load_playground_env(
        cfg_env.env_name,
        impl=cfg_env.get("impl", "jax"),
        naconmax=cfg_env.get("naconmax", None),
        njmax=cfg_env.get("njmax", None),
    )
    return EnvBundle(env=env, test_env=None, env_cfg=env_cfg)


# Dotted path of the default builder, used when ``env.builder`` is unset.
DEFAULT_BUILDER = "roxie.environment.loader.build_playground_env"
