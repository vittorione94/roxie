"""Environment loading, MuJoCo adapters, and vector environment builders."""

import functools
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf, open_dict
from mujoco_playground import registry

from roxie.environment import functional
from roxie.environment.functional import FuncEnv
from roxie.environment.vector import BatchVectorEnv, JaxVectorEnv


class PlaygroundFuncEnv(FuncEnv):
    """Functional environment wrapper for `mujoco_playground` environments."""

    def __init__(self, env: Any, impl: str):
        self.env = env
        self.observation_space = functional.unbounded_box(env.observation_size)
        ctrl_range = env.mj_model.actuator_ctrlrange
        self.action_space = functional.box(ctrl_range[:, 0], ctrl_range[:, 1])
        self.metadata = {"jax": True, "impl": impl}

    def initial(self, rng, params=None):
        return self.env.reset(rng)

    def transition(self, state, action, rng, params=None):
        return self.env.step(state, action)

    def observation(self, state, rng, params=None):
        return state.obs

    def reward(self, state, action, next_state, rng, params=None):
        return next_state.reward

    def terminal(self, state, rng, params=None):
        return state.done.astype(jnp.bool_)

    def truncal(self, state, rng, params=None):
        return jnp.asarray(state.info.get("truncation", False), dtype=jnp.bool_)

    def transition_info(self, state, action, next_state, params=None):
        return {"metrics": next_state.metrics}

    def __getattr__(self, name):
        if name.startswith("__") or "env" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.env, name)


BATCH_IMPL = "batch"


def _read_impl(env: Any) -> str:
    """Extracts the underlying physics implementation string from an environment."""
    impl = getattr(env, "_impl", None)
    if impl is None:
        cfg = getattr(env, "_config", None)
        if cfg is not None and "impl" in cfg:
            impl = cfg["impl"]
    return impl or "unknown"


def resolve_loaded_impl(env: Any) -> str:
    """Resolves the loaded physics backend implementation from a driver or functional env."""
    func_env = getattr(env, "func_env", env)
    meta = getattr(func_env, "metadata", None) or {}
    return meta.get("impl") or _read_impl(func_env)


def resolve_agent_device() -> tuple[str, str]:
    """Identifies the active agent backend device ('cpu' or 'gpu', backend_string)."""
    backend = jax.default_backend()
    return ("cpu" if backend == "cpu" else "gpu", backend)


def resolve_env_device(env: Any) -> tuple[str, str]:
    """Identifies the physics execution device and backend description."""
    impl = resolve_loaded_impl(env)
    if impl == BATCH_IMPL:
        return "cpu", "native MuJoCo, mujoco.batch thread pool"
    if impl == "native":
        return "cpu", "native MuJoCo, one world"
    device, backend = resolve_agent_device()
    return device, f"{impl} kernels on the JAX device ({backend})"


def log_loaded_backend(
    env: Any,
    requested_impl: str | None = None,
    device: str | None = None,
) -> None:
    """Logs runtime execution devices and checks for configuration mismatches."""
    actual = resolve_loaded_impl(env)
    agent_actual, agent_detail = resolve_agent_device()
    env_actual, env_detail = resolve_env_device(env)

    bar = "=" * 70
    lines = [
        "",
        bar,
        f"  PHYSICS BACKEND LOADED:  impl = {actual.upper()}",
        f"  AGENT (networks, optimizers, replay):  {agent_actual.upper():<4}"
        f"  [{agent_detail}]",
        f"  ENV   (physics):                       {env_actual.upper():<4}"
        f"  [{env_detail}]",
    ]
    for name, declared, got in (
        ("env.impl", requested_impl, actual),
        ("runtime.device (agent)", device, agent_actual),
        ("runtime.device (env)", device, env_actual),
    ):
        if declared is None:
            continue
        if str(declared) != got:
            lines.append(
                f"  !!! MISMATCH !!!  {name} declares {str(declared)!r} "
                f"but the run is on {got!r}"
            )
        else:
            lines.append(f"  OK - matches {name} = {str(declared)!r}")
    lines += [bar, ""]
    print("\n".join(lines), flush=True)


DEFAULT_WARP_GRAPH_MODE = "WARP_STAGED_EX"


def _put_warp_model(env: Any, graph_mode: str) -> None:
    """Re-puts a playground environment's MJX model for a long Warp run."""
    from mujoco import mjx
    from mujoco.mjx.third_party import mujoco_warp as mjw
    from mujoco.mjx.warp import types as mjxw_types

    if not hasattr(env, "_mjx_model"):
        raise RuntimeError(
            f"{type(env).__name__} has no `_mjx_model` to re-put; mujoco_playground "
            "changed shape."
        )
    model = mjx.put_model(
        env.mj_model,
        impl="warp",
        graph_mode=getattr(mjxw_types.GraphMode, graph_mode),
    )
    # Playground caps the solver low on purpose, so these two print per world per
    # step: 40-50k lines a minute on the humanoids at 256 worlds (2026-11-06). The
    # contact and constraint overflows stay on; those are a budget set too small.
    capped = mjw.OverflowType.ITERATIONS | mjw.OverflowType.LS_ITERATIONS
    impl = model.opt._impl
    impl = impl.replace(warn_overflow=int(impl.warn_overflow) & ~int(capped))
    env._mjx_model = model.replace(opt=model.opt.replace(_impl=impl))


def load_playground_env(
    env_name: str,
    impl: str | None = None,
    naconmax: int | None = None,
    njmax: int | None = None,
    graph_mode: str | None = None,
):
    """Loads a `mujoco_playground` environment with optional overrides."""
    env_cfg = registry.get_default_config(env_name)
    # `batch` swaps only the step: resets and the task's model stay on MJX.
    if impl is not None:
        env_cfg.impl = "jax" if impl == BATCH_IMPL else impl
    if naconmax is not None:
        env_cfg.naconmax = naconmax
    if njmax is not None:
        env_cfg.njmax = njmax

    env = registry.load(env_name, config=env_cfg)
    if env_cfg.impl == "warp":
        _put_warp_model(env, graph_mode or DEFAULT_WARP_GRAPH_MODE)
    loaded_impl = BATCH_IMPL if impl == BATCH_IMPL else env_cfg.impl
    return PlaygroundFuncEnv(env, loaded_impl), env_cfg


class EnvBundle(NamedTuple):
    """Container holding vectorized training and evaluation environment drivers."""

    env: Any
    test_env: Any
    env_cfg: Any


TRAINER_ENV_KEYS = frozenset({
    "parallel_envs",
    "test_episodes",
    "viewer",
    "player",
    "obs_size",
    "action_size",
    "obs_action_size",
})


def publish_env_shapes(cfg_env: DictConfig, obs_size: int, action_size: int) -> None:
    """Writes observation and action dimensions into the Hydra configuration block."""
    with open_dict(cfg_env):
        cfg_env.obs_size = int(obs_size)
        cfg_env.action_size = int(action_size)
        cfg_env.obs_action_size = int(obs_size) + int(action_size)


def build_playground_env(
    env_name: str,
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
    seed: int = 0,
    impl: str | None = "jax",
    naconmax: int | None = None,
    njmax: int | None = None,
    graph_mode: str | None = None,
    max_episode_steps: int | None = None,
    num_threads: int | None = None,
) -> EnvBundle:
    """Instantiates vectorized training and test drivers for `mujoco_playground` tasks."""
    func_env, env_cfg = load_playground_env(
        env_name, impl=impl, naconmax=naconmax, njmax=njmax,
        graph_mode=graph_mode,
    )
    if mode == "play":
        func_env.metadata["impl"] = "native"
    max_steps = int(max_episode_steps or env_cfg.episode_length)
    if impl == BATCH_IMPL:
        driver = functools.partial(BatchVectorEnv, num_threads=num_threads)
    else:
        driver = JaxVectorEnv
    return EnvBundle(
        env=driver(func_env, num_envs, max_episode_steps=max_steps),
        test_env=driver(func_env, test_episodes, max_episode_steps=max_steps),
        env_cfg=env_cfg,
    )


def plain(cfg: Any) -> Any:
    """A config node as plain containers, its interpolations resolved."""
    if OmegaConf.is_config(cfg):
        return OmegaConf.to_container(cfg, resolve=True)
    return cfg


def task_blocks(cfg_env: Any) -> list:
    """Each task's own env block, `task_defaults` merged in; `[cfg_env]` if single."""
    # Left a config, so `get` on a single block still resolves its interpolations.
    if "tasks" not in cfg_env:
        return [cfg_env]
    # Resolved while they are still part of the composed config their
    # interpolations point into, and copied out of it: Hydra locks it, so a
    # task block's own keys could not be merged into the defaults.
    defaults = plain(cfg_env.get("task_defaults", None)) or {}
    return [{**defaults, **(plain(block) or {})}
            for block in cfg_env["tasks"].values()]


def uses_batch(cfg_env: Any) -> bool:
    """Whether the block steps on `mujoco.batch`, read through a multi-task one."""
    blocks = task_blocks(cfg_env)
    return bool(blocks) and all(
        block.get("impl", None) == BATCH_IMPL for block in blocks
    )


def requested_impl(cfg_env: Any) -> str | None:
    """The physics `impl` the block asks for, read through a multi-task one's tasks."""
    impls = {block.get("impl", None) for block in task_blocks(cfg_env)}
    return impls.pop() if len(impls) == 1 else None


DEVICES = ("cpu", "gpu")


def resolve_placement(cfg: Any) -> tuple[str | None, str | None]:
    """Resolves target execution device and JAX platform constraints."""
    device = (cfg.get("runtime") or {}).get("device", None)
    if device is not None and str(device) not in DEVICES:
        raise ValueError(
            f"runtime.device={device!r} is not one of {DEVICES}."
        )

    if uses_batch(cfg.env):
        if device == "gpu":
            raise ValueError(
                "runtime.device=gpu cannot be used with impl=batch: its physics "
                "runs on CPU threads."
            )
        device = "cpu"

    platform = None if device in (None, "gpu") else device
    return device, platform


def build_env(
    cfg_env: DictConfig | dict[str, Any],
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
) -> EnvBundle:
    """Calls the builder an env block names, with the block's keys as kwargs.

    `_target_` is the builder's dotted path, so a builder may live outside
    roxie. Every other key is one of its keyword arguments, except the
    trainer's own (`TRAINER_ENV_KEYS`), which are dropped.
    """
    if OmegaConf.is_config(cfg_env):
        cfg_env = OmegaConf.to_container(
            cfg_env, resolve=True, throw_on_missing=True,
        )
    kwargs = {k: v for k, v in cfg_env.items() if k not in TRAINER_ENV_KEYS}
    if "_target_" not in kwargs:
        raise ValueError(f"env block {sorted(kwargs)} names no `_target_` builder")
    builder = get_method(kwargs.pop("_target_"))
    return builder(
        **kwargs, mode=mode, num_envs=num_envs, test_episodes=test_episodes,
    )