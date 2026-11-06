"""Environment loading, MuJoCo adapters, and vector environment builders."""

import functools
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import mujoco
from hydra.utils import get_method, instantiate
from omegaconf import DictConfig, OmegaConf, open_dict
from mujoco_playground import registry

from roxie.environment import functional, suites
from roxie.environment.functional import FuncEnv
from roxie.environment.vector import BatchVectorEnv, JaxVectorEnv

# playground 0.2.0's reacher.py still calls this; MuJoCo renamed it to `body`.
if not hasattr(mujoco.MjSpec, "find_body"):
    mujoco.MjSpec.find_body = mujoco.MjSpec.body


class MuJoCoFuncEnv(FuncEnv):
    """Abstract functional environment adapter for MuJoCo physics state."""

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


class PlaygroundFuncEnv(MuJoCoFuncEnv):
    """Functional environment wrapper for `mujoco_playground` environments."""

    def __init__(self, env: Any, impl: str | None = None):
        self.env = env
        self.observation_space = functional.unbounded_box(env.observation_size)
        ctrl_range = env.mj_model.actuator_ctrlrange
        self.action_space = functional.box(ctrl_range[:, 0], ctrl_range[:, 1])
        self.metadata = {"jax": True, "impl": impl or _read_impl(env)}

    def initial(self, rng, params=None):
        return self.env.reset(rng)

    def transition(self, state, action, rng, params=None):
        return self.env.step(state, action)

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


def _set_graph_mode(env: Any, graph_mode: str) -> None:
    """Re-puts a playground environment's MJX model using a specified CUDA graph mode."""
    from mujoco import mjx
    from mujoco.mjx.warp import types as mjxw_types

    if not hasattr(env, "_mjx_model"):
        raise RuntimeError(
            f"{type(env).__name__} has no `_mjx_model` to re-put; mujoco_playground "
            "changed shape."
        )
    env._mjx_model = mjx.put_model(
        env.mj_model,
        impl="warp",
        graph_mode=getattr(mjxw_types.GraphMode, graph_mode),
    )


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
    if impl is not None and "impl" in env_cfg:
        env_cfg.impl = "jax" if impl == BATCH_IMPL else impl
    if naconmax is not None and "naconmax" in env_cfg:
        env_cfg.naconmax = naconmax
    if njmax is not None and "njmax" in env_cfg:
        env_cfg.njmax = njmax

    env = registry.load(env_name, config=env_cfg)
    if str(env_cfg.get("impl", "")) == "warp":
        _set_graph_mode(env, graph_mode or DEFAULT_WARP_GRAPH_MODE)
    return PlaygroundFuncEnv(env, impl=BATCH_IMPL if impl == BATCH_IMPL else None), env_cfg


class EnvBundle(NamedTuple):
    """Container holding vectorized training and evaluation environment drivers."""

    env: Any
    test_env: Any
    env_cfg: Any


DEFAULT_MAX_EPISODE_STEPS = 1000

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
    if impl == BATCH_IMPL and env_name not in suites.DMC_TASKS:
        raise ValueError(
            f"impl=batch supports the dm_control suite tasks, not {env_name!r}."
        )
    func_env, env_cfg = load_playground_env(
        env_name, impl=impl, naconmax=naconmax, njmax=njmax,
        graph_mode=graph_mode,
    )
    max_steps = int(
        max_episode_steps
        or env_cfg.get("episode_length", None)
        or DEFAULT_MAX_EPISODE_STEPS
    )
    if impl == BATCH_IMPL:
        driver = functools.partial(BatchVectorEnv, num_threads=num_threads)
    else:
        driver = JaxVectorEnv
    return EnvBundle(
        env=driver(func_env, num_envs, max_episode_steps=max_steps),
        test_env=driver(func_env, test_episodes, max_episode_steps=max_steps),
        env_cfg=env_cfg,
    )


DEFAULT_BUILDER = "roxie.environment.loader.build_playground_env"


def builder_target(cfg_env: Any) -> str:
    """Returns the dotted import path of the target environment builder."""
    return cfg_env.get("_target_", None) or DEFAULT_BUILDER


def _builds_with(cfg_env: Any, builder) -> bool:
    try:
        return get_method(builder_target(cfg_env)) is builder
    except Exception:
        return False


def is_multitask(cfg_env: Any) -> bool:
    """Whether the env block composes several tasks into one batch."""
    from roxie.environment import multitask  # it builds through this module

    return _builds_with(cfg_env, multitask.build_multitask_env)


def task_blocks(cfg_env: Any) -> list:
    """Each task's own env block, `task_defaults` merged in; `[cfg_env]` if single."""
    if not is_multitask(cfg_env):
        return [cfg_env]
    def plain(node):
        # Unresolved, and copied out of the composed config: Hydra locks it,
        # so a task block's own keys could not be merged into the defaults.
        if OmegaConf.is_config(node):
            return OmegaConf.to_container(node, resolve=False)
        return dict(node or {})

    defaults = plain(cfg_env.get("task_defaults", None))
    return [
        {**defaults, **plain(block)}
        for block in (cfg_env.get("tasks", None) or {}).values()
    ]


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


def _parse_cpu_list(text: str) -> list[int]:
    """Parses a CPU range string into individual core indices."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _l3_domains(available: list[int]) -> list[list[int]]:
    """Groups available CPUs by their shared L3 cache domain."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in available:
        path = f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
        try:
            with open(path) as fh:
                key = tuple(sorted(_parse_cpu_list(fh.read())))
        except (OSError, ValueError):
            return [list(available)]
        groups.setdefault(key, []).append(cpu)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def select_cpu_cores(available: list[int], cores: int) -> list[int]:
    """Selects core indices packed to minimize L3 cache domain boundaries."""
    picked: list[int] = []
    for group in _l3_domains(available):
        if len(picked) >= cores:
            break
        picked.extend(group[: cores - len(picked)])
    return sorted(picked)


def pin_cpu_cores(cores: Any, platform: str | None) -> int | None:
    """Pins CPU affinity for the current process to a subset of available cores."""
    if not cores or str(platform) != "cpu" or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    cores = int(cores)
    if cores >= len(available):
        return None
    os.sched_setaffinity(0, set(select_cpu_cores(available, cores)))
    return cores


def build_env(
    cfg_env: Any,
    *,
    mode: str = "train",
    num_envs: int = 1,
    test_episodes: int = 1,
) -> EnvBundle:
    """Instantiates an environment bundle from a Hydra configuration block."""
    if isinstance(cfg_env, DictConfig):
        resolved = OmegaConf.to_container(
            cfg_env, resolve=True, throw_on_missing=True,
        )
    else:
        resolved = dict(cfg_env)
    node = {k: v for k, v in resolved.items() if k not in TRAINER_ENV_KEYS}
    node["_target_"] = builder_target(cfg_env)
    # Not recursive, as `build_agent` is not: a builder receives its nested
    # blocks as data and builds them itself, which is how the multi-task
    # builder hands each task's block to that task's own builder.
    return instantiate(
        node, _recursive_=False, mode=mode, num_envs=num_envs,
        test_episodes=test_episodes,
    )