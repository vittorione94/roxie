"""Restores a trained checkpoint into a single-world environment and agent."""

from __future__ import annotations

import dataclasses
import functools
import os
import sys
from pathlib import Path
from typing import Any, Callable

import jax
import numpy as np
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from roxie.environment.functional import space_size
from roxie.environment.loader import (
    BATCH_IMPL,
    build_env,
    TRAINER_ENV_KEYS,
    log_loaded_backend,
    publish_env_shapes,
    task_blocks,
)
from roxie.utils import hydra_searchpath
from roxie.utils.checkpoint import read_checkpoint


@dataclasses.dataclass(frozen=True)
class Playback:
    """A checkpoint restored into the pieces a replay loop needs.

    Attributes:
        cfg: the run's Hydra config, after the single-world rewrites.
        env: the environment bundle, built with `num_envs=1`.
        func_env: `env.func_env`, the unbatched functional environment.
        agent: the restored agent, ready for `select_action(..., evaluate=True)`.
        obs_size: width of the observation the agent was trained on.
        action_size: width of the action space.
        observe: maps one world's observation to what the policy was trained
            on; the identity for a single-task run, and for one task of a
            multi-task run, padding to the trained width plus its one-hot.
    """

    cfg: Any
    env: Any
    func_env: Any
    agent: Any
    obs_size: int
    action_size: int
    observe: Callable[[Any], Any]


def register_search_paths() -> None:
    """Puts the repo root and cwd on `sys.path`."""
    sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))
    sys.path.insert(0, os.getcwd())


def load_config(checkpoint_path: str, overrides: tuple[str, ...] = ()) -> Any:
    """Reads a run's config from a checkpoint dir and applies dotlist overrides."""
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def select_task(cfg: Any, task_id: str | None) -> tuple[Any, int | None]:
    """Narrows a multi-task run's config to the one task to replay.

    Args:
        cfg: the run's config.
        task_id: the task's name in the run's `env.tasks`; required for a
          multi-task run, refused for a single-task one.

    Returns:
        `(cfg, task)`: the config with `env` replaced by that task's own block,
        and the task's position in the run's one-hot order, or None for a
        single-task run.

    Raises:
        ValueError: on a missing or unknown task, or one passed to a
          single-task run.
    """
    if "tasks" not in cfg.env:
        if task_id is not None:
            raise ValueError(
                f"task_id={task_id!r} selects a task of a multi-task run, and "
                f"this run trained a single task."
            )
        return cfg, None
    names = list(cfg.env.tasks)
    if task_id not in names:
        raise ValueError(
            f"this run trained {len(names)} tasks, {names}; pass task_id as "
            f"one of them, not {task_id!r}."
        )
    task = names.index(task_id)
    # The trainer's keys stay: the agent's blocks interpolate
    # `${env.parallel_envs}` to rebuild the replay geometry it trained with.
    shared = {k: cfg.env[k] for k in TRAINER_ENV_KEYS if k in cfg.env}
    cfg.env = OmegaConf.create({**shared, **task_blocks(cfg.env)[task]})
    return cfg, task


@functools.lru_cache(maxsize=1)
def _task_layout(checkpoint_path: str) -> tuple[int, int, tuple[int, ...]]:
    """`(num_tasks, padded_width, task_widths)` a multi-task policy trained on.

    The layout is read off the checkpoint rather than the config: the trainer
    records each task's own width beside its progress, and the padding is to
    the widest.
    """
    loaded = read_checkpoint(Path(checkpoint_path).resolve(), only=("metadata",))
    widths = tuple(int(width) for width in loaded["metadata"]["task_obs_widths"])
    return len(widths), max(widths), widths


def _task_input(checkpoint_path: str, task: int, width: int):
    """`(observe, obs_size)` feeding one task's world to a multi-task policy."""
    num_tasks, padded, widths = _task_layout(checkpoint_path)
    trained = widths[task]
    if width != trained:
        raise ValueError(
            f"the replay env gives this task {width} observation columns, but "
            f"it trained on {trained}."
        )
    pad = np.zeros(padded - width, dtype=np.float32)
    hot = np.eye(num_tasks, dtype=np.float32)[task]

    def observe(obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32), pad, hot])

    return observe, padded + num_tasks


def _single_task(obs):
    return obs


def resolve_single_world(cfg: Any) -> Any:
    """Rewrites a trained config onto a backend that can step one world."""
    if cfg.get("vision"):
        # The policy reads pixels only warp's CUDA renderer draws, so it replays
        # where it trained. JAX's default 75% preallocation leaves warp too
        # little to allocate the world in.
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")
        jax.config.update("jax_platforms", "cuda")
    elif cfg.env.get("impl", None) in ("warp", BATCH_IMPL):
        cfg.env.impl = "jax"
    cfg.env.pop("num_threads", None)

    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    return cfg


def _load_world(
    checkpoint_path: str,
    overrides: tuple[str, ...],
    task_id: str | None,
    verbose: bool,
) -> dict[str, Any]:
    """Builds the `Playback` fields of one task's world, all but the agent."""
    cfg, task = select_task(load_config(checkpoint_path, overrides), task_id)
    cfg = resolve_single_world(cfg)

    env, _, _env_cfg = build_env(cfg.env, mode="play", num_envs=1, test_episodes=1)
    if verbose:
        log_loaded_backend(env)

    func_env = env.func_env
    obs_size = space_size(func_env.observation_space)
    action_size = space_size(func_env.action_space)
    observe = _single_task
    if task is not None:
        observe, obs_size = _task_input(checkpoint_path, task, obs_size)
    publish_env_shapes(
        cfg.env, obs_size, action_size,
        pixel_shape=getattr(func_env, "pixel_shape", None),
    )

    return {
        "cfg": cfg,
        "env": env,
        "func_env": func_env,
        "obs_size": obs_size,
        "action_size": action_size,
        "observe": observe,
    }


def _load_agent(checkpoint_path: str, world: dict[str, Any]) -> Any:
    """Restores the checkpoint's agent for a world's observation and action sizes."""
    cfg = world["cfg"]
    agent_args = {k: v for k, v in cfg.agent.items() if k.endswith("_config")}
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    return get_class(cfg.agent._target_).load(
        path=checkpoint_path,
        env_obs_size=world["obs_size"],
        env_act_size=world["action_size"],
        **agent_args,
    )


def load_policy(
    checkpoint_path: str,
    overrides: tuple[str, ...] = (),
    *,
    task_id: str | None = None,
    verbose: bool = True,
) -> Playback:
    """Restores the agent and the single-world env its checkpoint was trained on.

    A multi-task run replays the one task `task_id` names, on that task's own
    env.
    """
    register_search_paths()
    world = _load_world(checkpoint_path, overrides, task_id, verbose)
    return Playback(agent=_load_agent(checkpoint_path, world), **world)


def load_tasks(
    checkpoint_path: str,
    overrides: tuple[str, ...] = (),
    *,
    verbose: bool = True,
) -> dict[str | None, Playback]:
    """Restores one agent and the world of every task the run can replay.

    Args:
        checkpoint_path: the checkpoint's step directory.
        overrides: dotlist overrides of the run's saved config.
        verbose: print the backend the first world resolved to.

    Returns:
        `{task: Playback}` in the run's order, all holding the same agent. A
        single-task run's only key is None.

    Raises:
        ValueError: if a task's replay env does not give the observation width
          it trained on.
    """
    register_search_paths()
    cfg = load_config(checkpoint_path, overrides)
    if "tasks" not in cfg.env:
        return {None: load_policy(checkpoint_path, overrides, verbose=verbose)}

    worlds = {
        name: _load_world(checkpoint_path, overrides, name, verbose and not index)
        for index, name in enumerate(cfg.env.tasks)
    }
    agent = _load_agent(checkpoint_path, next(iter(worlds.values())))
    return {
        name: Playback(agent=agent, **world) for name, world in worlds.items()
    }


def frame_dt(func_env: Any, model: Any) -> float:
    """Seconds of simulated time per environment step, for real-time playback."""
    substeps = int(getattr(func_env, "n_substeps", None) or 1)
    return float(
        getattr(func_env, "dt", None) or model.opt.timestep * substeps
    )


def make_stepper(cfg: Any, func_env: Any, *, verbose: bool = True):
    """Returns `(reset, step)` for one world, with its physics in C MuJoCo.

    A functional env with no MuJoCo task behind it has no physics to move, and
    is jitted whole, as is a vision run, whose pixels only warp renders. The
    step key is fixed rather than threaded because playback is evaluation:
    consecutive replays of the same weights should be comparable.
    """
    player = None
    if not cfg.get("vision"):
        player_path = cfg.env.get(
            "player", "roxie.utils.native_player.make_native_player"
        )
        player = get_method(player_path)(func_env)
    if player is not None:
        if verbose:
            print("Playback stepper: native MuJoCo (CPU)", flush=True)
        return player.reset, player.step

    key = jax.random.PRNGKey(0)

    # The env state keeps a dict observation's groups apart; `obs` is swapped
    # for the flat vector the policy was trained on.
    @jax.jit
    def reset(reset_key):
        state = func_env.initial(reset_key)
        return state.replace(obs=func_env.observation(state, key))

    @jax.jit
    def step(state, action):
        state = func_env.transition(state, action, key)
        return state.replace(obs=func_env.observation(state, key))

    return reset, step
