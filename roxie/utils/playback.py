"""Restores a trained checkpoint into a single-world environment and agent."""

from __future__ import annotations

import dataclasses
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
    is_multitask,
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
    if not is_multitask(cfg.env):
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


def _task_input(checkpoint_path: str, task: int, width: int):
    """`(observe, obs_size)` feeding one task's world to a multi-task policy.

    The trained layout is read off the checkpoint rather than the config: the
    statistics give the task count and the padded width, and the trainer
    records each task's own width beside its progress.
    """
    loaded = read_checkpoint(Path(checkpoint_path).resolve())
    num_tasks, padded = np.shape(loaded["trainstate_state"]["obs_stats"]["sum"])
    trained = int(loaded["metadata"]["task_obs_widths"][task])
    if width != trained:
        raise ValueError(
            f"the replay env gives this task {width} observation columns, but "
            f"it trained on {trained}."
        )
    pad = np.zeros(padded - width, dtype=np.float32)
    hot = np.eye(num_tasks, dtype=np.float32)[task]

    def observe(obs):
        return np.concatenate([np.asarray(obs, dtype=np.float32), pad, hot])

    return observe, int(padded + num_tasks)


def _single_task(obs):
    return obs


def resolve_single_world(cfg: Any) -> Any:
    """Rewrites a trained config onto a backend that can step one world."""
    if is_multitask(cfg.env):
        raise ValueError(
            "a multi-task config replays one task at a time; narrow it with "
            "`select_task` first."
        )
    if cfg.env.get("impl", None) in ("warp", BATCH_IMPL):
        cfg.env.impl = "jax"
    cfg.env.pop("num_threads", None)

    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    return cfg


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
    cfg, task = select_task(load_config(checkpoint_path, overrides), task_id)
    cfg = resolve_single_world(cfg)

    env, _, _env_cfg = build_env(cfg.env, mode="play", num_envs=1, test_episodes=1)
    if verbose:
        log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

    func_env = env.func_env
    obs_size = space_size(func_env.observation_space)
    action_size = space_size(func_env.action_space)
    observe = _single_task
    if task is not None:
        observe, obs_size = _task_input(checkpoint_path, task, obs_size)
    publish_env_shapes(cfg.env, obs_size, action_size)

    agent_args = {k: v for k, v in cfg.agent.items() if k.endswith("_config")}
    if "noise" in cfg:
        agent_args["noise_config"] = cfg.noise

    agent = get_class(cfg.agent._target_).load(
        path=checkpoint_path,
        env_obs_size=obs_size,
        env_act_size=action_size,
        **agent_args,
    )

    return Playback(
        cfg=cfg,
        env=env,
        func_env=func_env,
        agent=agent,
        obs_size=obs_size,
        action_size=action_size,
        observe=observe,
    )


def frame_dt(func_env: Any, model: Any) -> float:
    """Seconds of simulated time per environment step, for real-time playback."""
    substeps = int(
        getattr(func_env, "n_substeps", None)
        or getattr(func_env, "native_n_substeps", None)
        or 1
    )
    return float(
        getattr(func_env, "dt", None) or model.opt.timestep * substeps
    )


def make_stepper(cfg: Any, func_env: Any, *, verbose: bool = True):
    """Returns `(reset, step)` for one world, native MuJoCo if the env offers it.

    A native stepper runs the C++ physics directly and needs no compile; the
    fallback jits the functional env. The step key is fixed rather than
    threaded because playback is evaluation: consecutive replays of the same
    weights should be comparable.
    """
    player_path = cfg.env.get(
        "player", "roxie.utils.native_player.make_native_player"
    )
    player = get_method(player_path)(func_env) if player_path else None
    if player is not None:
        if verbose:
            print("Playback stepper: native MuJoCo (CPU)", flush=True)
        return player.reset, player.step

    transition = jax.jit(func_env.transition)
    reset = jax.jit(func_env.initial)

    def step(state, action):
        return transition(state, action, jax.random.PRNGKey(0))

    return reset, step
