"""Restores a trained checkpoint into a single-world environment and agent."""

from __future__ import annotations

import dataclasses
import inspect
import os
import sys
from typing import Any

import jax
from hydra.utils import get_class, get_method
from omegaconf import OmegaConf

from roxie.environment import suites
from roxie.environment.functional import space_size
from roxie.environment.loader import (
    DEFAULT_BUILDER,
    TRAINER_ENV_KEYS,
    build_env,
    build_playground_env,
    log_loaded_backend,
    publish_env_shapes,
    uses_envpool,
)
from roxie.utils import hydra_searchpath

# Keys build_playground_env accepts, so the EnvPool rewrite below can drop the
# EnvPool-only ones (task_id, pool sizing) instead of passing them through.
_PLAYGROUND_ENV_KEYS = (
    set(inspect.signature(build_playground_env).parameters)
    | set(TRAINER_ENV_KEYS)
    | {"_target_"}
)


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
    """

    cfg: Any
    env: Any
    func_env: Any
    agent: Any
    obs_size: int
    action_size: int


def register_search_paths() -> None:
    """Puts the repo root and cwd on `sys.path` and registers config resolvers."""
    sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))
    sys.path.insert(0, os.getcwd())
    suites.register_resolvers()


def load_config(checkpoint_path: str, overrides: tuple[str, ...] = ()) -> Any:
    """Reads a run's config from a checkpoint dir and applies dotlist overrides."""
    cfg_path = os.path.join(checkpoint_path, "../../.hydra/config.yaml")
    cfg = OmegaConf.load(cfg_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def resolve_single_world(cfg: Any, *, verbose: bool = True) -> Any:
    """Rewrites a trained config onto a backend that can step one world."""
    if cfg.env.get("impl", None) == "warp":
        cfg.env.impl = "jax"

    if uses_envpool(cfg.env):
        task = suites.playground_task(cfg.env.task_id)
        if verbose:
            print(
                f"EnvPool has no single-world stepper; replaying "
                f"{cfg.env.task_id} on playground twin {task!r}.",
                flush=True,
            )
        cfg.env["_target_"] = DEFAULT_BUILDER
        cfg.env.env_name = task
        cfg.env.impl = "jax"
        cfg.env.pop("task_id", None)
        for key in list(cfg.env.keys()):
            if key not in _PLAYGROUND_ENV_KEYS:
                cfg.env.pop(key, None)

    matmul_precision = (cfg.get("runtime") or {}).get("matmul_precision", None)
    if matmul_precision:
        jax.config.update("jax_default_matmul_precision", matmul_precision)

    return cfg


def load_policy(
    checkpoint_path: str,
    overrides: tuple[str, ...] = (),
    *,
    verbose: bool = True,
) -> Playback:
    """Restores the agent and the single-world env its checkpoint was trained on."""
    register_search_paths()
    cfg = resolve_single_world(
        load_config(checkpoint_path, overrides), verbose=verbose
    )

    env, _, _env_cfg = build_env(cfg.env, mode="play", num_envs=1, test_episodes=1)
    if verbose:
        log_loaded_backend(env, requested_impl=cfg.env.get("impl", "jax"))

    func_env = env.func_env
    obs_size = space_size(func_env.observation_space)
    action_size = space_size(func_env.action_space)
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
