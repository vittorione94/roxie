"""Single-world interactive policy playback in a MuJoCo window."""

import os

# Force CPU execution before JAX import for deterministic single-world playback.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import dataclasses
import time
from pathlib import Path
from typing import Any, Callable

import click
import jax
import jax.numpy as jnp
import mujoco

from hydra.utils import get_method

from roxie.utils import playback
from roxie.utils.viewer import Viewer

# What is left to compile per launch is the agent's parameter init, 0.9 s of a
# 1.5 s restore without this. Only `play.py` turns it on, for its own programs.
_COMPILE_CACHE = Path.home() / ".cache" / "roxie" / "jax_compile"

_CONTROLS_HELP = """Viewer controls:
  space            pause / resume
  delete           reset the environment
  tab              hide / show the panel
  esc              free the camera
  left drag        rotate the camera (shift: around the vertical)
  right drag       move the camera (shift: horizontally)
  scroll           zoom
  double-click     left: select a body, right: look at a point
  ctrl + right double-click
                   track a body with the camera
  ctrl + drag      running: push selected body (right: translate, left: rotate)
                   paused: drag body pose directly"""


@dataclasses.dataclass(frozen=True)
class _World:
    """One task as the viewer replays it.

    Attributes:
        observe: maps the world's observation to what the policy trained on.
        ghost: the env's reference-pose overlay, or None.
        model: the model the viewer draws, which carries the ghost if any.
        data: the viewer's buffer for `model`.
        step_dt: seconds of simulated time per environment step.
        reset: starts an episode from a key.
        step: advances the env's state by one action.
        env_nq: `nq` of the env's own model, which excludes the ghost.
        env_nbody: `nbody` of the env's own model, which excludes the ghost.
    """

    observe: Callable[[Any], Any]
    ghost: Any
    model: mujoco.MjModel
    data: mujoco.MjData
    step_dt: float
    reset: Callable[[Any], Any]
    step: Callable[[Any, Any], Any]
    env_nq: int
    env_nbody: int


def _build_world(play: playback.Playback, *, verbose: bool = True) -> _World:
    """Builds the stepper and the viewer's buffers for one restored task."""
    func_env = play.func_env
    viewer_path = play.cfg.env.get("viewer", None)
    ghost = get_method(viewer_path)(func_env) if viewer_path else None
    model = ghost.model if ghost is not None else func_env.mj_model
    reset, step = playback.make_stepper(play.cfg, func_env, verbose=verbose)
    env_model = getattr(func_env, "mj_model", None) or model
    return _World(
        observe=play.observe,
        ghost=ghost,
        model=model,
        data=mujoco.MjData(model),
        step_dt=playback.frame_dt(func_env, model),
        reset=reset,
        step=step,
        env_nq=int(env_model.nq),
        env_nbody=int(env_model.nbody),
    )


def _warm_up(world: _World, agent: Any, key: Any) -> None:
    """Runs one step of a world, so its first click finds it all traced."""
    state = world.reset(key)
    obs = jnp.expand_dims(world.observe(state.obs), axis=0)
    action, _noise, _extras = agent.select_action(obs, key, evaluate=True)
    jax.block_until_ready(world.step(state, action[0]).obs)


def _enable_compile_cache() -> None:
    _COMPILE_CACHE.mkdir(parents=True, exist_ok=True)
    jax.config.update("jax_compilation_cache_dir", str(_COMPILE_CACHE))
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)


def _write_state_data(state, **fields):
    """Updates physics buffers on native MjData or MJX Data state instances."""
    data = state.data
    if isinstance(data, mujoco.MjData):
        for name, value in fields.items():
            getattr(data, name)[:] = value
        return state
    return state.replace(
        data=data.replace(**{k: jnp.asarray(v) for k, v in fields.items()})
    )


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint directory.")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(checkpoint_path: str, overrides: tuple[str, ...]) -> None:
    """Replays a trained agent checkpoint in an interactive MuJoCo window."""
    _enable_compile_cache()
    plays = playback.load_tasks(checkpoint_path, overrides)
    agent = next(iter(plays.values())).agent
    worlds = {
        name: _build_world(play, verbose=index == 0)
        for index, (name, play) in enumerate(plays.items())
    }

    key = jax.random.PRNGKey(seed=0)
    for world in worlds.values():
        _warm_up(world, agent, key)

    _replay(worlds, agent, key)


def _replay(worlds: dict[str | None, _World], agent: Any, key: Any) -> None:
    """Runs the window until it closes, resetting into whichever task is chosen."""
    current = next(iter(worlds))
    world = worlds[current]

    print(_CONTROLS_HELP, flush=True)

    with Viewer(world.model, world.data, tasks=tuple(worlds)) as viewer:
        for other in worlds.values():
            viewer.prepare(other.model)

        def show(state):
            """Synchronizes environment state into the visualizer MjData buffer."""
            ghost, data = world.ghost, world.data
            if ghost is not None:
                info = state.info
                abs_idx = int(info["clip_start"]) + int(info["phase_idx"])
                data.qpos[:ghost.nq] = state.data.qpos
                data.qvel[:ghost.nv] = state.data.qvel
                data.qpos[ghost.nq:] = ghost.ref_qpos[abs_idx]
                data.qvel[ghost.nv:] = ghost.ref_qvel[abs_idx]
            else:
                data.qpos = state.data.qpos
                data.qvel = state.data.qvel
            mujoco.mj_forward(world.model, data)

        state = None
        score = 0.0
        actions = []
        perturbing = False
        reset_requested = True

        while viewer.is_running():
            step_start = time.time()

            reset_requested |= viewer.take_reset_request()
            clicked = viewer.take_clicked_task()
            if clicked is not None:
                print(f"[viewer] task {clicked}", flush=True)
                if worlds[clicked] is not world:
                    world = worlds[clicked]
                    viewer.show(world.model, world.data)
                viewer.resume()
                reset_requested = True

            if reset_requested:
                reset_requested = False
                key, reset_key = jax.random.split(key)
                state = world.reset(reset_key)
                mujoco.mj_resetData(world.model, world.data)
                show(state)
                score = 0.0
                actions = []
                perturbing = False
                print("[viewer] reset", flush=True)

            pert = viewer.perturb
            pert_active = bool(pert.active) and int(pert.select) > 0

            if viewer.paused:
                if pert_active:
                    mujoco.mjv_applyPerturbPose(world.model, world.data, pert, 1)
                    mujoco.mj_forward(world.model, world.data)
                    state = _write_state_data(
                        state, qpos=world.data.qpos[:world.env_nq]
                    )
                viewer.sync()
                time.sleep(max(0.0, world.step_dt - (time.time() - step_start)))
                continue

            if pert_active or perturbing:
                world.data.xfrc_applied[:] = 0.0
                if pert_active:
                    mujoco.mjv_applyPerturbForce(world.model, world.data, pert)
                state = _write_state_data(
                    state, xfrc_applied=world.data.xfrc_applied[:world.env_nbody].copy()
                )
                perturbing = pert_active

            obs_b = jnp.expand_dims(world.observe(state.obs), axis=0)
            action, _noise, _extras = agent.select_action(
                obs_b, key, evaluate=True,
            )

            state = world.step(state, action[0])

            world.data.ctrl = action
            show(state)

            score += state.reward
            actions.append(action)

            metrics = state.metrics
            components = " ".join(
                f"{k.split('/', 1)[1]}={float(v):+.3f}"
                for k, v in metrics.items()
                if k.startswith("reward/")
            )
            print(f"reward={float(state.reward):+.3f} {components}")

            if state.done:
                print(f"Total score: {score}")
                print(f"Actions mean: {jnp.mean(jnp.array(actions)):.2f}")
                print(f"Actions std: {jnp.std(jnp.array(actions)):.2f}")
                key, reset_key = jax.random.split(reset_key)
                state = world.reset(reset_key)
                mujoco.mj_resetData(world.model, world.data)
                show(state)
                score = 0.0
                actions = []
                perturbing = False

            viewer.sync()

            time_until_next_step = world.step_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()