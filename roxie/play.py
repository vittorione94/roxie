"""Single-world interactive policy playback in MuJoCo's passive viewer."""

import os
import sys
import sysconfig

# Force CPU execution before JAX import for deterministic single-world playback.
os.environ["JAX_PLATFORMS"] = "cpu"
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")


def _exec_under_mjpython() -> None:
    """Replaces this process with the same command run under `mjpython`.

    On macOS `launch_passive` has to drive Cocoa from the thread holding the
    com.apple.main-thread queue, which only mujoco's `mjpython` trampoline
    arranges; under a plain interpreter it raises, and only after the whole
    checkpoint restore has already run.
    """
    mjpython = os.path.join(os.path.dirname(sys.executable), "mjpython")
    if not os.path.exists(mjpython):
        sys.exit(
            "The MuJoCo viewer needs mujoco's `mjpython` on macOS, and there "
            f"is none next to {sys.executable}."
        )

    env = dict(os.environ)
    # The trampoline resolves libpython's @executable_path against the venv's
    # bin/, which for a uv-managed interpreter holds a symlink and no dylib, so
    # dyld finds nothing. Point it at the real interpreter's lib dir first.
    libdir = sysconfig.get_config_var("LIBDIR")
    if libdir:
        fallback = env.get("DYLD_FALLBACK_LIBRARY_PATH", "/usr/local/lib:/usr/lib")
        env["DYLD_FALLBACK_LIBRARY_PATH"] = f"{libdir}:{fallback}"

    entry = ["-m", __spec__.name] if __spec__ else [os.path.abspath(__file__)]
    print(f"macOS viewer: re-executing under {mjpython}", flush=True)
    os.execve(mjpython, [mjpython, *entry, *sys.argv[1:]], env)


# MJPYTHON_BIN is set by the trampoline and inherited, so this fires at most once.
if (
    __name__ == "__main__"
    and sys.platform == "darwin"
    and "MJPYTHON_BIN" not in os.environ
):
    _exec_under_mjpython()

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Callable

import click
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from hydra.utils import get_method

from roxie.utils import playback

# What is left to compile per launch is the agent's parameter init, 0.9 s of a
# 1.5 s restore without this. Only `play.py` turns it on, for its own programs.
_COMPILE_CACHE = Path.home() / ".cache" / "roxie" / "jax_compile"

# GLFW key codes. Up and down are the arrows `simulate` leaves unbound.
_KEY_SPACE = 32
_KEY_BACKSPACE = 259
_KEY_DELETE = 261
_KEY_DOWN = 264
_KEY_UP = 265

_CONTROLS_HELP = """Viewer controls:
  space            pause / resume
  delete           reset the environment
  double-click     select a body
  ctrl + drag      running: push selected body (right: translate, left: rotate)
                   paused: drag body pose directly"""

_TASKS_HELP = """
  up / down        previous / next task"""

# mujoco_warp renders colour, depth, or both stacked colour-first.
_PIXEL_PANELS = {
    1: (("depth", 0),),
    3: (("RGB", slice(0, 3)),),
    4: (("RGB", slice(0, 3)), ("depth", 3)),
}


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


class _Controls:
    """What the viewer's keys asked for, set from its thread and read from ours.

    Attributes:
        tasks: the names of the tasks the replay can switch between.
        task: the index of the chosen task in `tasks`.
        paused: whether the replay is paused.
        task_changed: whether `task` changed since the window opened on it.
    """

    def __init__(self, tasks: tuple[str | None, ...], task: int):
        self.tasks = tasks
        self.task = task
        self.paused = False
        self.task_changed = False
        self._reset_requested = False

    def key_callback(self, keycode: int) -> None:
        """Handles a GLFW key code from the viewer's thread."""
        if keycode == _KEY_SPACE:
            self.paused = not self.paused
            print("[viewer] paused" if self.paused else "[viewer] running", flush=True)
        elif keycode in (_KEY_BACKSPACE, _KEY_DELETE):
            self._reset_requested = True
        elif keycode in (_KEY_UP, _KEY_DOWN) and len(self.tasks) > 1:
            offset = 1 if keycode == _KEY_DOWN else -1
            self.task = (self.task + offset) % len(self.tasks)
            self.task_changed = True

    def take_reset_request(self) -> bool:
        """Whether a reset was asked for since the last call."""
        requested, self._reset_requested = self._reset_requested, False
        return requested


class _PixelWindow:
    """A window showing the image at the front of each observation."""

    def __init__(self, pixel_shape: tuple[int, ...]):
        self._shape = tuple(pixel_shape)
        self._size = math.prod(self._shape)
        self._panels = _PIXEL_PANELS.get(self._shape[-1])
        if self._panels is None:
            raise click.UsageError(
                f"--vision shows RGB and depth images; this run's pixels have "
                f"{self._shape[-1]} channels"
            )
        plt.ion()
        self._fig, axes = plt.subplots(
            1, len(self._panels), figsize=(4 * len(self._panels), 4.2),
            squeeze=False,
        )
        self._fig.canvas.manager.set_window_title("What the policy sees")
        self._images = []
        for ax, (name, channels) in zip(axes[0], self._panels):
            rgb = isinstance(channels, slice)
            blank = np.zeros(self._shape[:2] + ((3,) if rgb else ()))
            image = ax.imshow(
                blank, vmin=0.0, vmax=1.0, interpolation="nearest",
                cmap=None if rgb else "gray_r",
            )
            if name == "depth":
                self._fig.colorbar(image, ax=ax, fraction=0.046, label="near → far")
            ax.set_title(name)
            ax.axis("off")
            self._images.append(image)
        self._fig.tight_layout()
        plt.show(block=False)

    def show(self, obs) -> None:
        """Draws the image part of a flat observation."""
        if not plt.fignum_exists(self._fig.number):
            return
        pixels = np.asarray(obs[: self._size]).reshape(self._shape)
        for image, (_name, channels) in zip(self._images, self._panels):
            frame = pixels[..., channels]
            low, high = frame.min(), frame.max()
            # Some tasks centre their frames (playground's Cartpole is `gray -
            # 0.5`); clipping those to [0, 1] would black out half the image.
            if low < 0.0 or high > 1.0:
                frame = (frame - low) / max(high - low, 1e-6)
            image.set_data(frame)
        self._fig.canvas.draw_idle()
        self.poll()

    def poll(self) -> None:
        """Keeps the window responsive between draws."""
        if plt.fignum_exists(self._fig.number):
            self._fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self._fig)


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
    """Runs one step of a world, so switching to it finds it all traced."""
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


def _aim_camera(
    camera: mujoco.MjvCamera, model: mujoco.MjModel, previous: mujoco.MjvCamera | None,
) -> None:
    """Carries the last window's view over, or follows the root body at first."""
    if previous is not None:
        for name in ("type", "trackbodyid", "fixedcamid", "lookat", "distance",
                     "azimuth", "elevation"):
            setattr(camera, name, getattr(previous, name))
    elif model.nbody > 1:
        # A locomotion task leaves a fixed view within seconds.
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = 1
    # A tracked body or a fixed camera the model lacks aborts the whole process.
    if camera.trackbodyid >= model.nbody or camera.fixedcamid >= model.ncam:
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", type=str, help="Path to the checkpoint directory.")
@click.option(
    "--task-id", type=str, default=None,
    help="For a multi-task run, the task to start on; the first by default.",
)
@click.option(
    "--vision", is_flag=True,
    help="Show the policy's camera images and the camera's frustum.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(
    checkpoint_path: str,
    task_id: str | None,
    vision: bool,
    overrides: tuple[str, ...],
) -> None:
    """Replays a trained agent checkpoint in an interactive MuJoCo viewer."""
    _enable_compile_cache()
    plays = playback.load_tasks(checkpoint_path, overrides)
    if task_id is not None and task_id not in plays:
        raise click.BadParameter(
            f"{task_id!r} is not a task this run can replay: {list(plays)}.",
            param_hint="--task-id",
        )
    agent = next(iter(plays.values())).agent

    pixel_window = None
    if vision:
        pixel_shape = next(iter(plays.values())).func_env.pixel_shape
        if pixel_shape is None:
            raise click.UsageError("--vision needs a run whose policy reads pixels")
        pixel_window = _PixelWindow(pixel_shape)
    worlds = {
        name: _build_world(play, verbose=index == 0)
        for index, (name, play) in enumerate(plays.items())
    }

    key = jax.random.PRNGKey(seed=0)
    for world in worlds.values():
        _warm_up(world, agent, key)

    tasks = tuple(worlds)
    controls = _Controls(tasks, tasks.index(task_id) if task_id is not None else 0)
    print(_CONTROLS_HELP + (_TASKS_HELP if len(tasks) > 1 else ""), flush=True)

    camera = None
    while True:
        world = worlds[tasks[controls.task]]
        if len(tasks) > 1:
            print(f"[viewer] task {tasks[controls.task]}", flush=True)
        # The passive viewer is bound to the model it opens with, so each task
        # gets a window of its own and a switch closes one and opens the next.
        with mujoco.viewer.launch_passive(
            world.model, world.data, key_callback=controls.key_callback,
        ) as viewer:
            with viewer.lock():
                _aim_camera(viewer.cam, world.model, camera)
                if vision:
                    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1
            if vision and not (world.model.cam_resolution > 1).any():
                print(
                    "--vision: no camera declares a resolution, so MuJoCo has "
                    "no frustum to draw",
                    flush=True,
                )
            key = _replay(world, agent, key, viewer, controls, pixel_window)
            camera = mujoco.MjvCamera()
            _aim_camera(camera, world.model, viewer.cam)
        if not controls.task_changed:
            break
        controls.task_changed = False

    if pixel_window is not None:
        pixel_window.close()


def _replay(
    world: _World,
    agent: Any,
    key: Any,
    viewer: Any,
    controls: _Controls,
    pixel_window: _PixelWindow | None = None,
) -> Any:
    """Plays one task until its window closes or another task is chosen.

    Returns:
        The key to carry on with.
    """

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
        if pixel_window is not None:
            pixel_window.show(state.obs)

    state = None
    score = 0.0
    actions = []
    perturbing = False
    reset_requested = True
    controls.paused = False

    while viewer.is_running():
        step_start = time.time()

        if controls.task_changed:
            break

        reset_requested |= controls.take_reset_request()
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

        if controls.paused:
            if pert_active:
                mujoco.mjv_applyPerturbPose(world.model, world.data, pert, 1)
                mujoco.mj_forward(world.model, world.data)
                state = _write_state_data(state, qpos=world.data.qpos[:world.env_nq])
            viewer.sync()
            if pixel_window is not None:
                pixel_window.poll()
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
        action, _noise, _extras = agent.select_action(obs_b, key, evaluate=True)

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
            reset_requested = True

        viewer.sync()

        time_until_next_step = world.step_dt - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)

    return key


if __name__ == "__main__":
    main()
