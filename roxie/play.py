"""Single-world interactive policy playback and MuJoCo viewer launcher."""

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

import math
import time

import click
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np

from hydra.utils import get_method

from roxie.utils import playback

_KEY_SPACE = 32
_KEY_BACKSPACE = 259
_KEY_DELETE = 261

_CONTROLS_HELP = """Viewer controls:
  space            pause / resume
  delete           reset the environment
  double-click     select a body
  ctrl + drag      running: push selected body (right: translate, left: rotate)
                   paused: drag body pose directly"""

# mujoco_warp renders colour, depth, or both stacked colour-first.
_PIXEL_PANELS = {
    1: (("depth", 0),),
    3: (("RGB", slice(0, 3)),),
    4: (("RGB", slice(0, 3)), ("depth", 3)),
}


class _ViewerControls:
    """Thread-safe state tracker for viewer UI callbacks."""

    def __init__(self):
        self.paused = False
        self.reset_requested = False

    def key_callback(self, keycode: int) -> None:
        """Processes GLFW keycode events from the passive viewer thread."""
        if keycode == _KEY_SPACE:
            self.paused = not self.paused
            print("[viewer] paused" if self.paused else "[viewer] running", flush=True)
        elif keycode in (_KEY_BACKSPACE, _KEY_DELETE):
            self.reset_requested = True


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
            image.set_data(np.clip(pixels[..., channels], 0.0, 1.0))
        self._fig.canvas.draw_idle()
        self.poll()

    def poll(self) -> None:
        """Keeps the window responsive between draws."""
        if plt.fignum_exists(self._fig.number):
            self._fig.canvas.flush_events()

    def close(self) -> None:
        plt.close(self._fig)


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
@click.option(
    "--vision", is_flag=True,
    help="Show the policy's camera images and the camera's frustum.",
)
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(checkpoint_path: str, vision: bool, overrides: tuple[str, ...]) -> None:
    """Replays a trained agent checkpoint in an interactive MuJoCo viewer."""
    play = playback.load_policy(checkpoint_path, overrides)
    cfg, func_env, agent = play.cfg, play.func_env, play.agent

    pixel_window = None
    if vision:
        if func_env.pixel_shape is None:
            raise click.UsageError("--vision needs a run whose policy reads pixels")
        pixel_window = _PixelWindow(func_env.pixel_shape)

    key = jax.random.PRNGKey(seed=0)

    viewer_path = cfg.env.get("viewer", None)
    ghost = get_method(viewer_path)(func_env) if viewer_path else None
    model = ghost.model if ghost is not None else func_env.mj_model

    data = mujoco.MjData(model)

    step_dt = playback.frame_dt(func_env, model)
    jit_reset, jit_step = playback.make_stepper(cfg, func_env)

    env_model = getattr(func_env, "mj_model", None) or model
    env_nq, env_nbody = int(env_model.nq), int(env_model.nbody)

    controls = _ViewerControls()
    print(_CONTROLS_HELP, flush=True)

    with mujoco.viewer.launch_passive(
        model, data, key_callback=controls.key_callback
    ) as viewer:
        if vision:
            if not (model.cam_resolution > 1).any():
                print(
                    "--vision: no camera declares a resolution, so MuJoCo has "
                    "no frustum to draw",
                    flush=True,
                )
            with viewer.lock():
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CAMERA] = 1

        key, reset_key = jax.random.split(key)
        state = jit_reset(reset_key)

        def show(state):
            """Synchronizes environment state into the visualizer MjData buffer."""
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
            mujoco.mj_forward(model, data)
            if pixel_window is not None:
                pixel_window.show(state.obs)

        show(state)

        score = 0.0
        actions = []
        perturbing = False

        while viewer.is_running():
            step_start = time.time()

            if controls.reset_requested:
                controls.reset_requested = False
                key, reset_key = jax.random.split(key)
                state = jit_reset(reset_key)
                mujoco.mj_resetData(model, data)
                show(state)
                score = 0.0
                actions = []
                perturbing = False
                print("[viewer] reset", flush=True)

            pert = viewer.perturb
            pert_active = bool(pert.active) and int(pert.select) > 0

            if controls.paused:
                if pert_active:
                    mujoco.mjv_applyPerturbPose(model, data, pert, 1)
                    mujoco.mj_forward(model, data)
                    state = _write_state_data(state, qpos=data.qpos[:env_nq])
                viewer.sync()
                if pixel_window is not None:
                    pixel_window.poll()
                time.sleep(max(0.0, step_dt - (time.time() - step_start)))
                continue

            if pert_active or perturbing:
                data.xfrc_applied[:] = 0.0
                if pert_active:
                    mujoco.mjv_applyPerturbForce(model, data, pert)
                state = _write_state_data(
                    state, xfrc_applied=data.xfrc_applied[:env_nbody].copy()
                )
                perturbing = pert_active

            obs_b = jnp.expand_dims(state.obs, axis=0)
            action, _noise, _extras = agent.select_action(
                obs_b, key, evaluate=True,
            )

            state = jit_step(state, action[0])

            data.ctrl = action
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
                state = jit_reset(reset_key)
                mujoco.mj_resetData(model, data)
                show(state)
                score = 0.0
                actions = []
                perturbing = False

            viewer.sync()

            time_until_next_step = step_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    if pixel_window is not None:
        pixel_window.close()


if __name__ == "__main__":
    main()