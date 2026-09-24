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

import time

import click
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer

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
    """Replays a trained agent checkpoint in an interactive MuJoCo viewer."""
    play = playback.load_policy(checkpoint_path, overrides)
    cfg, func_env, agent = play.cfg, play.func_env, play.agent

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


if __name__ == "__main__":
    main()