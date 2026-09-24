"""Renders a trained checkpoint into an animated GIF, offscreen and headless.

`roxie/play.py` replays a checkpoint into an interactive MuJoCo viewer, which
needs a display. This renders the same rollout to frames instead, for the
README and for looking at a policy over ssh.

Rolling out and rendering are two passes on purpose: the first collects
`(qpos, qvel)` per step for `--episodes` episodes, and only the best-scoring
one is handed to the renderer, so a single unlucky reset does not decide the
figure. That defaults to one episode because several of the dm_control tasks
reset to a fixed pose — `HumanoidWalk` does — and with a deterministic actor
every extra episode there retraces the first exactly. Raise it for a task
whose reset actually randomizes.

    uv run python scripts/render_gif.py \\
        --checkpoint-path outputs/<run>/checkpoints/<step> \\
        --output images/humanoid-walk.gif
"""

import os

# Selects the headless EGL rendering backend; must precede the mujoco import.
os.environ.setdefault("MUJOCO_GL", "egl")

import click
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from PIL import Image

from roxie.utils import playback


def roll_episode(agent, reset, step, key, max_steps):
    """Runs one episode, returning `(score, qpos frames, qvel frames)`."""
    state = reset(key)
    qpos, qvel = [np.array(state.data.qpos)], [np.array(state.data.qvel)]
    score = 0.0

    for _ in range(max_steps):
        action, _noise, _extras = agent.select_action(
            jnp.expand_dims(state.obs, axis=0), key, evaluate=True,
        )
        state = step(state, action[0])
        qpos.append(np.array(state.data.qpos))
        qvel.append(np.array(state.data.qvel))
        score += float(state.reward)
        if bool(state.done):
            break

    return score, np.stack(qpos), np.stack(qvel)


def make_camera(name, distance, azimuth, elevation):
    """Builds the camera to shoot from: a named one, or a tracking rig.

    The model's own cameras sit at a fixed distance chosen for the task, which
    frames a humanoid small. Passing no name instead tracks the root body, so
    distance and angles are free.
    """
    if name:
        return name

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = 1  # The root body; body 0 is always the world.
    camera.distance = distance
    camera.azimuth = azimuth
    camera.elevation = elevation
    return camera


def render_frames(model, qpos, qvel, camera, width, height):
    """Replays recorded physics states through an offscreen MuJoCo renderer."""
    data = mujoco.MjData(model)
    frames = []
    with mujoco.Renderer(model, height=height, width=width) as renderer:
        for q, v in zip(qpos, qvel):
            data.qpos[:] = q
            data.qvel[:] = v
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render().copy())
    return frames


def write_gif(frames, path, fps, colors, dither):
    """Writes RGB frames as a looping, palette-quantized GIF."""
    # One adaptive palette is built from the middle frame and reused, so the
    # background does not shimmer as the palette is refit per frame.
    reference = Image.fromarray(frames[len(frames) // 2]).quantize(
        colors=colors, method=Image.MEDIANCUT
    )
    # Dithering is off by default: it turns the smooth sky gradient into
    # per-frame speckle, which reads as noise and, being uncorrelated between
    # frames, also defeats the GIF's inter-frame compression.
    mode = Image.FLOYDSTEINBERG if dither else Image.NONE
    images = [
        Image.fromarray(f).quantize(palette=reference, dither=mode) for f in frames
    ]
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )


@click.command(context_settings=dict(ignore_unknown_options=True))
@click.option("--checkpoint-path", required=True, help="Path to the checkpoint dir.")
@click.option("--output", required=True, help="Path of the .gif to write.")
@click.option(
    "--camera", default="",
    help="Model camera to use; empty tracks the root body.",
)
@click.option("--distance", default=4.0, help="Tracking camera distance in metres.")
@click.option("--azimuth", default=135.0, help="Tracking camera azimuth in degrees.")
@click.option(
    "--elevation", default=-12.0,
    help="Tracking camera elevation in degrees.",
)
@click.option("--width", default=480, help="Frame width in pixels.")
@click.option("--height", default=360, help="Frame height in pixels.")
@click.option("--seconds", default=6.0, help="Seconds of simulated time to show.")
@click.option("--fps", default=25, help="Playback frames per second.")
@click.option("--episodes", default=1, help="Episodes to roll; the best is rendered.")
@click.option("--skip", default=0.0, help="Seconds to drop from the episode start.")
@click.option("--colors", default=128, help="GIF palette size.")
@click.option("--dither/--no-dither", default=False, help="Dither the palette.")
@click.option("--seed", default=0, help="Seed for the episode resets.")
@click.argument("overrides", nargs=-1, type=click.UNPROCESSED)
def main(
    checkpoint_path, output, camera, distance, azimuth, elevation,
    width, height, seconds, fps, episodes, skip, colors, dither, seed, overrides,
):
    """Renders a checkpoint's policy to an animated GIF."""
    play = playback.load_policy(checkpoint_path, overrides)
    func_env = play.func_env
    model = func_env.mj_model

    step_dt = playback.frame_dt(func_env, model)
    reset, step = playback.make_stepper(play.cfg, func_env)

    stride = max(1, round(1.0 / (fps * step_dt)))
    skip_steps = round(skip / step_dt)
    wanted = skip_steps + round(seconds / step_dt)

    key = jax.random.PRNGKey(seed)
    best = None
    for episode in range(episodes):
        key, reset_key = jax.random.split(key)
        score, qpos, qvel = roll_episode(play.agent, reset, step, reset_key, wanted)
        print(
            f"episode {episode}: score={score:.1f} steps={len(qpos) - 1}", flush=True
        )
        if best is None or score > best[0]:
            best = (score, qpos, qvel)

    score, qpos, qvel = best
    qpos, qvel = qpos[skip_steps::stride], qvel[skip_steps::stride]
    print(f"rendering {len(qpos)} frames from the episode scoring {score:.1f}")

    shot = make_camera(camera, distance, azimuth, elevation)
    frames = render_frames(model, qpos, qvel, shot, width, height)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    write_gif(frames, output, fps, colors, dither)

    size_mb = os.path.getsize(output) / 1e6
    print(f"wrote {output} — {len(frames)} frames, {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
