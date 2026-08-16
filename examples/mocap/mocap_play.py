"""Kinematic playback of a single CMU mocap clip — no policy, no physics.

``roxie/play.py`` renders a *trained policy* (with the reference ghost beside
it), so a bad clip and a bad policy look alike there. This script drops the
agent entirely: it loads one clip through the exact training pipeline
(``_load_single_clip`` — same resample to ``ctrl_dt``, same quaternion
canonicalisation, despiking and floor-grounding) and drives ``data.qpos``
frame by frame with ``mj_forward``. Whatever you see here is precisely the
target the tracking reward asks the policy to reproduce.

Before opening the viewer it prints a feasibility report on the clip: floor
penetration of the reference pose, joint angles outside the model's own limits,
and per-frame jumps in the joint angles. Those are the three ways a retargeted
clip can be untrackable no matter how good the agent is — the reference asks for
a pose the plant cannot occupy, or teleports between frames.

``--raw`` replays the clip *before* despiking and grounding (only the resample
and the quaternion sign fix, which linear interpolation requires), so the
pipeline's own contribution can be separated from the source data's.

Run from the repo root::

    python examples/mocap/mocap_play.py CMU_016_22
    python examples/mocap/mocap_play.py CMU_016_22 --speed 0.25
    python examples/mocap/mocap_play.py CMU_016_22 --raw

Viewer keys: SPACE pause/resume, LEFT/RIGHT step one frame while paused,
R restart from the first frame.
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import sys
import time
import warnings

import click
import mujoco
import mujoco.viewer
import numpy as np

from roxie.utils import hydra_searchpath

sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))

from examples.mocap.cmu_mocap_data import (  # noqa: E402
    _canonicalize_quat_sign,
    _cmu_data,
    _foot_geom_ids,
    _load_single_clip,
    _recompute_qvel,
    build_cmu_humanoid,
    mocap_loader,
)

# GLFW keycodes handed to the passive viewer's key_callback.
_KEY_SPACE = 32
_KEY_RIGHT = 262
_KEY_LEFT = 263
_KEY_R = 82


def _open_loader():
    h5_path = _cmu_data.get_path_for_cmu(version="2020")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="label\\(\\) is deprecated")
        return mocap_loader.HDF5TrajectoryLoader(h5_path)


def _load_raw_clip(clip_id, loader, mj_model, ctrl_dt):
    """Load a clip with only the steps linear resampling *requires*.

    Mirrors ``_load_single_clip`` minus the Hampel despike and the floor
    grounding, so playing this next to the processed version attributes any
    weirdness either to the source retargeting or to our cleanup.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="label\\(\\) is deprecated")
        trajectory = loader.get_trajectory(clip_id, zero_out_velocities=False)
        traj = trajectory.as_dict()

    qpos = np.concatenate(
        [traj["walker/position"], traj["walker/quaternion"], traj["walker/joints"]],
        axis=1,
    ).astype(np.float32)
    qpos, _ = _canonicalize_quat_sign(qpos)

    clip_dt = trajectory.dt
    if abs(clip_dt - ctrl_dt) > 1e-6:
        new_len = int(qpos.shape[0] * clip_dt / ctrl_dt)
        orig_t = np.linspace(0, 1, qpos.shape[0])
        new_t = np.linspace(0, 1, new_len)
        qpos = np.array(
            [np.interp(new_t, orig_t, qpos[:, i]) for i in range(qpos.shape[1])]
        ).T.astype(np.float32)

    rq = qpos[:, 3:7]
    qpos[:, 3:7] = rq / np.clip(np.linalg.norm(rq, axis=1, keepdims=True), 1e-8, None)
    return {
        "qpos": qpos,
        "qvel": _recompute_qvel(mj_model, qpos, ctrl_dt),
        "ground_offset": 0.0,
    }


def _report(mj_model, qpos, qvel, ctrl_dt):
    """Print a trackability report for the reference trajectory.

    Everything here is a property of the *clip*, evaluated with the same model
    the env simulates, so a failure means no policy can track it:

      - ``floor penetration``: how deep the reference pose puts a geom into the
        floor (``contact.dist`` < 0). The policy would have to sink through the
        ground to match it.
      - ``self penetration``: the same for humanoid-vs-humanoid pairs. Retargeted
        mocap routinely puts a limb inside the torso; under ``collisions: full``
        the solver pushes the body out of that pose every step, so the reference
        is unreachable by construction (``collisions: ground`` removes those
        contacts — see ``_configure_collisions``).
      - ``joint-limit violations``: reference angles outside ``jnt_range``. The
        constraint solver pushes back against these every step.
      - ``max frame jump``: the largest single-frame change in any joint angle.
        A surviving retargeting glitch shows up as a jump far above the clip's
        own typical motion, which no torque/servo bandwidth can follow.
    """
    data = mujoco.MjData(mj_model)
    T = qpos.shape[0]

    floor_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    # (worst penetration depth, frame) for floor contacts and for self contacts.
    worst = {"floor": (0.0, -1), "self": (0.0, -1)}
    self_frames = 0
    min_body_z = np.inf
    for t in range(T):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(mj_model, data)
        min_body_z = min(min_body_z, float(data.xpos[2:, 2].min()))
        n = data.ncon
        if not n:
            continue
        dist = data.contact.dist[:n]
        is_floor = (data.contact.geom[:n] == floor_id).any(axis=1)
        if is_floor.any() and dist[is_floor].min() < worst["floor"][0]:
            worst["floor"] = (float(dist[is_floor].min()), t)
        if (~is_floor).any():
            self_frames += 1
            if dist[~is_floor].min() < worst["self"][0]:
                worst["self"] = (float(dist[~is_floor].min()), t)

    # Hinge joints only (the free joint has no range); jnt_qposadr maps each to
    # its qpos slot, and jnt_limited says whether the range is enforced.
    limited = np.nonzero(mj_model.jnt_limited)[0]
    adr = mj_model.jnt_qposadr[limited]
    lo, hi = mj_model.jnt_range[limited, 0], mj_model.jnt_range[limited, 1]
    q = qpos[:, adr]
    over = np.maximum(q - hi, lo - q)                       # >0 == outside range
    viol_frac = float((over > 0).mean())
    worst_j = int(over.max(axis=0).argmax()) if limited.size else -1

    dq = np.abs(np.diff(qpos[:, adr], axis=0))
    jump = float(dq.max()) if dq.size else 0.0
    jump_t = int(dq.max(axis=1).argmax()) if dq.size else -1

    print(f"  frames            : {T}  ({T * ctrl_dt:.2f} s @ ctrl_dt={ctrl_dt})")
    print(f"  root height       : min={qpos[:, 2].min():.3f}  max={qpos[:, 2].max():.3f}")
    print(f"  lowest body z     : {min_body_z:+.4f} m")
    pen, pen_t = worst["floor"]
    if pen_t >= 0:
        print(f"  floor penetration : {-pen * 100:.2f} cm at frame {pen_t}")
    else:
        print("  floor penetration : none (reference never touches the floor)")
    pen, pen_t = worst["self"]
    if pen_t >= 0:
        print(
            f"  self penetration  : {-pen * 100:.2f} cm at frame {pen_t}; "
            f"limbs in contact on {100 * self_frames / T:.0f}% of frames"
        )
    else:
        print("  self penetration  : none (limbs never touch)")
    if limited.size:
        name = mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_JOINT, int(limited[worst_j]))
        print(
            f"  joint limits      : {100 * viol_frac:.2f}% of (frame, joint) samples "
            f"outside range; worst {name} by {np.degrees(over[:, worst_j].max()):.3f} deg"
        )
    print(
        f"  max frame jump    : {np.degrees(jump):.1f} deg at frame {jump_t} "
        f"({np.degrees(jump) / ctrl_dt:.0f} deg/s)"
    )
    print(f"  max |qvel|        : {np.abs(qvel).max():.1f}")


@click.command()
@click.argument("clip_id", type=str)
@click.option("--ctrl-dt", type=float, default=0.025, help="Resample timestep (s).")
@click.option("--speed", type=float, default=1.0, help="Playback rate (1.0 = realtime).")
@click.option("--start-frame", type=int, default=0, help="First frame to play.")
@click.option(
    "--raw",
    is_flag=True,
    help="Play the clip without despiking/grounding (source data as retargeted).",
)
@click.option("--loop/--no-loop", default=True, help="Restart at the end of the clip.")
def main(clip_id, ctrl_dt, speed, start_frame, raw, loop):
    mj_model, _ = build_cmu_humanoid()
    loader = _open_loader()

    if clip_id not in loader.keys():
        raise click.BadParameter(f"unknown clip id {clip_id!r}")

    if raw:
        clip = _load_raw_clip(clip_id, loader, mj_model, ctrl_dt)
    else:
        clip = _load_single_clip(
            clip_id, loader, mj_model, ctrl_dt, _foot_geom_ids(mj_model)
        )

    qpos, qvel = clip["qpos"], clip["qvel"]
    print(f"\n{clip_id} ({'raw' if raw else 'processed'}):")
    print(f"  ground offset     : {clip['ground_offset']:.4f} m")
    _report(mj_model, qpos, qvel, ctrl_dt)
    print()

    T = qpos.shape[0]
    frame = start_frame % T
    state = {"paused": False, "step": 0, "restart": False}

    def key_callback(keycode):
        if keycode == _KEY_SPACE:
            state["paused"] = not state["paused"]
        elif keycode == _KEY_RIGHT:
            state["step"] += 1
        elif keycode == _KEY_LEFT:
            state["step"] -= 1
        elif keycode == _KEY_R:
            state["restart"] = True

    data = mujoco.MjData(mj_model)
    frame_dt = ctrl_dt / max(speed, 1e-6)

    with mujoco.viewer.launch_passive(
        mj_model, data, key_callback=key_callback
    ) as viewer:
        while viewer.is_running():
            step_start = time.time()

            data.qpos[:] = qpos[frame]
            data.qvel[:] = qvel[frame]
            mujoco.mj_forward(mj_model, data)
            viewer.sync()

            print(
                f"\rframe {frame + 1:5d}/{T}  t={frame * ctrl_dt:6.2f}s  "
                f"root_z={qpos[frame, 2]:+.3f}"
                f"{'  [paused]' if state['paused'] else '         '}",
                end="",
                flush=True,
            )

            if state["restart"]:
                frame, state["restart"] = 0, False
            elif state["paused"]:
                frame = (frame + state["step"]) % T
            else:
                frame += 1
                if frame >= T:
                    if not loop:
                        break
                    frame = 0
            state["step"] = 0

            remaining = frame_dt - (time.time() - step_start)
            if remaining > 0:
                time.sleep(remaining)

    print()


if __name__ == "__main__":
    main()
