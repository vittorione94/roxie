"""Offline script to export humanoid XML and convert mocap clips to NPZ.

Requires dm_control: pip install dm_control

Usage:
    # Export CMU humanoid model and convert a mocap clip
    python -m roxie.data.convert_cmu --clip-id 03_01 --output-dir roxie/data

    # Export the simpler dm_control suite humanoid (21 DoF) with a standing clip
    python -m roxie.data.convert_cmu --model suite --output-dir roxie/data

    # Generate a standing-still clip from an existing XML (no dm_control needed)
    python -m roxie.data.convert_cmu --model standalone \
        --xml-path roxie/data/assets/humanoid.xml --output-dir roxie/data
"""

import argparse
import os
from pathlib import Path

import mujoco
import numpy as np


def generate_standing_clip(
    mj_model: mujoco.MjModel,
    num_frames: int = 400,
    ctrl_dt: float = 0.025,
) -> dict:
    """Generate a standing-still reference clip from the model's default pose.

    Repeats the initial equilibrium pose for all frames (no simulation),
    so the reference stays upright regardless of control.
    """
    data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, data)
    mujoco.mj_forward(mj_model, data)

    nq = mj_model.nq
    nv = mj_model.nv
    nbody = mj_model.nbody

    qpos_frame = data.qpos.copy().astype(np.float32)
    qvel_frame = np.zeros(nv, dtype=np.float32)
    body_pos_frame = data.xpos.copy().astype(np.float32)

    qpos_traj = np.tile(qpos_frame, (num_frames, 1))
    qvel_traj = np.tile(qvel_frame, (num_frames, 1))
    body_pos_traj = np.tile(body_pos_frame, (num_frames, 1, 1))

    return {
        "qpos": qpos_traj,
        "qvel": qvel_traj,
        "body_pos": body_pos_traj,
        "dt": np.float32(ctrl_dt),
    }


def export_suite_humanoid(output_dir: str) -> str:
    """Export the dm_control suite humanoid XML (bundled in mujoco_playground)."""
    from mujoco_playground._src import mjx_env
    from mujoco_playground._src.dm_control_suite import common

    xml_path = mjx_env.ROOT_PATH / "dm_control_suite" / "xmls" / "humanoid.xml"
    xml_string = xml_path.read_text()
    assets = common.get_assets()

    mj_model = mujoco.MjModel.from_xml_string(xml_string, assets)

    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    out_path = os.path.join(assets_dir, "humanoid_suite.xml")
    mujoco.mj_saveLastXML(out_path, mj_model)

    print(f"Exported suite humanoid XML to {out_path}")
    print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    return out_path


def export_cmu_humanoid(output_dir: str) -> str:
    """Export the CMU humanoid from dm_control.locomotion."""
    try:
        from dm_control import mjcf
        from dm_control.locomotion.walkers import cmu_humanoid
    except ImportError:
        raise ImportError(
            "dm_control is required for CMU humanoid export. "
            "Install with: pip install dm_control"
        )

    walker = cmu_humanoid.CMUHumanoidPositionControlledV2020()
    arena = mjcf.RootElement()
    arena.worldbody.add("geom", name="floor", type="plane", size=[100, 100, 0.2])
    arena.worldbody.add("light", name="top", pos=[0, 0, 3], directional="true")
    spawn_site = arena.worldbody.add(
        "site", name="spawn", pos=[0, 0, 0], group=3
    )
    walker.create_root_joints(spawn_site)

    physics = mjcf.Physics.from_mjcf_model(arena)
    mj_model = physics.model.ptr

    assets_dir = os.path.join(output_dir, "assets")
    os.makedirs(assets_dir, exist_ok=True)

    out_path = os.path.join(assets_dir, "cmu_humanoid.xml")
    mujoco.mj_saveLastXML(out_path, mj_model)

    print(f"Exported CMU humanoid XML to {out_path}")
    print(f"  nq={mj_model.nq}, nv={mj_model.nv}, nu={mj_model.nu}")
    return out_path


def convert_cmu_clip(
    clip_id: str,
    mj_model: mujoco.MjModel,
    output_dir: str,
    ctrl_dt: float = 0.025,
) -> str:
    """Convert a CMU mocap clip to NPZ format."""
    try:
        from dm_control.locomotion.mocap import cmu_mocap_data
        from dm_control.locomotion.walkers import cmu_humanoid
    except ImportError:
        raise ImportError(
            "dm_control is required for clip conversion. "
            "Install with: pip install dm_control"
        )

    loader = cmu_mocap_data.CMUMocapData()
    clip = loader.get_clip(clip_id)

    clip_dt = clip.dt
    clip_qpos = np.array(clip.qpos, dtype=np.float32)
    clip_qvel = np.array(clip.qvel, dtype=np.float32)

    if abs(clip_dt - ctrl_dt) > 1e-6:
        ratio = clip_dt / ctrl_dt
        orig_len = clip_qpos.shape[0]
        new_len = int(orig_len * ratio)
        orig_t = np.linspace(0, 1, orig_len)
        new_t = np.linspace(0, 1, new_len)
        clip_qpos = np.array(
            [np.interp(new_t, orig_t, clip_qpos[:, i]) for i in range(clip_qpos.shape[1])]
        ).T.astype(np.float32)
        clip_qvel = np.array(
            [np.interp(new_t, orig_t, clip_qvel[:, i]) for i in range(clip_qvel.shape[1])]
        ).T.astype(np.float32)

    data = mujoco.MjData(mj_model)
    body_pos_traj = np.zeros(
        (clip_qpos.shape[0], mj_model.nbody, 3), dtype=np.float32
    )

    for t in range(clip_qpos.shape[0]):
        nq = min(clip_qpos.shape[1], mj_model.nq)
        data.qpos[:nq] = clip_qpos[t, :nq]
        mujoco.mj_forward(mj_model, data)
        body_pos_traj[t] = data.xpos.copy()

    clips_dir = os.path.join(output_dir, "clips")
    os.makedirs(clips_dir, exist_ok=True)

    out_path = os.path.join(clips_dir, f"{clip_id}.npz")
    np.savez(
        out_path,
        qpos=clip_qpos,
        qvel=clip_qvel,
        body_pos=body_pos_traj,
        dt=np.float32(ctrl_dt),
    )
    print(
        f"Saved clip '{clip_id}' to {out_path} "
        f"({clip_qpos.shape[0]} frames, dt={ctrl_dt})"
    )
    return out_path


def main():
    parser = argparse.ArgumentParser(
        description="Export humanoid XML and convert mocap clips to NPZ"
    )
    parser.add_argument(
        "--model",
        choices=["cmu", "suite", "standalone"],
        default="cmu",
        help="Humanoid model to export (default: cmu)",
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=None,
        help="CMU clip ID to convert (e.g. '03_01'). If omitted, generates standing clip.",
    )
    parser.add_argument(
        "--xml-path",
        type=str,
        default=None,
        help="Path to existing XML (for standalone mode)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="roxie/data",
        help="Output directory (default: roxie/data)",
    )
    parser.add_argument(
        "--ctrl-dt",
        type=float,
        default=0.025,
        help="Control timestep for resampling (default: 0.025)",
    )
    parser.add_argument(
        "--num-frames",
        type=int,
        default=400,
        help="Number of frames for generated standing clip (default: 400)",
    )
    args = parser.parse_args()

    if args.model == "cmu":
        xml_path = export_cmu_humanoid(args.output_dir)
    elif args.model == "suite":
        xml_path = export_suite_humanoid(args.output_dir)
    elif args.model == "standalone":
        if args.xml_path is None:
            parser.error("--xml-path is required for standalone mode")
        xml_path = args.xml_path

    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    if args.clip_id and args.model == "cmu":
        convert_cmu_clip(args.clip_id, mj_model, args.output_dir, args.ctrl_dt)
    else:
        print("Generating standing-still reference clip...")
        clip = generate_standing_clip(
            mj_model, num_frames=args.num_frames, ctrl_dt=args.ctrl_dt
        )
        clips_dir = os.path.join(args.output_dir, "clips")
        os.makedirs(clips_dir, exist_ok=True)
        out_path = os.path.join(clips_dir, "standing.npz")
        np.savez(out_path, **clip)
        print(f"Saved standing clip to {out_path} ({clip['qpos'].shape[0]} frames)")


if __name__ == "__main__":
    main()
