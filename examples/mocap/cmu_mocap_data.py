"""Load CMU mocap data and build humanoid model from dm_control.

The CMU motion capture data is retargeted to dm_control's CMU Humanoid
(V2020 position-controlled variant). This module loads the model XML
directly from the dm_control package and fetches pre-processed clips
from DeepMind's public HDF5 file.
"""

import os
import warnings
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from tqdm import tqdm

import dm_control.locomotion.walkers.cmu_humanoid as _cmu_module
from dm_control.locomotion.mocap import cmu_mocap_data as _cmu_data
from dm_control.locomotion.mocap import loader as mocap_loader

_WALKER_XML = os.path.join(
    os.path.dirname(_cmu_module.__file__),
    "assets",
    "humanoid_CMU_V2020.xml",
)

_CACHE_DIR = os.path.expanduser("~/.cache/roxie")

# Foot/toe geoms used to ground each clip on the floor. The retargeted CMU
# data floats above z=0, so we shift every clip down until its lowest foot
# contact rests on the floor (see ``_load_single_clip``).
_FOOT_GEOMS = (
    "lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2",
    "rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2",
)


def _foot_geom_ids(mj_model: mujoco.MjModel) -> np.ndarray:
    ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in _FOOT_GEOMS]
    return np.array([i for i in ids if i >= 0], dtype=np.int32)


def build_cmu_humanoid() -> tuple[mujoco.MjModel, str]:
    """Build the CMU humanoid MuJoCo model with a free joint and floor.

    Returns ``(mj_model, xml_path)`` where *xml_path* points to a cached
    compiled XML that can be reused for ghost-model building, etc.
    """
    cache_path = os.path.join(_CACHE_DIR, "cmu_humanoid_v2020_grid.xml")

    if os.path.exists(cache_path):
        model = mujoco.MjModel.from_xml_path(cache_path)
        return model, cache_path

    tree = ET.parse(_WALKER_XML)
    root = tree.getroot()

    # Classic dm_control blue checkered floor (texture + material).
    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset, "texture",
        {"name": "grid", "type": "2d", "builtin": "checker",
         "rgb1": ".1 .2 .3", "rgb2": ".2 .3 .4",
         "width": "512", "height": "512"},
    )
    ET.SubElement(
        asset, "material",
        {"name": "grid", "texture": "grid", "texrepeat": "1 1",
         "texuniform": "true", "reflectance": ".2"},
    )

    worldbody = root.find("worldbody")
    ET.SubElement(
        worldbody, "geom",
        {"name": "floor", "type": "plane", "size": "100 100 0.2",
         "material": "grid"},
    )
    ET.SubElement(
        worldbody, "light",
        {"name": "top", "pos": "0 0 3", "directional": "true"},
    )

    root_body = worldbody.find('body[@name="root"]')
    root_body.insert(0, ET.Element("freejoint", {"name": "rootjoint"}))

    xml_string = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    mujoco.mj_saveLastXML(cache_path, model)

    return model, cache_path


def _load_single_clip(
    clip_id: str,
    loader: mocap_loader.HDF5TrajectoryLoader,
    mj_model: mujoco.MjModel,
    ctrl_dt: float,
    foot_geom_ids: np.ndarray,
) -> dict:
    """Load one clip, resample to ctrl_dt, compute body positions."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="label\\(\\) is deprecated")
        trajectory = loader.get_trajectory(clip_id, zero_out_velocities=False)
        traj_dict = trajectory.as_dict()

    positions = traj_dict["walker/position"]
    quaternions = traj_dict["walker/quaternion"]
    joints = traj_dict["walker/joints"]
    velocities = traj_dict["walker/velocity"]
    ang_velocities = traj_dict["walker/angular_velocity"]
    joints_vel = traj_dict["walker/joints_velocity"]

    qpos = np.concatenate([positions, quaternions, joints], axis=1).astype(np.float32)
    qvel = np.concatenate([velocities, ang_velocities, joints_vel], axis=1).astype(np.float32)

    clip_dt = trajectory.dt
    if abs(clip_dt - ctrl_dt) > 1e-6:
        orig_len = qpos.shape[0]
        new_len = int(orig_len * clip_dt / ctrl_dt)
        orig_t = np.linspace(0, 1, orig_len)
        new_t = np.linspace(0, 1, new_len)
        qpos = np.array(
            [np.interp(new_t, orig_t, qpos[:, i]) for i in range(qpos.shape[1])]
        ).T.astype(np.float32)
        qvel = np.array(
            [np.interp(new_t, orig_t, qvel[:, i]) for i in range(qvel.shape[1])]
        ).T.astype(np.float32)

    data = mujoco.MjData(mj_model)
    body_pos = np.zeros((qpos.shape[0], mj_model.nbody, 3), dtype=np.float32)
    min_foot_z = np.inf
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(mj_model, data)
        body_pos[t] = data.xpos.copy()
        min_foot_z = min(min_foot_z, float(data.geom_xpos[foot_geom_ids, 2].min()))

    # Ground the clip: the retargeted CMU data floats above the floor, which
    # would force the policy to fly to track it. Shifting the root z (and every
    # derived world position) down by the lowest foot height over the clip is a
    # rigid vertical translation, so velocities are unchanged.
    qpos[:, 2] -= min_foot_z
    body_pos[:, :, 2] -= min_foot_z

    return {"qpos": qpos, "qvel": qvel, "body_pos": body_pos}


def _cache_path(clip_ids: list[str] | None, ctrl_dt: float) -> str:
    import hashlib
    # Bump the version suffix whenever the on-disk layout/semantics change so
    # stale caches are not silently reused (e.g. the v2 grounding offset).
    key = f"{sorted(clip_ids) if clip_ids else 'all'}_{ctrl_dt}_v2grounded"
    h = hashlib.md5(key.encode()).hexdigest()[:12]
    return os.path.join(_CACHE_DIR, f"cmu_dataset_{h}.npz")


def load_cmu_clips(
    mj_model: mujoco.MjModel,
    clip_ids: list[str] | None = None,
    ctrl_dt: float = 0.025,
) -> dict:
    """Load CMU mocap clips into a concatenated dataset.

    When *clip_ids* is ``None`` or empty, every clip in the HDF5 file is
    loaded.  Pass an explicit list only to restrict the dataset (e.g. for
    debugging or to fit in GPU memory).

    Returns a dict with:
      - ``qpos``    (total_frames, nq)
      - ``qvel``    (total_frames, nv)
      - ``body_pos``(total_frames, nbody, 3)
      - ``clip_starts``  (num_clips,)  start index per clip
      - ``clip_lengths`` (num_clips,)  frame count per clip
    """
    cache = _cache_path(clip_ids, ctrl_dt)
    if os.path.exists(cache):
        print(f"Loading cached dataset from {cache}")
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    h5_path = _cmu_data.get_path_for_cmu(version="2020")
    loader = mocap_loader.HDF5TrajectoryLoader(h5_path)

    if not clip_ids:
        clip_ids = list(loader.keys())

    foot_geom_ids = _foot_geom_ids(mj_model)

    all_qpos, all_qvel, all_body_pos = [], [], []
    clip_starts, clip_lengths = [], []
    offset = 0

    for cid in tqdm(clip_ids, desc="Loading mocap clips"):
        clip = _load_single_clip(cid, loader, mj_model, ctrl_dt, foot_geom_ids)
        n = clip["qpos"].shape[0]
        all_qpos.append(clip["qpos"])
        all_qvel.append(clip["qvel"])
        all_body_pos.append(clip["body_pos"])
        clip_starts.append(offset)
        clip_lengths.append(n)
        offset += n

    result = {
        "qpos": np.concatenate(all_qpos, axis=0),
        "qvel": np.concatenate(all_qvel, axis=0),
        "body_pos": np.concatenate(all_body_pos, axis=0),
        "clip_starts": np.array(clip_starts, dtype=np.int32),
        "clip_lengths": np.array(clip_lengths, dtype=np.int32),
    }

    os.makedirs(_CACHE_DIR, exist_ok=True)
    np.savez(cache, **result)
    print(f"Cached dataset to {cache}")

    return result


def list_cmu_clips() -> list[str]:
    """Return available CMU mocap clip IDs."""
    h5_path = _cmu_data.get_path_for_cmu(version="2020")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="label\\(\\) is deprecated")
        loader = mocap_loader.HDF5TrajectoryLoader(h5_path)
        return list(loader.keys())
