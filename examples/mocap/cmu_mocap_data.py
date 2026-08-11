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


def _lowest_foot_surface_z(
    mj_model: mujoco.MjModel, data: mujoco.MjData, foot_geom_ids: np.ndarray
) -> float:
    """Lowest world-z of the foot *collision surface* in the current pose.

    ``geom_xpos`` is the geom ORIGIN; grounding on that leaves the collision
    surface a full geom radius below the floor (the CMU feet are 25 mm-radius
    capsules/spheres), so the reference has feet buried in the ground — an
    infeasible target that fights every stance. We instead take the geom's
    actual lowest surface point:

      - sphere (toes): ``center_z - radius``.
      - capsule (foot pads): the lower of the two end-cap centres
        (``center ± half_len * local_z``) minus the radius, so a pitched foot
        (heel-strike / toe-off) is measured at its lowest cap, not its centre.
    """
    lo = np.inf
    for gid in foot_geom_ids:
        z = float(data.geom_xpos[gid, 2])
        r = float(mj_model.geom_size[gid, 0])
        if mj_model.geom_type[gid] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            half_len = float(mj_model.geom_size[gid, 1])
            local_z_world_z = float(data.geom_xmat[gid].reshape(3, 3)[2, 2])
            z -= abs(half_len * local_z_world_z)
        lo = min(lo, z - r)
    return lo


# Bodies that carry a foot touch sensor, grouped by physical foot (the ankle
# body plus its toe body). Consumed by MocapTrackingEnv for the binary
# foot-contact observation; see _add_foot_touch_sensors.
FOOT_TOUCH_BODIES = ("lfoot", "ltoes", "rfoot", "rtoes")


def _add_foot_touch_sensors(root: ET.Element, probe_model: mujoco.MjModel) -> None:
    """Add a MuJoCo ``touch`` sensor over each foot/toe body, in place.

    A touch sensor sums the normal force of every contact whose point falls
    inside a companion site's volume, so it needs a site that encloses the
    body's collision geoms. We size each site as the body-local axis-aligned
    box bounding those geoms (using each geom's bounding radius, then a small
    margin so surface contact points land inside), taken from an already-
    compiled ``probe_model``. Unlike the raw contact list, the resulting
    ``sensordata`` is a per-world field on every backend (jax / warp / native),
    which is why the env reads foot contact through it rather than by scanning
    contacts (the Warp backend keeps contacts in a non-vmapped global arena).

    The sites are transparent and in group 4 so they never render.
    """
    sensor_el = root.find("sensor")
    if sensor_el is None:
        sensor_el = ET.SubElement(root, "sensor")

    margin = 0.01
    for name in FOOT_TOUCH_BODIES:
        bid = probe_model.body(name).id
        gids = np.nonzero(probe_model.geom_bodyid == bid)[0]
        if gids.size == 0:
            raise ValueError(f"body {name!r} has no geoms to bound a touch site")
        pos = probe_model.geom_pos[gids]            # (k, 3) body-local
        rbound = probe_model.geom_rbound[gids]      # (k,)
        lo = (pos - rbound[:, None]).min(axis=0)
        hi = (pos + rbound[:, None]).max(axis=0)
        center = (lo + hi) / 2.0
        half = (hi - lo) / 2.0 + margin

        body_el = root.find(f".//body[@name='{name}']")
        if body_el is None:
            raise ValueError(f"body {name!r} not found in walker XML")
        ET.SubElement(
            body_el, "site",
            {"name": f"{name}_touch_site", "type": "box",
             "pos": " ".join(f"{v:.5f}" for v in center),
             "size": " ".join(f"{v:.5f}" for v in half),
             "group": "4", "rgba": "0 0 0 0"},
        )
        ET.SubElement(
            sensor_el, "touch",
            {"name": f"{name}_touch", "site": f"{name}_touch_site"},
        )


def build_cmu_humanoid() -> tuple[mujoco.MjModel, str]:
    """Build the CMU humanoid MuJoCo model with a free joint and floor.

    Returns ``(mj_model, xml_path)`` where *xml_path* points to a cached
    compiled XML that can be reused for ghost-model building, etc.
    """
    # Cache key bumped to ``_touch`` when per-foot touch sensors were added, so
    # stale pre-sensor caches are rebuilt rather than silently reloaded.
    cache_path = os.path.join(_CACHE_DIR, "cmu_humanoid_v2020_grid_touch.xml")

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

    # Compile once to read foot-geom bounds, add per-foot touch sensors sized to
    # those bounds, then recompile the sensor-bearing model.
    probe_model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    _add_foot_touch_sensors(root, probe_model)

    xml_string = ET.tostring(root, encoding="unicode")
    model = mujoco.MjModel.from_xml_string(xml_string)

    os.makedirs(_CACHE_DIR, exist_ok=True)
    mujoco.mj_saveLastXML(cache_path, model)

    return model, cache_path


# MAD multiplier for the qpos spike detector: a sample is treated as a
# retargeting glitch when its per-column second difference (curvature) exceeds
# this many robust sigmas of that column's curvature. ~5 is conservative (only
# clear single-frame outliers); lower it to clean more aggressively.
DESPIKE_C = 5.0


def _canonicalize_quat_sign(qpos: np.ndarray):
    """Enforce sign continuity on the root quaternion (columns 3:7).

    A quaternion and its negation encode the same rotation (double cover), but
    retargeting can flip the sign between consecutive frames; linearly
    resampling across such a flip corrupts the orientation. Flip ``q -> -q``
    wherever it restores a non-negative dot with the previous frame. The flip
    relationship telescopes over raw consecutive dots, so a cumulative product
    of the per-pair signs gives the correct per-frame sign in one pass. Operates
    on the RAW clip, before resampling. Returns ``(qpos, num_flips)``.
    """
    q = qpos[:, 3:7]
    if q.shape[0] < 2:
        return qpos, 0
    dots = np.sum(q[1:] * q[:-1], axis=1)
    signs = np.where(dots < 0.0, -1.0, 1.0)
    cum = np.concatenate([[1.0], np.cumprod(signs)]).astype(qpos.dtype)
    qpos[:, 3:7] = q * cum[:, None]
    return qpos, int((signs < 0).sum())


def _despike_hampel(arr: np.ndarray, C: float = DESPIKE_C, eps: float = 1e-9):
    """Remove isolated single-frame spikes along time (axis 0), per column.

    Retargeting glitches show up as an out-and-back excursion in one frame:
    large curvature (second difference) that the neighbouring frames do not
    share. We flag samples whose per-column second difference deviates from the
    column median by more than ``C`` robust sigmas (MAD) and replace them with
    the mean of their two temporal neighbours. A genuine fast transition is a
    ramp (small curvature) and is left untouched.

    Returns ``(cleaned, num_corrections)``. Only interior frames (1..T-2) can be
    corrected, and only single-frame spikes -- 2+ consecutive bad frames need a
    wider window or a second pass.
    """
    out = arr.copy()
    T = out.shape[0]
    if T < 3:
        return out, 0
    d2 = out[:-2] - 2.0 * out[1:-1] + out[2:]        # curvature, frames 1..T-2
    med = np.median(d2, axis=0)
    mad = np.median(np.abs(d2 - med), axis=0)
    thresh = C * 1.4826 * mad + eps                  # ~C sigma per column
    flag = np.abs(d2 - med) > thresh                 # (T-2, D)
    neigh = 0.5 * (out[:-2] + out[2:])               # neighbour average
    out[1:-1] = np.where(flag, neigh, out[1:-1])
    return out, int(flag.sum())


def _recompute_qvel(mj_model: mujoco.MjModel, qpos: np.ndarray,
                    dt: float) -> np.ndarray:
    """Re-derive reference qvel from (cleaned) qpos in MuJoCo's convention.

    Uses ``mj_differentiatePos`` so the free-joint quaternion is mapped to an
    angular velocity in exactly the frame the simulator/reward compare against
    (``data.qvel``) -- unlike the retargeter's supplied velocity channels, whose
    frame convention need not match. Central difference on the interior, one-
    sided at the two ends.
    """
    T, nv = qpos.shape[0], mj_model.nv
    qvel = np.zeros((T, nv), dtype=np.float32)
    dq = np.zeros(nv)
    for t in range(T):
        lo, hi = max(t - 1, 0), min(t + 1, T - 1)
        if hi == lo:
            continue
        mujoco.mj_differentiatePos(mj_model, dq, (hi - lo) * dt,
                                   qpos[lo], qpos[hi])
        qvel[t] = dq
    return qvel


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

    qpos = np.concatenate([positions, quaternions, joints], axis=1).astype(np.float32)

    # Canonicalise root-quaternion sign continuity BEFORE resampling: linearly
    # interpolating across a double-cover sign flip corrupts the orientation.
    qpos, n_quatflips = _canonicalize_quat_sign(qpos)
    if n_quatflips:
        print(f"Clip {clip_id}: canonicalised {n_quatflips} quaternion sign-flips")

    clip_dt = trajectory.dt
    if abs(clip_dt - ctrl_dt) > 1e-6:
        orig_len = qpos.shape[0]
        new_len = int(orig_len * clip_dt / ctrl_dt)
        orig_t = np.linspace(0, 1, orig_len)
        new_t = np.linspace(0, 1, new_len)
        qpos = np.array(
            [np.interp(new_t, orig_t, qpos[:, i]) for i in range(qpos.shape[1])]
        ).T.astype(np.float32)

    # Clean isolated retargeting spikes on qpos, then RE-DERIVE qvel from the
    # cleaned positions. The retargeter's supplied velocity channels
    # (walker/velocity, angular_velocity, joints_velocity) are discarded: they
    # can be spiky and their frame convention need not match MuJoCo's qvel, so
    # they were never comparable to data.qvel in the reward. mj_differentiatePos
    # (in _recompute_qvel) guarantees the same convention.
    qpos, n_despiked = _despike_hampel(qpos)
    rq = qpos[:, 3:7]
    qpos[:, 3:7] = rq / np.clip(np.linalg.norm(rq, axis=1, keepdims=True), 1e-8, None)
    qvel = _recompute_qvel(mj_model, qpos, ctrl_dt)
    if n_despiked:
        print(f"Clip {clip_id}: despiked {n_despiked} qpos entries "
              f"({100.0 * n_despiked / qpos.size:.3f}%)")

    data = mujoco.MjData(mj_model)
    body_pos = np.zeros((qpos.shape[0], mj_model.nbody, 3), dtype=np.float32)
    min_surface_z = np.inf
    for t in range(qpos.shape[0]):
        data.qpos[:] = qpos[t]
        mujoco.mj_forward(mj_model, data)
        body_pos[t] = data.xpos.copy()
        lowest_body = min(data.xpos[2:, 2]) # discard the worldbody and floor
        min_surface_z = max(min(
            min_surface_z, lowest_body
        ), 0)

    min_surface_z += 0.1 # the CMU data is retargeted with a 10 cm offset above the floor, so we shift down to the lowest foot surface, not the lowest body position.
    print(f"Clip {clip_id}: {qpos.shape[0]} frames, lowest foot surface z={min_surface_z:.3f}")

    # Ground the clip: the retargeted CMU data floats above (and, at the lowest
    # frame, would sink below) the floor, which forces the policy to fly / clip
    # through the ground to track it. Shift the root z (and every derived world
    # position) down so the lowest foot COLLISION SURFACE over the clip rests
    # exactly on the floor. Using the surface (not the geom origin) is what keeps
    # the reference physically feasible — origin-grounding buries the 25 mm feet
    # a full radius deep, an unrecoverable delta that fights every stance. Rigid
    # vertical translation, so velocities are unchanged.
    qpos[:, 2] -= min_surface_z
    body_pos[:, :, 2] -= min_surface_z

    return {
        "qpos": qpos,
        "qvel": qvel,
        "body_pos": body_pos,
        "ground_offset": float(min_surface_z),
    }


def _cache_path(clip_ids: list[str] | None, ctrl_dt: float) -> str:
    import hashlib
    # Bump the version suffix whenever the on-disk layout/semantics change so
    # stale caches are not silently reused (e.g. the v2 grounding offset).
    key = f"{sorted(clip_ids) if clip_ids else 'all'}_{ctrl_dt}_v5canonquat"
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
