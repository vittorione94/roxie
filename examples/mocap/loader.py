"""Loader for the CMU mocap-tracking example environment.

This lives in ``examples`` rather than the core ``roxie`` package: the mocap
task is a worked example, so the env code and its loader are kept out of
``roxie/environment`` (which holds the generic env infrastructure). The
dependency direction is one-way — examples import from ``roxie``, never the
reverse. ``train.py``/``play.py`` reach this lazily through the ``env.builder``
(and ``env.viewer``) dotted paths set in the mocap configs, resolved with
``hydra.utils.get_method`` — the core loops never name "mocap".
"""

from typing import Any, NamedTuple

import numpy as np
from omegaconf import OmegaConf

from examples.mocap.cmu_mocap_data import build_cmu_humanoid, load_cmu_clips
from roxie.environment.loader import EnvBundle, TerminationWrapper
import xml.etree.ElementTree as ET

import mujoco

import copy
from examples.mocap.mocap_tracking import MocapTrackingEnv


def load_mocap_env(
    clip_ids: list[str] | None = None,
    ctrl_dt: float = 0.025,
    gpu_clip_budget: int = 0,
    impl: str = "jax",
    naconmax: int | None = None,
    njmax: int | None = None,
    naccdmax: int | None = None,
    self_collisions: bool = True,
    config_overrides: dict | None = None,
):
    mj_model, xml_path = build_cmu_humanoid()
    dataset = load_cmu_clips(mj_model, clip_ids=clip_ids, ctrl_dt=ctrl_dt)
    env = MocapTrackingEnv(
        mj_model=mj_model, dataset=dataset, gpu_clip_budget=gpu_clip_budget,
        impl=impl, naconmax=naconmax, njmax=njmax, naccdmax=naccdmax,
        self_collisions=self_collisions, config_overrides=config_overrides,
    )
    env._xml_path = xml_path
    train_wrapper = TerminationWrapper(env)
    test_wrapper = TerminationWrapper(env)
    return train_wrapper, test_wrapper, xml_path


def build_mocap_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for the CMU mocap-tracking env (see ``env.builder`` in configs).

    Pulls everything it needs off ``cfg_env`` so the core train/play loops stay
    env-agnostic. The mocap env ships no built-in Warp budgets, so when ``impl``
    is "warp" we auto-size them (global contact arena scales with the number of
    parallel worlds; per-world constraints are fixed) — but only for training,
    since playback is single-world. In ``"play"`` mode we also clamp the GPU
    clip pool: loading every clip bakes the full reference arrays into the
    jitted step as constants and can exhaust GPU memory.

    With self-collisions enabled the per-world contact count is higher (~15 at
    peak on real mocap poses vs. a handful for ground-only), so the Warp contact
    arena is sized with extra headroom. On Warp memory stays linear in the
    budget (naconmax/njmax) and does not blow up with the number of *potential*
    geom pairs. NOTE: the classic JAX/MJX backend ignores naconmax and instead
    statically sizes its contact arrays to all potential pairs (~980 here), so
    self-collisions are markedly heavier there — prefer ground-only
    (self_collisions=false) for memory-constrained impl=jax runs.

    ``naccdmax`` sizes the GJK/EPA convex-narrowphase scratch and defaults (in
    mujoco_warp) to the full ``naconmax``. For this humanoid only the 2 hand
    ellipsoids use the convex path — every other pair is an analytic primitive —
    and they touch convexly almost never (max 2 contacts/world). We therefore
    cap ``naccdmax`` tightly; otherwise the EPA buffers alone reserve ~90MB each
    and OOM at high ``parallel_envs``.
    """
    impl = cfg_env.get("impl", "jax")
    naconmax = cfg_env.get("naconmax", None)
    njmax = cfg_env.get("njmax", None)
    naccdmax = cfg_env.get("naccdmax", None)
    self_collisions = cfg_env.get("self_collisions", True)
    if mode == "train" and impl == "warp":
        # naconmax must cover the BROADPHASE candidate pairs (AABB overlaps),
        # not just the ~15 actual contacts: with self-collision many limb AABBs
        # overlap, so broadphase peaks near ~48/world (vs a handful ground-only).
        # 64 leaves headroom over the observed peak.
        per_world = 64 if self_collisions else 16
        if naconmax is None:
            naconmax = int(cfg_env.parallel_envs) * per_world
        if njmax is None:
            # Per-world constraint budget (nefc). Each contact adds several
            # friction rows on top of joint limits, so self-collision peaks
            # near ~180; 256 leaves headroom. Ground-only stays well under 128.
            njmax = 256 if self_collisions else 128
        if naccdmax is None:
            # Only the hand ellipsoids hit the EPA path (max ~2/world, almost
            # never); 4/world is generous and keeps the EPA scratch ~8x smaller
            # than the default (= naconmax).
            naccdmax = int(cfg_env.parallel_envs) * 4
    elif mode == "play" and impl == "warp":
        # Single-world playback: mujoco_warp's own defaults are too small for
        # self-collision (broadphase peaks ~52, nefc ~88 for one world). Memory
        # is irrelevant here, so size generously above those peaks.
        if naconmax is None:
            naconmax = 128 if self_collisions else 32
        if njmax is None:
            njmax = 256 if self_collisions else 128
        if naccdmax is None:
            naccdmax = 16

    gpu_clip_budget = cfg_env.get("gpu_clip_budget", 0)
    if mode == "play":
        gpu_clip_budget = gpu_clip_budget or 32

    clip_ids = list(cfg_env.clip_ids) if cfg_env.get("clip_ids") else None

    # Optional per-experiment overrides for the env's own config_dict. Only the
    # termination curriculum is exposed for now: a nested ``termination_curriculum``
    # mapping under ``env`` is flattened to dotted keys
    # (``termination_curriculum.<field>``) for ml_collections'
    # ``update_from_flattened_dict``. Partial overrides are fine; omitting the
    # block entirely leaves the env's default_config() values untouched.
    config_overrides = None
    tc = cfg_env.get("termination_curriculum", None)
    if tc is not None:
        tc = OmegaConf.to_container(tc, resolve=True)
        config_overrides = {f"termination_curriculum.{k}": v for k, v in tc.items()}

    env, test_env, _ = load_mocap_env(
        clip_ids,
        gpu_clip_budget=gpu_clip_budget,
        impl=impl,
        naconmax=naconmax,
        njmax=njmax,
        naccdmax=naccdmax,
        self_collisions=self_collisions,
        config_overrides=config_overrides,
    )
    return EnvBundle(env=env, test_env=test_env, env_cfg=None)


class GhostViewerExtras(NamedTuple):
    """Viewer overrides for envs that render a reference "ghost" body.

    ``play.py`` swaps in ``model`` (the env model plus a ghost copy) and steps
    the ghost along ``ref_qpos``/``ref_qvel``; ``nq``/``nv`` are the real env's
    sizes so it can split the combined state.
    """

    model: Any
    ref_qpos: Any
    ref_qvel: Any
    nq: int
    nv: int

def _build_ghost_model(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    asset = root.find("asset")
    ET.SubElement(asset, "material", {
        "name": "ghost",
        "rgba": "0.2 0.8 0.2 0.3",
    })

    worldbody = root.find("worldbody")
    torso = worldbody.find("body[@name='root']")
    ghost = copy.deepcopy(torso)

    def _process(elem):
        if "name" in elem.attrib:
            elem.attrib["name"] = "ghost/" + elem.attrib["name"]
        if elem.tag == "geom":
            elem.attrib["contype"] = "0"
            elem.attrib["conaffinity"] = "0"
            elem.attrib["material"] = "ghost"
        if elem.tag == "freejoint":
            elem.attrib["name"] = "ghost/" + elem.attrib.get("name", "freejoint")
        to_remove = [c for c in elem if c.tag in ("site", "camera", "light")]
        for c in to_remove:
            elem.remove(c)
        for child in elem:
            _process(child)

    _process(ghost)
    worldbody.append(ghost)

    xml_string = ET.tostring(root, encoding="unicode")
    return mujoco.MjModel.from_xml_string(xml_string)


def mocap_viewer_extras(env: Any) -> GhostViewerExtras:
    """Build the ghost-rendering overrides for the mocap playback viewer.

    Named by ``env.viewer`` in the mocap configs and resolved by ``play.py``;
    keeps the ghost-model construction with the env it belongs to instead of
    branching on an env-type string in the core viewer loop.
    """
    mocap_env = env.env
    model = _build_ghost_model(env._xml_path)
    return GhostViewerExtras(
        model=model,
        ref_qpos=np.array(mocap_env._ref_qpos),
        ref_qvel=np.array(mocap_env._ref_qvel),
        nq=env.mj_model.nq,
        nv=env.mj_model.nv,
    )
