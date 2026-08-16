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
from ml_collections import config_dict

from examples.mocap.cmu_mocap_data import build_cmu_humanoid, load_cmu_clips
from roxie.environment.loader import EnvBundle, TerminationWrapper
from roxie.utils import hydra_searchpath
import xml.etree.ElementTree as ET

import mujoco

import copy
from examples.mocap.mocap_tracking import MocapTrackingEnv, resolve_collision_mode

# The env has no hard-coded defaults: its config_dict is assembled from the
# Hydra config groups under experiments/mocap/ — the env.config block (core
# params) and the env.reward block (reward shaping). These two files are the
# single source of truth, used both by Hydra-launched runs (via build_mocap_env)
# and by standalone scripts (via load_default_config).
_MOCAP_CONFIG_DIR = hydra_searchpath.REPO_ROOT / "experiments" / "mocap"
_DEFAULT_ENV_CONFIG = _MOCAP_CONFIG_DIR / "env_config" / "cmu.yaml"
_DEFAULT_REWARD_CONFIG = _MOCAP_CONFIG_DIR / "reward" / "cmu_tracking.yaml"


def _assemble_config(config_block: Any, reward_block: Any) -> config_dict.ConfigDict:
    """Assemble the env's ml_collections config from the two Hydra blocks.

    ``config_block`` is the env.config group (core params + nested
    ``reward_termination``); ``reward_block`` is the env.reward group, which the
    env reads under ``reward_config``. Both are OmegaConf nodes; we resolve them
    to plain Python containers and hand them to ``config_dict.ConfigDict``, which
    recursively wraps the nested dicts (so ``cfg.reward_config.w_pose`` etc.
    work). The env locks this in its constructor.
    """
    from omegaconf import OmegaConf

    raw = OmegaConf.to_container(config_block, resolve=True)
    raw["reward_config"] = OmegaConf.to_container(reward_block, resolve=True)
    return config_dict.ConfigDict(raw)


def build_env_config(cfg_env: Any) -> config_dict.ConfigDict:
    """Build the env config from a composed Hydra ``cfg.env`` (the run config)."""
    return _assemble_config(cfg_env.config, cfg_env.reward)


def load_default_config() -> config_dict.ConfigDict:
    """Build the env config straight from the YAML defaults, without Hydra.

    For standalone scripts (e.g. check_mocap_reward.py) that construct the env
    outside a Hydra run but still want the canonical, config-owned defaults —
    not a duplicate hard-coded in Python. Reads the same env_config/cmu.yaml and
    reward/cmu_tracking.yaml that the experiments compose.
    """
    from omegaconf import OmegaConf

    env_cfg = OmegaConf.load(_DEFAULT_ENV_CONFIG)
    reward_cfg = OmegaConf.load(_DEFAULT_REWARD_CONFIG)
    return _assemble_config(env_cfg.config, reward_cfg.reward)


def load_mocap_env(
    config: config_dict.ConfigDict | None = None,
    clip_ids: list[str] | None = None,
    gpu_clip_budget: int = 0,
    impl: str = "jax",
    naconmax: int | None = None,
    njmax: int | None = None,
    naccdmax: int | None = None,
    collisions: str = "full",
    graph_mode: str | None = None,
    clip_swap: bool = True,
    clip_seed: int = 0,
    actuation: str = "torque",
    actuation_kp_scale: float = 1.0,
    actuation_kv_ratio: float = 0.1,
):
    # No Python default schema: fall back to the YAML-owned defaults so callers
    # outside a Hydra run still get the canonical config.
    if config is None:
        config = load_default_config()

    mj_model, xml_path = build_cmu_humanoid()
    dataset = load_cmu_clips(mj_model, clip_ids=clip_ids, ctrl_dt=config.ctrl_dt)
    env = MocapTrackingEnv(
        mj_model=mj_model, dataset=dataset, config=config,
        gpu_clip_budget=gpu_clip_budget,
        impl=impl, naconmax=naconmax, njmax=njmax, naccdmax=naccdmax,
        collisions=collisions, graph_mode=graph_mode,
        clip_swap=clip_swap, clip_seed=clip_seed,
        actuation=actuation, actuation_kp_scale=actuation_kp_scale,
        actuation_kv_ratio=actuation_kv_ratio,
    )
    env._xml_path = xml_path
    train_wrapper = TerminationWrapper(env, max_episode_steps=config.episode_length)

    # Evaluation env: a shallow copy of the training env, so it SHARES the heavy
    # GPU arrays (reference clips + mjx model) — nothing is loaded twice — but
    # resets DETERMINISTICALLY. The training config resets stochastically
    # (`random_start` picks a random clip phase, `reset_noise_scale` jitters the
    # initial state), which is right for exploration but wrong for eval: it makes
    # every test episode start somewhere different, so even a single-clip run with
    # a deterministic policy reports nonzero test std. Override just those two
    # knobs on a copied config (the training env keeps the original) so eval
    # starts at frame 0 with no noise: single-clip eval is now exactly
    # reproducible (0 std), and multi-clip eval still samples clips at reset but
    # each rollout is deterministic.
    eval_env = copy.copy(env)
    eval_config = copy.deepcopy(config)
    eval_config.random_start = False
    eval_config.reset_noise_scale = 0.0
    eval_env._config = eval_config
    test_wrapper = TerminationWrapper(eval_env, max_episode_steps=config.episode_length)
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

    The Warp budgets are sized per collision mode (``env.collisions``: "full",
    "ground" or "feet" — see ``_configure_collisions``), since that decides how
    many geom pairs can ever be in contact. On Warp memory stays linear in the
    budget (naconmax/njmax) and does not blow up with the number of *potential*
    geom pairs. NOTE: the classic JAX/MJX backend ignores naconmax and instead
    statically sizes its contact arrays to all potential pairs (~980 for "full"
    vs ~45 for "ground"), so self-collision is markedly heavier there — prefer
    "ground" for memory-constrained impl=jax runs.

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
    collisions = resolve_collision_mode(cfg_env)
    # Per-mode (naconmax_per_world, njmax) for training. naconmax must cover the
    # BROADPHASE candidate pairs (AABB overlaps), not just the actual contacts:
    # with self-collision many limb AABBs overlap, so broadphase peaks near
    # ~48/world (64 leaves headroom). "ground" only ever pairs geoms with the
    # floor plane, whose AABB spans the world, so its candidates are the ~45
    # humanoid geoms; "feet" restricts that to the 13 whitelisted ones. njmax is
    # the per-world constraint budget (nefc): each contact adds several friction
    # rows on top of the joint limits, so self-collision peaks near ~180.
    _WARP_BUDGETS = {"full": (64, 256), "ground": (48, 192), "feet": (16, 128)}
    if mode == "train" and impl == "warp":
        per_world, per_world_njmax = _WARP_BUDGETS[collisions]
        if naconmax is None:
            naconmax = int(cfg_env.parallel_envs) * per_world
        if njmax is None:
            njmax = per_world_njmax
        if naccdmax is None:
            # Only the hand ellipsoids hit the EPA path (max ~2/world, almost
            # never); 4/world is generous and keeps the EPA scratch ~8x smaller
            # than the default (= naconmax).
            naccdmax = int(cfg_env.parallel_envs) * 4
    elif mode == "play" and impl == "warp":
        # Single-world playback: mujoco_warp's own defaults are too small for
        # self-collision (broadphase peaks ~52, nefc ~88 for one world). Memory
        # is irrelevant here, so size generously above those peaks.
        per_world, per_world_njmax = _WARP_BUDGETS[collisions]
        if naconmax is None:
            naconmax = 2 * per_world
        if njmax is None:
            njmax = per_world_njmax
        if naccdmax is None:
            naccdmax = 16

    gpu_clip_budget = cfg_env.get("gpu_clip_budget", 0)
    if mode == "play":
        gpu_clip_budget = gpu_clip_budget or 32

    # Warp CUDA-graph mode. mjx's GraphMode.WARP default recaptures a graph every
    # step under JAX (buffer addresses change), which is both slow and leaks host
    # RAM (evicted graphs' native descriptors are never freed). WARP_STAGED_EX
    # captures the graph ONCE on fixed staging buffers and replays it every step
    # (a cheap device→staging memcpy per step), keeping graph-replay speed with no
    # per-step recapture and no leak. JAX/NONE avoid the leak too but run kernels
    # eagerly — far slower for this many-kernel step. Default warp to STAGED_EX.
    graph_mode = cfg_env.get("graph_mode", None)
    if impl == "warp" and graph_mode is None:
        graph_mode = "WARP_STAGED_EX"

    clip_ids = list(cfg_env.clip_ids) if cfg_env.get("clip_ids") else None

    # The env's config_dict is owned entirely by Hydra (env.config + env.reward
    # groups), not hard-coded in the env. Assemble it here; ctrl_dt and
    # episode_length, read off it inside load_mocap_env, also drive clip
    # resampling and the episode-length wrapper.
    config = build_env_config(cfg_env)

    env, test_env, _ = load_mocap_env(
        config=config,
        clip_ids=clip_ids,
        gpu_clip_budget=gpu_clip_budget,
        impl=impl,
        naconmax=naconmax,
        njmax=njmax,
        naccdmax=naccdmax,
        collisions=collisions,
        graph_mode=graph_mode,
        # clip_swap=False pins the initial gpu_clip_budget subset for the whole
        # run (fixed-subset training); the pick is seeded by env.seed so the
        # same subset recurs across runs.
        clip_swap=bool(cfg_env.get("clip_swap", True)),
        clip_seed=int(cfg_env.get("seed", 0) or 0),
        # "position" = PD-target servos (dm_control tuned gains, ctrl stays
        # [-1,1] mapped onto joint ranges); "torque" = the raw motors.
        actuation=str(cfg_env.get("actuation", "torque")),
        actuation_kp_scale=float(cfg_env.get("actuation_kp_scale", 1.0)),
        actuation_kv_ratio=float(cfg_env.get("actuation_kv_ratio", 0.1)),
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
