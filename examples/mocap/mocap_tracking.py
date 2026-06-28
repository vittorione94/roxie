from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward

CMU_BODY_NAMES = {
    "head": "head",
    "torso": "thorax",
    "end_effectors": ["lhand", "rhand", "lfoot", "rfoot"],
}


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.025,
        sim_dt=0.005,
        episode_length=1000,
        early_termination=True,
        action_repeat=1,
        action_scale=1.0,
        cyclic=False,
        random_start=True,
        look_ahead=5,
        reset_noise_scale=1e-3,
        # Reward shaping. For Hydra-launched runs these are overridden by the
        # `env.reward` config group (experiments/mocap/reward/) via
        # config_overrides — that group, not this block, is the source of truth
        # for experiments. The values here are the schema + a sane standalone
        # default (the sigmas are tuned so no component saturates at 0/1; see the
        # reward group for the rationale).
        reward_config=config_dict.create(
            w_pose=0.5,
            w_vel=0.1,
            w_ee=0.15,
            w_root=0.2,
            w_alive=0.05,
            sigma_pose=8.0,
            sigma_vel=10.0,
            sigma_ee=0.15,
            sigma_root=0.3,
        ),
        min_head_height=0.7,
        # Tracking-collapse early termination. End the episode when the weighted
        # tracking reward (the four components, excluding the alive bonus) falls
        # below `min_tracking_frac` of its theoretical maximum (= sum of the
        # component weights). This is deliberately a single, lenient criterion
        # rather than a per-component cutoff: a strong component can compensate a
        # momentarily weak one, and the low default (10%) means only a genuine
        # loss of tracking ends the episode, leaving the agent room to recover
        # from transient drift early in training. Only applied when
        # `early_termination` is also True. Raise it to demand tighter tracking,
        # set `enabled=False` to fall back to fall/NaN termination only.
        reward_termination=config_dict.create(
            enabled=True,
            min_tracking_frac=0.1,
        ),
    )


GROUND_CONTACT_GEOMS = {
    "lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2",
    "rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2",
    "lhand", "rhand", "head",
}


def _configure_collisions(m: mujoco.MjModel, self_collisions: bool) -> None:
    """Set up the collision filter for the CMU humanoid.

    The dm_control CMU humanoid ships with contype=1/conaffinity=1 on every
    geom, i.e. full self-collision plus ground contact (MuJoCo still skips
    same-body and welded parent/child pairs automatically). On real mocap
    reference poses this stays cheap — at most ~15 simultaneous contacts — so
    the cost is bounded by the Warp contact budget (``naconmax``/``njmax``),
    not by the number of *potential* geom pairs. We therefore leave the native
    full-collision model in place when ``self_collisions`` is set.

    When ``self_collisions`` is False we fall back to the old behaviour:
    restrict collisions to feet/hands/head vs the floor only. This keeps the
    contact budget minimal for memory-constrained runs at the cost of letting
    limbs pass through each other.
    """
    if self_collisions:
        # Native model already has full self-collision + ground; nothing to do.
        return

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    m.geom_conaffinity[floor_id] = 1

    for name in GROUND_CONTACT_GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            m.geom_contype[gid] = 1


class MocapTrackingEnv(mjx_env.MjxEnv):

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        dataset: dict,
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        body_names: Optional[dict] = None,
        gpu_clip_budget: int = 0,
        impl: str = "jax",
        naconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        naccdmax: Optional[int] = None,
        self_collisions: bool = True,
    ):
        super().__init__(config, config_overrides)

        self._mj_model = mj_model
        self._mj_model.opt.timestep = self.sim_dt

        # Full self-collision by default; set self_collisions=False to fall back
        # to ground-only contacts for memory-constrained runs.
        _configure_collisions(self._mj_model, self_collisions)

        # Physics backend: "jax" is the classic MJX implementation; "warp" is
        # the NVIDIA-Warp backend (mujoco_warp). Both are dispatched through the
        # same mjx API, so only data construction differs (Warp needs explicit
        # contact/constraint budgets).
        self._impl = impl
        self._naconmax = naconmax
        self._njmax = njmax
        self._naccdmax = naccdmax
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl)

        self._cpu_qpos = np.asarray(dataset["qpos"])
        self._cpu_qvel = np.asarray(dataset["qvel"])
        self._cpu_body_pos = np.asarray(dataset["body_pos"])
        self._cpu_clip_starts = np.asarray(dataset["clip_starts"])
        self._cpu_clip_lengths = np.asarray(dataset["clip_lengths"])
        self._total_clips = len(self._cpu_clip_starts)

        self._gpu_clip_budget = (
            min(gpu_clip_budget, self._total_clips)
            if gpu_clip_budget > 0
            else self._total_clips
        )

        self._load_gpu_chunk()

        self._xml_path = None
        self._body_names = body_names or CMU_BODY_NAMES
        self._post_init()

    def _post_init(self) -> None:
        bn = self._body_names
        self._head_body_id = self._mj_model.body(bn["head"]).id
        self._torso_body_id = self._mj_model.body(bn["torso"]).id

        self._ee_body_ids = jp.array(
            [self._mj_model.body(name).id for name in bn["end_effectors"]]
        )

        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

    def _load_gpu_chunk(self, clip_indices=None):
        if clip_indices is None:
            if self._gpu_clip_budget >= self._total_clips:
                clip_indices = np.arange(self._total_clips)
            else:
                clip_indices = np.random.choice(
                    self._total_clips, self._gpu_clip_budget, replace=False
                )

        chunks_qpos, chunks_qvel, chunks_body_pos = [], [], []
        new_starts, new_lengths = [], []
        offset = 0

        for i in clip_indices:
            start = int(self._cpu_clip_starts[i])
            length = int(self._cpu_clip_lengths[i])
            chunks_qpos.append(self._cpu_qpos[start:start + length])
            chunks_qvel.append(self._cpu_qvel[start:start + length])
            chunks_body_pos.append(self._cpu_body_pos[start:start + length])
            new_starts.append(offset)
            new_lengths.append(length)
            offset += length

        self._ref_qpos = jp.array(np.concatenate(chunks_qpos, axis=0), dtype=jp.float32)
        self._ref_qvel = jp.array(np.concatenate(chunks_qvel, axis=0), dtype=jp.float32)
        self._ref_body_pos = jp.array(np.concatenate(chunks_body_pos, axis=0), dtype=jp.float32)
        self._clip_starts = jp.array(new_starts, dtype=jp.int32)
        self._clip_lengths = jp.array(new_lengths, dtype=jp.int32)
        self._num_clips = len(clip_indices)

    def swap_clips(self, seed=None):
        """Reshuffle the on-GPU clip subset. Returns True if clips actually
        changed (callers must reset in-progress envs, whose stored clip
        indices reference the previous chunk), False for a no-op swap."""
        if self._gpu_clip_budget >= self._total_clips:
            return False
        rng = np.random.default_rng(seed)
        clip_indices = rng.choice(
            self._total_clips, self._gpu_clip_budget, replace=False
        )
        self._load_gpu_chunk(clip_indices)
        return True

    def _init_data(self, qpos: jax.Array, qvel: jax.Array) -> mjx.Data:
        """Build and forward initial mjx.Data for the active backend.

        ``mjx_env.make_data`` takes the raw MjModel and forwards the backend
        (``impl``) plus the Warp contact/constraint budgets, so both "jax" and
        "warp" go through the same path; the budgets are ``None`` (ignored) for
        the JAX backend. It does not run a forward pass, so we do that here.
        """
        data = mjx_env.make_data(
            self._mj_model,
            qpos=qpos,
            qvel=qvel,
            impl=self._impl,
            naconmax=self._naconmax,
            naccdmax=self._naccdmax,
            njmax=self._njmax,
        )
        return mjx.forward(self.mjx_model, data)

    def _abs_idx(self, info: dict[str, Any]) -> jax.Array:
        return info["clip_start"] + info["phase_idx"]

    def reset(self, rng: jax.Array) -> mjx_env.State:
        rng, clip_rng, start_rng, qpos_rng, qvel_rng = jax.random.split(rng, 5)

        clip_idx = jax.random.randint(clip_rng, (), 0, self._num_clips)
        clip_start = self._clip_starts[clip_idx]
        clip_len = self._clip_lengths[clip_idx]

        # For a non-cyclic clip, keep the random start far enough from the end
        # that the full look_ahead horizon stays inside the clip (mirrors the
        # early termination in step). Cyclic clips wrap, so the whole clip is
        # fair game.
        start_high = jp.where(
            self._config.cyclic,
            clip_len,
            jp.maximum(clip_len - self._config.look_ahead, 1),
        )
        start_idx = jax.lax.cond(
            self._config.random_start,
            lambda r: jax.random.randint(r, (), 0, start_high),
            lambda r: jp.int32(0),
            start_rng,
        )

        abs_idx = clip_start + start_idx
        # Add slight noise to the reference start state to encourage exploration
        # and make the policy robust to small state perturbations.
        noise = self._config.reset_noise_scale
        qpos = self._ref_qpos[abs_idx] + noise * jax.random.normal(
            qpos_rng, (self.mjx_model.nq,)
        )
        qvel = self._ref_qvel[abs_idx] + noise * jax.random.normal(
            qvel_rng, (self.mjx_model.nv,)
        )
        data = self._init_data(qpos, qvel)

        info = {
            "rng": rng,
            "phase_idx": start_idx,
            "clip_start": clip_start,
            "clip_len": clip_len,
            "last_act": jp.zeros(self.mjx_model.nu),
            # Clip-end truncation flag (see step). False at reset; kept in the
            # info pytree so reset/step states share an identical structure (the
            # trainer's auto-reset selects between them leaf-by-leaf).
            "truncation": jp.bool_(False),
        }

        metrics = {
            "reward/pose": jp.zeros(()),
            "reward/vel": jp.zeros(()),
            "reward/ee": jp.zeros(()),
            "reward/root": jp.zeros(()),
        }

        reward_val, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, _ = jax.random.split(state.info["rng"])

        ctrl = jp.clip(
            action * self._config.action_scale, self._lowers, self._uppers
        )
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        clip_len = state.info["clip_len"]
        phase_idx = (state.info["phase_idx"] + 1) % clip_len

        abs_idx = state.info["clip_start"] + phase_idx
        reward_val, tracking = self._get_reward(data, abs_idx, state.metrics)

        # Genuine termination: fall / NaN / tracking collapse.
        terminated = self._get_termination(data, tracking)

        # For a non-cyclic clip, end `look_ahead` frames before the last frame:
        # the obs references frames up to phase_idx + look_ahead, so stopping at
        # the final frame would let the horizon run off the clip end (and wrap
        # via the modulo in _get_obs). This is a TRUNCATION -- the reference
        # trajectory simply ran out, not a failure -- so it is reported via
        # info["truncation"] and kept OUT of the termination signal, letting the
        # critic bootstrap the cut-off state's value instead of zeroing it.
        # look_ahead=1 reduces to the last frame.
        clip_truncated = jp.where(
            self._config.cyclic,
            jp.bool_(False),
            phase_idx >= clip_len - self._config.look_ahead,
        )

        done = jp.logical_or(terminated, clip_truncated)

        info = {
            "rng": rng,
            "phase_idx": phase_idx,
            "clip_start": state.info["clip_start"],
            "clip_len": clip_len,
            "last_act": action,
            "truncation": clip_truncated,
        }

        obs = self._get_obs(data, info)
        done = done.astype(jp.float32)
        return mjx_env.State(data, obs, reward_val, done, state.metrics, info)

    def _get_obs(self, data: mjx.Data, info: dict[str, Any]) -> jax.Array:
        clip_len = info["clip_len"]

        # Look ahead over the next `look_ahead` reference frames and feed the
        # delta between each future frame and the *current* state, rather than
        # the absolute reference. The per-frame diffs are stitched together so
        # the policy sees the trajectory it has to close over the horizon.
        steps = jp.arange(1, self._config.look_ahead + 1)
        future_local = (info["phase_idx"] + steps) % clip_len
        future_abs = info["clip_start"] + future_local

        ref_qpos = self._ref_qpos[future_abs]  # (look_ahead, nq)
        ref_qvel = self._ref_qvel[future_abs]  # (look_ahead, nv)

        d_joints = ref_qpos[:, 7:] - data.qpos[7:]
        d_jvel = ref_qvel[:, 6:] - data.qvel[6:]
        d_quat = ref_qpos[:, 3:7] - data.qpos[3:7]
        d_height = ref_qpos[:, 2:3] - data.qpos[2:3]

        # Flatten frame-by-frame: [frame1 block, frame2 block, ...].
        ref_delta = jp.concatenate(
            [d_joints, d_jvel, d_quat, d_height], axis=-1
        ).reshape(-1)

        obs = jp.concatenate([
            data.qpos[7:],
            data.qvel[6:],
            data.qpos[3:7],
            data.qvel[3:6],
            data.qpos[2:3],
            data.qvel[0:3],
            info["last_act"],
            ref_delta,
        ])
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        abs_idx: jax.Array,
        metrics: dict[str, Any],
    ) -> jax.Array:
        cfg = self._config.reward_config
        ref_qpos = self._ref_qpos[abs_idx]
        ref_qvel = self._ref_qvel[abs_idx]
        ref_body_pos = self._ref_body_pos[abs_idx]

        pose_err = jp.sum(jp.square(data.qpos[7:] - ref_qpos[7:]))
        r_pose = jp.exp(-pose_err / cfg.sigma_pose)

        vel_err = jp.sum(jp.square(data.qvel[6:] - ref_qvel[6:]))
        r_vel = jp.exp(-vel_err / cfg.sigma_vel)

        ee_pos = data.xpos[self._ee_body_ids]
        ref_ee_pos = ref_body_pos[self._ee_body_ids]
        ee_err = jp.sum(jp.square(ee_pos - ref_ee_pos))
        r_ee = jp.exp(-ee_err / cfg.sigma_ee)

        root_pos_err = jp.sum(jp.square(data.qpos[:3] - ref_qpos[:3]))
        quat_dot = jp.dot(data.qpos[3:7], ref_qpos[3:7])
        root_quat_err = 1.0 - jp.square(quat_dot)
        root_err = root_pos_err + root_quat_err
        r_root = jp.exp(-root_err / cfg.sigma_root)

        metrics["reward/pose"] = r_pose
        metrics["reward/vel"] = r_vel
        metrics["reward/ee"] = r_ee
        metrics["reward/root"] = r_root

        # Weighted tracking reward (everything but the constant alive bonus).
        # Returned alongside the total so termination can gate on it without
        # recomputing the components.
        tracking = (
            cfg.w_pose * r_pose
            + cfg.w_vel * r_vel
            + cfg.w_ee * r_ee
            + cfg.w_root * r_root
        )
        return tracking + cfg.w_alive, tracking

    def _get_termination(
        self, data: mjx.Data, tracking: jax.Array
    ) -> jax.Array:
        head_height = data.xpos[self._head_body_id, 2]
        fall = head_height < self._config.min_head_height
        nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # Tracking collapse: terminate once the weighted tracking reward drops
        # below a fraction of its max (= sum of component weights). Weights and
        # the fraction are static config, so this floor is a trace-time constant.
        cfg = self._config.reward_config
        rt = self._config.reward_termination
        if rt.enabled:
            max_track = cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root
            low_track = tracking < rt.min_tracking_frac * max_track
        else:
            low_track = jp.bool_(False)

        return jp.where(
            self._config.early_termination,
            fall | nan_check | low_track,
            nan_check,
        )

    @property
    def xml_path(self) -> str:
        return self._xml_path

    @property
    def action_size(self) -> int:
        return self._mjx_model.nu

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> mjx.Model:
        return self._mjx_model
