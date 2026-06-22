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
        reward_config=config_dict.create(
            w_pose=0.5,
            w_vel=0.1,
            w_ee=0.15,
            w_root=0.2,
            w_alive=0.05,
            sigma_pose=2.0,
            sigma_vel=0.1,
            sigma_ee=0.04,
            sigma_root=0.5,
        ),
        min_head_height=0.7,
        # Early-termination curriculum (DeepMimic-style). When enabled, an
        # episode is also terminated as soon as the joint-pose error or the root
        # (position + orientation) error exceeds a threshold. Each threshold
        # widens LINEARLY with training progress, from a tight start (``*_tight``,
        # at progress 0 -- forcing the policy to stay glued to the reference
        # early, which gives dense high-quality signal and faster learning) to a
        # loose bound (``*_loose``) reached at ``relax_fraction`` of training,
        # past which the tracking-error check is turned OFF entirely and only the
        # head-height / NaN checks remain. Progress in [0, 1] is fed in via the
        # env state (``info["progress"]``); it defaults to 1.0 (curriculum off)
        # so evaluation and playback measure the true task.
        #
        # Errors are in the same units as ``_get_reward``'s ``pose_err`` /
        # ``root_err``. Calibrated against random-action rollouts on this
        # humanoid: pose_err grows ~2/step (≈5 by step ~3, ≈25-30 near the
        # natural head-height fall at ~step 17); root_err is far smaller
        # (≈0.03 early, ~0.25-0.8 by the fall). So ``pose_tight=5`` terminates a
        # diverging episode within a few steps while ``pose_loose=30`` (≈p90-99)
        # barely fires before the fall -- a smooth handoff to "off".
        termination_curriculum=config_dict.create(
            enabled=True,
            pose=5.0,
            root=0.1
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
            # Training progress in [0, 1] driving the termination curriculum.
            # Overwritten each iteration by the trainer; defaults to 1.0 so the
            # curriculum is OFF for evaluation / playback (true-task behaviour).
            "progress": jp.float32(1.0),
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
        reward_val = self._get_reward(data, abs_idx, state.metrics)

        done = self._get_termination(data, abs_idx)

        # For a non-cyclic clip, terminate `look_ahead` frames early: the obs
        # references frames up to phase_idx + look_ahead, so stopping at the
        # last frame would let the horizon run off the clip end (and wrap via
        # the modulo in _get_obs). look_ahead=1 reduces to the last frame.
        clip_ended = jp.where(
            self._config.cyclic,
            jp.float32(0),
            jp.float32(phase_idx >= clip_len - self._config.look_ahead),
        )
        done = jp.maximum(done, clip_ended)

        info = {
            "rng": rng,
            "phase_idx": phase_idx,
            "clip_start": state.info["clip_start"],
            "clip_len": clip_len,
            "last_act": action,
            # Carry progress across steps (the trainer refreshes it each
            # iteration; eval/playback keep the reset default).
            "progress": state.info["progress"],
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

        return (
            cfg.w_pose * r_pose
            + cfg.w_vel * r_vel
            + cfg.w_ee * r_ee
            + cfg.w_root * r_root
            + cfg.w_alive
        )

    def _get_termination(
        self, data: mjx.Data, abs_idx: jax.Array
    ) -> jax.Array:
        head_height = data.xpos[self._head_body_id, 2]
        fall = head_height < self._config.min_head_height
        nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # DeepMimic-style tracking-error termination, with thresholds that widen
        # as training progresses (see ``termination_curriculum`` in the config).
        # ``progress`` defaults to 1.0 at reset, which puts both schedules at
        # their "off" value -> diverged is never True for eval / playback.
        tc = self._config.termination_curriculum
        if tc.enabled:
            ref_qpos = self._ref_qpos[abs_idx]
            pose_err = jp.sum(jp.square(data.qpos[7:] - ref_qpos[7:]))
            root_pos_err = jp.sum(jp.square(data.qpos[:3] - ref_qpos[:3]))
            quat_dot = jp.dot(data.qpos[3:7], ref_qpos[3:7])
            root_err = root_pos_err + (1.0 - jp.square(quat_dot))

            # Linear ramp on normalized training progress: w goes 0 -> 1 over
            # [0, relax_fraction], so each bound widens tight -> loose. Progress
            # is the fraction of the total env-step budget consumed
            # (steps / max_steps), so the schedule is independent of the number
            # of parallel envs. Past relax_fraction the tracking-error check is
            # turned fully OFF (only the head-height / NaN backstop remains).
            diverged = ((pose_err > tc.pose) | (root_err > tc.root))
        else:
            diverged = jp.bool_(False)

        return jp.where(
            self._config.early_termination,
            fall | nan_check | diverged,
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
