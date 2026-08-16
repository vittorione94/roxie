"""CPU EnvPool-style mirror of the CMU mocap-tracking task.

This is a faithful re-implementation of ``MocapTrackingEnv`` (+ its
``TerminationWrapper``) on native MuJoCo (``mujoco.mj_step``) and numpy,
exposed through the same pool protocol as EnvPool's gymnasium API so the
Trainer's CPU loop (``_run_envpool``) drives it unchanged. Its purpose is a
clean CPU-vs-GPU comparison against the MJX ("jax") and Warp backends: same
observations, rewards, termination/truncation semantics and config — only the
physics backend and the vectorization strategy differ.

Semantics mirrored from the GPU path (see ``mocap_tracking.py``):
  - obs layout, look-ahead reference deltas, reset noise, random clip/start;
  - reward components (pose/vel/ee/root/torque + alive) and their metric keys;
  - termination = NaN | tracking collapse; truncation = clip end (look_ahead
    guard) | episode step-limit, with clip-end truncation clearing the
    termination flag exactly like ``TerminationWrapper`` does.

Known, deliberate difference: auto-reset is same-step (EnvPool convention —
the returned obs on a done step is the *reset* obs, not the final obs), so
truncated transitions bootstrap from the reset obs instead of the true final
obs. The JAX path bootstraps truncated steps from the true final obs. This
touches only the ~1/episode_length fraction of transitions that truncate and
matches the pre-existing EnvPool adapter behaviour.

Parallelism: one ``MjData`` per env, stepped by a persistent thread pool
(MuJoCo's python bindings release the GIL inside ``mj_step``, so threads scale
across cores without pickling). Unlike the GPU backends there are no contact
budgets to size (native MuJoCo allocates contacts dynamically) and no GPU clip
budget — the full clip dataset always lives in host RAM.

Usage via experiment YAML::

    env:
      builder: examples.mocap.mocap_envpool.build_mocap_envpool_env
      parallel_envs: 20
      num_threads: null   # default: min(parallel_envs, cpu cores)
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional

import mujoco
import numpy as np
from ml_collections import config_dict

from examples.mocap.mocap_tracking import (
    CMU_BODY_NAMES,
    FOOT_TOUCH_SENSORS,
    _configure_collisions,
    resolve_collision_mode,
)
from roxie.environment.envpool_adapter import EnvPoolWrapper
from roxie.environment.loader import EnvBundle


# Numpy mirrors of roxie.utils.math (the module is jax-only; this CPU path
# stays numpy). Kept here so the envpool obs matches the MJX obs field-for-field.
def _quat_diff_6d_np(q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
    """6D continuous rotation rep of the relative rotation q_from^-1 * q_to.

    Mirrors quat_to_rot6d(batched_quat_diff(...)) in the MJX env. Broadcasts
    over leading dims. See roxie/utils/math.py for the rationale (Zhou et al.
    2019 continuous rotation representation).
    """
    conj = np.concatenate([q_from[..., :1], -q_from[..., 1:]], axis=-1)
    q_inv = conj / np.sum(q_from * q_from, axis=-1, keepdims=True)
    w1, x1, y1, z1 = q_inv[..., 0], q_inv[..., 1], q_inv[..., 2], q_inv[..., 3]
    w2, x2, y2, z2 = q_to[..., 0], q_to[..., 1], q_to[..., 2], q_to[..., 3]
    q = np.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], axis=-1)
    return _quat_to_rot6d_np(q)


def _quat_to_rot6d_np(q: np.ndarray) -> np.ndarray:
    """MuJoCo quat (w, x, y, z) -> 6D rep (first two rotation-matrix columns)."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack([
        1.0 - 2.0 * (y * y + z * z),
        2.0 * (x * y + w * z),
        2.0 * (x * z - w * y),
        2.0 * (x * y - w * z),
        1.0 - 2.0 * (x * x + z * z),
        2.0 * (y * z + w * x),
    ], axis=-1)


def _mat_to_rot6d_np(mat: np.ndarray) -> np.ndarray:
    """Row-major 3x3 rotation matrix -> 6D rep (first two columns).

    Numpy mirror of roxie.utils.math.mat_to_rot6d. Native MjData stores each
    body frame (`xmat`) row-major as 9 contiguous floats; the 6D rep is its
    first two columns [c0x, c0y, c0z, c1x, c1y, c1z]. Batches over leading dims:
    (..., 9) -> (..., 6).
    """
    r = mat.reshape(mat.shape[:-1] + (3, 3))
    return np.concatenate([r[..., :, 0], r[..., :, 1]], axis=-1)


def _quaternion_distance_np(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Angular geodesic distance (radians) between unit quaternions.

    Numpy mirror of roxie.utils.math.quaternion_distance: 2 * arccos(|<q1, q2>|),
    the double-cover-safe geodesic angle used by the root-orientation reward.
    Batches over the leading dims (dot is taken over the last axis), so it works
    on a single pair (returns a scalar) or a stack of (..., 4) quaternions.
    """
    dot = np.clip(np.abs(np.sum(q1 * q2, axis=-1)), -1.0, 1.0)
    return 2.0 * np.arccos(dot)

# Metric keys match the GPU env's `metrics` dict so epoch logs (CSV/wandb)
# line up column-for-column across backends. Mirrors MocapTrackingEnv.reset's
# `metrics` dict field-for-field (five reward kernels + action-rate penalty +
# the root-drift diagnostic).
_METRIC_KEYS = (
    "reward/pose", "reward/vel", "reward/ee", "reward/root", "reward/torque",
    "reward/action_rate", "root_dist",
)


class MocapCpuPool:
    """Vectorized CPU mocap-tracking pool with EnvPool-style auto-reset.

    Exposes the minimal gymnasium-pool surface the ``EnvPoolWrapper`` adapter
    consumes: ``observation_space``/``action_space`` (shape/low/high only),
    ``reset() -> (obs, info)`` and
    ``step(actions) -> (obs, reward, terminated, truncated, info)``, with
    per-step reward components under ``info["metrics"]``.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        dataset: dict,
        config: config_dict.ConfigDict,
        num_envs: int,
        seed: int = 0,
        num_threads: Optional[int] = None,
        body_names: Optional[dict] = None,
    ):
        self._model = mj_model
        self._config = config
        self._num_envs = int(num_envs)
        self._n_substeps = int(round(config.ctrl_dt / config.sim_dt))

        bn = body_names or CMU_BODY_NAMES
        self._ee_body_ids = np.array(
            [mj_model.body(name).id for name in bn["end_effectors"]]
        )
        self._lowers = mj_model.actuator_ctrlrange[:, 0].copy()
        self._uppers = mj_model.actuator_ctrlrange[:, 1].copy()

        # -- proprioception + feet-contact + effort setup (mirror _post_init) --
        # Binary foot ground-contact obs: per-foot touch-sensor addresses (2, 2)
        # and the normal-force threshold that binarizes them.
        self._foot_sensor_adr = np.array(
            [[mj_model.sensor(s).adr[0] for s in foot]
             for foot in FOOT_TOUCH_SENSORS]
        )
        self._foot_contact_force_thresh = float(
            config.get("foot_contact_force_thresh", 1.0)
        )

        # Per-actuator effort normalizer for the torque penalty: forcerange when
        # force-limited, else |gear| — so |actuator_force| / limit is ~[0, 1]
        # under either actuation (identical to MocapTrackingEnv._force_limit).
        self._force_limit = np.where(
            mj_model.actuator_forcelimited.astype(bool),
            mj_model.actuator_forcerange[:, 1],
            np.abs(mj_model.actuator_gear[:, 0]),
        ).astype(np.float64)

        # Proprioception: every non-world body's frame relative to the root
        # (see MocapTrackingEnv._post_init for the rationale).
        free_jnts = np.nonzero(
            mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )[0]
        assert len(free_jnts) == 1, "expected exactly one free (root) joint"
        self._root_body_id = int(mj_model.jnt_bodyid[free_jnts[0]])
        self._proprio_body_ids = np.arange(1, mj_model.nbody)

        # First-order target-filter EMA weight (mirror _post_init): alpha =
        # exp(-ctrl_dt / tc), the weight on the previous filtered command.
        # This env runs torque actuation only, so the reset filter state is
        # zero (the hold-pose command is only needed for position servos).
        tc = float(config.get("target_filter_tc", 0.0))
        self._filter_alpha = (
            float(np.exp(-float(config.ctrl_dt) / tc)) if tc > 0 else 0.0
        )

        # Reference data stays float32 to match the GPU env's on-device arrays.
        self._ref_qpos = np.asarray(dataset["qpos"], dtype=np.float32)
        self._ref_qvel = np.asarray(dataset["qvel"], dtype=np.float32)
        self._ref_body_pos = np.asarray(dataset["body_pos"], dtype=np.float32)
        self._clip_starts = np.asarray(dataset["clip_starts"], dtype=np.int64)
        self._clip_lengths = np.asarray(dataset["clip_lengths"], dtype=np.int64)
        self._num_clips = len(self._clip_starts)

        nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
        # Root orientation is the 6D rep (6, not the raw 4-quat), plus the two
        # binary foot-contact flags and the per-body proprioception block
        # (nbody-1 bodies x [3 root-relative pos + 6 orientation-6d]). Each
        # look-ahead frame carries the reference delta (joints, jvel, rot6d, h).
        n_proprio = (mj_model.nbody - 1) * 9
        self._obs_size = (
            (nq - 7) + (nv - 6) + 6 + 3 + 1 + 3 + 2 + n_proprio + nu
            + int(config.look_ahead) * ((nq - 7) + (nv - 6) + 6 + 1)
        )
        # Minimal space stand-ins (shape/low/high are all the adapter reads);
        # avoids a gymnasium dependency for what is a pure-numpy pool.
        self.observation_space = SimpleNamespace(shape=(self._obs_size,))
        self.action_space = SimpleNamespace(
            shape=(nu,),
            low=self._lowers.astype(np.float32),
            high=self._uppers.astype(np.float32),
        )

        # Tracking-collapse floor is static config, precomputed once (mirrors
        # the trace-time constant in the GPU env).
        cfg = config.reward_config
        rt = config.reward_termination
        self._max_tracking = cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root
        self._track_floor = (
            rt.min_tracking_frac * self._max_tracking if rt.enabled else None
        )

        self._datas = [mujoco.MjData(mj_model) for _ in range(self._num_envs)]
        self._rngs = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(seed).spawn(self._num_envs)
        ]

        self._phase_idx = np.zeros(self._num_envs, dtype=np.int64)
        self._clip_start = np.zeros(self._num_envs, dtype=np.int64)
        self._clip_len = np.ones(self._num_envs, dtype=np.int64)
        self._last_act = np.zeros((self._num_envs, nu), dtype=np.float64)
        # Post-filter applied command carried across steps (mirror info
        # ["filtered_ctrl"]); starts at zero force (torque-mode hold command).
        self._filtered_ctrl = np.zeros((self._num_envs, nu), dtype=np.float64)
        self._step_count = np.zeros(self._num_envs, dtype=np.int64)

        if num_threads is None:
            num_threads = min(self._num_envs, os.cpu_count() or 1)
        self._num_threads = max(1, int(num_threads))
        self._env_chunks = [
            chunk for chunk in
            np.array_split(np.arange(self._num_envs), self._num_threads)
            if len(chunk)
        ]
        self._executor = (
            ThreadPoolExecutor(max_workers=self._num_threads)
            if self._num_threads > 1 else None
        )

    # -- per-env logic (mirrors MocapTrackingEnv) ----------------------------

    def _reset_env(self, i: int) -> None:
        cfg = self._config
        rng = self._rngs[i]

        clip_idx = int(rng.integers(self._num_clips))
        clip_start = int(self._clip_starts[clip_idx])
        clip_len = int(self._clip_lengths[clip_idx])

        # Mirror reset(): non-cyclic clips keep the random start far enough
        # from the end that the look_ahead horizon stays inside the clip.
        if cfg.cyclic:
            start_high = clip_len
        else:
            start_high = max(clip_len - int(cfg.look_ahead), 1)
        start_idx = int(rng.integers(start_high)) if cfg.random_start else 0

        abs_idx = clip_start + start_idx
        noise = cfg.reset_noise_scale
        qpos = self._ref_qpos[abs_idx] + noise * rng.standard_normal(
            self._model.nq
        )
        qvel = self._ref_qvel[abs_idx] + noise * rng.standard_normal(
            self._model.nv
        )

        d = self._datas[i]
        mujoco.mj_resetData(self._model, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        # The proprioception and foot-contact obs terms read forward-kinematics
        # outputs (xpos/xmat/sensordata), so a forward pass is now required at
        # reset. But native mj_forward normalizes the noisy root quat in place,
        # while MJX (mjx.forward in reset) leaves qpos as sampled and only
        # normalizes the derived xquat — so we restore the raw quat afterwards.
        # This keeps the qpos-derived obs terms (root rot6d, ref deltas) bit-for
        # -bit with MJX while xpos/xmat come from the same normalized xquat both
        # backends' kinematics use.
        q_root = d.qpos[3:7].copy()
        mujoco.mj_forward(self._model, d)
        d.qpos[3:7] = q_root

        self._phase_idx[i] = start_idx
        self._clip_start[i] = clip_start
        self._clip_len[i] = clip_len
        self._last_act[i] = 0.0
        self._filtered_ctrl[i] = 0.0
        self._step_count[i] = 0

    # -- vectorized obs/reward (batched over ALL envs at once) ---------------
    #
    # The heavy per-frame math (obs assembly, reward kernels) runs ONCE on
    # stacked (num_envs, ...) arrays rather than in a per-env Python loop. This
    # is the key to CPU scaling: the per-env loop held the GIL for every small
    # numpy op x num_envs, so the thread pool could never exceed ~3 cores no
    # matter how many threads. Only mj_step / mj_forward stay per-env in the
    # worker threads (they release the GIL, so they parallelize); the batched
    # numpy below collapses hundreds of GIL-held dispatches into a handful of
    # large C ops. The results are bit-for-bit identical to the old per-env
    # code, which is why the single-env adapters (`_get_obs` / `_get_reward`,
    # used by check_envpool_parity.py) just call these with a 1-row slice.

    def _obs_batch(
        self, qpos, qvel, xpos, xmat, sensordata,
        phase_idx, clip_start, clip_len, last_act,
    ) -> np.ndarray:
        """Assemble the (N, obs_size) observation from stacked per-env state."""
        n = qpos.shape[0]
        steps = np.arange(1, int(self._config.look_ahead) + 1)  # (la,)
        future_local = (phase_idx[:, None] + steps[None, :]) % clip_len[:, None]
        future_abs = clip_start[:, None] + future_local          # (N, la)

        ref_qpos = self._ref_qpos[future_abs]  # (N, la, nq)
        ref_qvel = self._ref_qvel[future_abs]  # (N, la, nv)

        d_joints = ref_qpos[:, :, 7:] - qpos[:, None, 7:]
        d_jvel = ref_qvel[:, :, 6:] - qvel[:, None, 6:]
        d_rot6d = _quat_diff_6d_np(ref_qpos[:, :, 3:7], qpos[:, None, 3:7])
        d_height = ref_qpos[:, :, 2:3] - qpos[:, None, 2:3]
        ref_delta = np.concatenate(
            [d_joints, d_jvel, d_rot6d, d_height], axis=-1
        ).reshape(n, -1)

        # Proprioception: each non-world body's frame relative to the root.
        root_pos = xpos[:, self._root_body_id]                   # (N, 3)
        body_pos = (
            xpos[:, self._proprio_body_ids] - root_pos[:, None, :]
        ).reshape(n, -1)
        body_rot6d = _mat_to_rot6d_np(
            xmat[:, self._proprio_body_ids]
        ).reshape(n, -1)

        touch = sensordata[:, self._foot_sensor_adr]             # (N, 2, 2)
        feet = (
            np.sum(touch, axis=-1) > self._foot_contact_force_thresh
        ).astype(np.float64)

        return np.concatenate([
            qpos[:, 7:],
            qvel[:, 6:],
            _quat_to_rot6d_np(qpos[:, 3:7]),
            qvel[:, 3:6],
            qpos[:, 2:3],
            qvel[:, 0:3],
            feet,
            body_pos,
            body_rot6d,
            last_act,
            ref_delta,
        ], axis=1).astype(np.float32)

    def _reward_batch(self, qpos, qvel, xpos, afrc, abs_idx, action, last_action):
        """Reward + components for all envs from stacked per-env state.

        Returns (total (N,), tracking (N,), root_dist (N,), components) where
        each component is an (N,) array."""
        cfg = self._config.reward_config
        ref_qpos = self._ref_qpos[abs_idx]         # (N, nq)
        ref_qvel = self._ref_qvel[abs_idx]         # (N, nv)
        ref_body_pos = self._ref_body_pos[abs_idx]  # (N, nbody, 3)

        pose_err = np.sum(np.square(qpos[:, 7:] - ref_qpos[:, 7:]), axis=1)
        r_pose = np.exp(-pose_err / cfg.sigma_pose)

        vel_err = np.sum(np.square(qvel[:, 6:] - ref_qvel[:, 6:]), axis=1)
        r_vel = np.exp(-vel_err / cfg.sigma_vel)

        ee_pos = xpos[:, self._ee_body_ids]        # (N, n_ee, 3)
        ref_ee_pos = ref_body_pos[:, self._ee_body_ids]
        ee_err = np.sum(np.square(ee_pos - ref_ee_pos), axis=(1, 2))
        r_ee = np.exp(-ee_err / cfg.sigma_ee)

        root_pos_err = np.sum(np.square(qpos[:, :3] - ref_qpos[:, :3]), axis=1)
        root_quat_err = _quaternion_distance_np(qpos[:, 3:7], ref_qpos[:, 3:7])
        root_err = root_pos_err + root_quat_err
        r_root = np.exp(-root_err / cfg.sigma_root)
        root_dist = np.sqrt(root_pos_err)

        # Effort penalty on the ACTUAL actuator force normalized by each
        # actuator's strength limit (mirror MocapTrackingEnv): in torque mode
        # actuator_force = gear*ctrl and limit = |gear|, so this equals the
        # mean-square applied command.
        effort = afrc / self._force_limit
        r_torque = -cfg.w_torque * np.mean(np.square(effort), axis=1)

        # Action-rate penalty on the RAW policy output (pre-scale, pre-filter).
        r_action_rate = -cfg.w_action_rate * np.mean(
            np.square(action - last_action), axis=1
        )

        tracking = (
            cfg.w_pose * r_pose
            + cfg.w_vel * r_vel
            + cfg.w_ee * r_ee
            + cfg.w_root * r_root
        )
        components = {
            "reward/pose": r_pose,
            "reward/vel": r_vel,
            "reward/ee": r_ee,
            "reward/root": r_root,
            "reward/torque": r_torque,
            "reward/action_rate": r_action_rate,
            "root_dist": root_dist,
        }
        return (
            tracking + cfg.w_alive + r_torque + r_action_rate,
            tracking,
            root_dist,
            components,
        )

    # -- single-env adapters (used only by check_envpool_parity.py) ----------

    def _get_obs(self, i: int) -> np.ndarray:
        d = self._datas[i]
        return self._obs_batch(
            d.qpos[None], d.qvel[None], d.xpos[None], d.xmat[None],
            d.sensordata[None], self._phase_idx[i:i + 1],
            self._clip_start[i:i + 1], self._clip_len[i:i + 1],
            self._last_act[i][None],
        )[0]

    def _get_reward(self, d, abs_idx, action, last_action):
        total, tracking, root_dist, comps = self._reward_batch(
            d.qpos[None], d.qvel[None], d.xpos[None], d.actuator_force[None],
            np.array([abs_idx]), np.asarray(action)[None],
            np.asarray(last_action)[None],
        )
        return (
            float(total[0]), float(tracking[0]), float(root_dist[0]),
            {k: float(v[0]) for k, v in comps.items()},
        )

    def _advance_env(self, i: int, action: np.ndarray) -> None:
        """Apply the (filtered) command and step one env's physics + phase.

        The only per-env work left in the hot path: mj_step releases the GIL, so
        running this across the thread pool is what actually uses the cores."""
        cfg = self._config
        d = self._datas[i]
        ctrl = np.clip(action * cfg.action_scale, self._lowers, self._uppers)
        # First-order target-filter smoothing (mirror MocapTrackingEnv.step):
        # blend the previous applied command in, then carry the result forward.
        if self._filter_alpha > 0.0:
            ctrl = (
                self._filter_alpha * self._filtered_ctrl[i]
                + (1.0 - self._filter_alpha) * ctrl
            )
        self._filtered_ctrl[i] = ctrl
        d.ctrl[:] = ctrl
        mujoco.mj_step(self._model, d, nstep=self._n_substeps)
        self._phase_idx[i] = (self._phase_idx[i] + 1) % self._clip_len[i]

    # -- pool protocol (what EnvPoolWrapper consumes) ------------------------

    def _run_chunked(self, worker) -> None:
        if self._executor is None:
            for chunk in self._env_chunks:
                worker(chunk)
        else:
            # list() re-raises worker exceptions instead of dropping them.
            list(self._executor.map(worker, self._env_chunks))

    def _reset_many(self, idxs: np.ndarray) -> None:
        """Reset the given env indices (auto-reset of done envs), threaded."""
        if len(idxs) == 0:
            return
        if self._executor is None or len(idxs) == 1:
            for i in idxs:
                self._reset_env(int(i))
            return
        chunks = [
            c for c in np.array_split(idxs, min(len(idxs), self._num_threads))
            if len(c)
        ]
        list(self._executor.map(
            lambda ch: [self._reset_env(int(i)) for i in ch], chunks
        ))

    def _gather_obs(self) -> np.ndarray:
        """Stack current per-env state and assemble the batched observation."""
        d = self._datas
        return self._obs_batch(
            np.stack([x.qpos for x in d]),
            np.stack([x.qvel for x in d]),
            np.stack([x.xpos for x in d]),
            np.stack([x.xmat for x in d]),
            np.stack([x.sensordata for x in d]),
            self._phase_idx, self._clip_start, self._clip_len, self._last_act,
        )

    def reset(self) -> tuple[np.ndarray, dict]:
        def worker(chunk):
            for i in chunk:
                self._reset_env(i)

        self._run_chunked(worker)
        return self._gather_obs(), {}

    def step(self, actions: Any) -> tuple[np.ndarray, ...]:
        actions = np.asarray(actions, dtype=np.float64)
        cfg = self._config

        # 1. Physics + phase advance, per env across the thread pool (mj_step
        #    releases the GIL, so this is where the cores get used).
        def worker(chunk):
            for i in chunk:
                self._advance_env(i, actions[i])

        self._run_chunked(worker)

        # 2. Reward + termination from the post-step state, batched over envs.
        d = self._datas
        qpos = np.stack([x.qpos for x in d])
        qvel = np.stack([x.qvel for x in d])
        xpos = np.stack([x.xpos for x in d])
        afrc = np.stack([x.actuator_force for x in d])
        abs_idx = self._clip_start + self._phase_idx
        reward, tracking, root_dist, comps = self._reward_batch(
            qpos, qvel, xpos, afrc, abs_idx, actions, self._last_act,
        )

        nan_check = np.isnan(qpos).any(axis=1) | np.isnan(qvel).any(axis=1)
        low_track = (
            tracking < self._track_floor if self._track_floor is not None
            else np.zeros(self._num_envs, dtype=bool)
        )
        rrt = self._config.root_termination
        root_too_far = (
            root_dist > rrt.max_dist if rrt.enabled
            else np.zeros(self._num_envs, dtype=bool)
        )
        terminated = (
            (nan_check | low_track | root_too_far) if cfg.early_termination
            else nan_check
        )
        clip_truncated = (
            (self._phase_idx >= self._clip_len - int(cfg.look_ahead))
            if not cfg.cyclic else np.zeros(self._num_envs, dtype=bool)
        )
        self._step_count += 1
        step_truncated = self._step_count >= int(cfg.episode_length)

        # Mirror TerminationWrapper: clip-end truncation clears the termination
        # flag; the step-limit does not (a genuine fall at the limit is terminal).
        terminated_final = terminated & ~clip_truncated
        truncated_final = clip_truncated | step_truncated
        done = terminated | clip_truncated | step_truncated

        # 3. last_act obs field = the current action (mirror step's info update),
        #    then auto-reset done envs (which zeroes their last_act/filter/phase).
        self._last_act = actions.copy()
        self._reset_many(np.nonzero(done)[0])

        # 4. Observation from the post-reset state, batched over envs.
        obs = self._gather_obs()
        metrics = {k: comps[k].astype(np.float32) for k in _METRIC_KEYS}
        return (
            obs, reward.astype(np.float32), terminated_final, truncated_final,
            {"metrics": metrics},
        )


def build_mocap_envpool_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for the CPU EnvPool-style mocap env (see ``env.builder``).

    Reads from cfg_env (beyond the shared config/reward groups):
      parallel_envs (int):   training pool size; throughput saturates around
                             the physical core count, unlike the GPU backends.
      num_threads (int?):    stepping threads; null = min(parallel_envs, cores).
      test_episodes (int):   eval pool size; must match trainer.test_episodes.
      seed, clip_ids, collisions: as in the GPU builder.

    ``gpu_clip_budget`` and the Warp budgets (naconmax/njmax/naccdmax,
    graph_mode) do not apply: native MuJoCo allocates contacts dynamically and
    the full clip dataset lives in host RAM.

    In ``"play"`` mode this delegates to the MJX builder (impl="jax"): playback
    is a single jitted world driving the interactive viewer, which the CPU pool
    protocol doesn't fit — and checkpoints are agent-side, so they replay
    identically on any backend.
    """
    if mode == "play":
        from omegaconf import OmegaConf

        from examples.mocap.loader import build_mocap_env

        return build_mocap_env(
            OmegaConf.merge(cfg_env, {"impl": "jax"}), mode="play"
        )

    from examples.mocap.cmu_mocap_data import build_cmu_humanoid, load_cmu_clips
    from examples.mocap.loader import build_env_config

    config = build_env_config(cfg_env)

    mj_model, _ = build_cmu_humanoid()
    mj_model.opt.timestep = config.sim_dt
    _configure_collisions(mj_model, resolve_collision_mode(cfg_env))

    clip_ids = list(cfg_env.clip_ids) if cfg_env.get("clip_ids") else None
    dataset = load_cmu_clips(mj_model, clip_ids=clip_ids, ctrl_dt=config.ctrl_dt)

    seed = int(cfg_env.get("seed", 0))
    num_threads = cfg_env.get("num_threads", None)
    test_episodes = int(cfg_env.get("test_episodes", 5))

    train_pool = MocapCpuPool(
        mj_model, dataset, config,
        num_envs=int(cfg_env.parallel_envs),
        seed=seed,
        num_threads=num_threads,
    )
    # Deterministic eval config (mirrors the GPU loader): disable the stochastic
    # reset knobs so every test episode starts at frame 0 with no state noise —
    # otherwise even a single-clip run reports nonzero test std for a
    # deterministic policy. The training pool keeps the original config.
    import copy

    eval_config = copy.deepcopy(config)
    eval_config.random_start = False
    eval_config.reset_noise_scale = 0.0
    test_pool = MocapCpuPool(
        mj_model, dataset, eval_config,
        num_envs=test_episodes,
        seed=seed + 1,
        num_threads=num_threads,
    )

    max_steps = int(config.episode_length)
    return EnvBundle(
        env=EnvPoolWrapper(train_pool, max_episode_steps=max_steps),
        test_env=EnvPoolWrapper(test_pool, max_episode_steps=max_steps),
        env_cfg=None,
    )
