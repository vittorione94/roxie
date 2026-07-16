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

from examples.mocap.mocap_tracking import CMU_BODY_NAMES, _configure_collisions
from roxie.environment.envpool_adapter import EnvPoolWrapper
from roxie.environment.loader import EnvBundle

# Metric keys match the GPU env's `metrics` dict so epoch logs (CSV/wandb)
# line up column-for-column across backends.
_METRIC_KEYS = (
    "reward/pose", "reward/vel", "reward/ee", "reward/root", "reward/torque",
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

        # Reference data stays float32 to match the GPU env's on-device arrays.
        self._ref_qpos = np.asarray(dataset["qpos"], dtype=np.float32)
        self._ref_qvel = np.asarray(dataset["qvel"], dtype=np.float32)
        self._ref_body_pos = np.asarray(dataset["body_pos"], dtype=np.float32)
        self._clip_starts = np.asarray(dataset["clip_starts"], dtype=np.int64)
        self._clip_lengths = np.asarray(dataset["clip_lengths"], dtype=np.int64)
        self._num_clips = len(self._clip_starts)

        nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
        self._obs_size = (
            (nq - 7) + (nv - 6) + 4 + 3 + 1 + 3 + nu
            + int(config.look_ahead) * ((nq - 7) + (nv - 6) + 4 + 1)
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
        # No mj_forward here: the reset obs reads only qpos/qvel, and native
        # mj_forward would normalize the noisy root quat in place — MJX leaves
        # it as sampled, so forwarding would make the two backends' reset obs
        # differ by O(reset_noise_scale). mj_step runs the full pipeline anyway.

        self._phase_idx[i] = start_idx
        self._clip_start[i] = clip_start
        self._clip_len[i] = clip_len
        self._last_act[i] = 0.0
        self._step_count[i] = 0

    def _get_obs(self, i: int) -> np.ndarray:
        d = self._datas[i]
        qpos, qvel = d.qpos, d.qvel
        clip_len = self._clip_len[i]

        steps = np.arange(1, int(self._config.look_ahead) + 1)
        future_local = (self._phase_idx[i] + steps) % clip_len
        future_abs = self._clip_start[i] + future_local

        ref_qpos = self._ref_qpos[future_abs]  # (look_ahead, nq)
        ref_qvel = self._ref_qvel[future_abs]  # (look_ahead, nv)

        d_joints = ref_qpos[:, 7:] - qpos[7:]
        d_jvel = ref_qvel[:, 6:] - qvel[6:]
        d_quat = ref_qpos[:, 3:7] - qpos[3:7]
        d_height = ref_qpos[:, 2:3] - qpos[2:3]

        ref_delta = np.concatenate(
            [d_joints, d_jvel, d_quat, d_height], axis=-1
        ).reshape(-1)

        return np.concatenate([
            qpos[7:],
            qvel[6:],
            qpos[3:7],
            qvel[3:6],
            qpos[2:3],
            qvel[0:3],
            self._last_act[i],
            ref_delta,
        ]).astype(np.float32)

    def _get_reward(
        self, d: mujoco.MjData, abs_idx: int, ctrl: np.ndarray
    ) -> tuple[float, float, dict[str, float]]:
        cfg = self._config.reward_config
        ref_qpos = self._ref_qpos[abs_idx]
        ref_qvel = self._ref_qvel[abs_idx]
        ref_body_pos = self._ref_body_pos[abs_idx]

        pose_err = np.sum(np.square(d.qpos[7:] - ref_qpos[7:]))
        r_pose = np.exp(-pose_err / cfg.sigma_pose)

        vel_err = np.sum(np.square(d.qvel[6:] - ref_qvel[6:]))
        r_vel = np.exp(-vel_err / cfg.sigma_vel)

        ee_pos = d.xpos[self._ee_body_ids]
        ref_ee_pos = ref_body_pos[self._ee_body_ids]
        ee_err = np.sum(np.square(ee_pos - ref_ee_pos))
        r_ee = np.exp(-ee_err / cfg.sigma_ee)

        root_pos_err = np.sum(np.square(d.qpos[:3] - ref_qpos[:3]))
        quat_dot = np.dot(d.qpos[3:7], ref_qpos[3:7])
        root_err = root_pos_err + 1.0 - np.square(quat_dot)
        r_root = np.exp(-root_err / cfg.sigma_root)

        r_torque = -cfg.w_torque * np.mean(np.square(ctrl))

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
        }
        return tracking + cfg.w_alive + r_torque, tracking, components

    def _step_env(self, i: int, action: np.ndarray):
        cfg = self._config
        d = self._datas[i]

        ctrl = np.clip(action * cfg.action_scale, self._lowers, self._uppers)
        d.ctrl[:] = ctrl
        mujoco.mj_step(self._model, d, nstep=self._n_substeps)

        clip_len = self._clip_len[i]
        self._phase_idx[i] = (self._phase_idx[i] + 1) % clip_len
        abs_idx = int(self._clip_start[i] + self._phase_idx[i])

        reward, tracking, components = self._get_reward(d, abs_idx, ctrl)

        nan_check = bool(np.isnan(d.qpos).any() or np.isnan(d.qvel).any())
        low_track = (
            self._track_floor is not None and tracking < self._track_floor
        )
        terminated = (
            (nan_check or low_track) if cfg.early_termination else nan_check
        )

        clip_truncated = (not cfg.cyclic) and (
            self._phase_idx[i] >= clip_len - int(cfg.look_ahead)
        )
        self._step_count[i] += 1
        step_truncated = self._step_count[i] >= int(cfg.episode_length)

        # Mirror TerminationWrapper: env-internal (clip-end) truncation clears
        # the termination flag; the step-limit does not (a genuine fall at the
        # limit stays terminal).
        terminated_final = terminated and not clip_truncated
        truncated_final = clip_truncated or step_truncated

        self._last_act[i] = action

        if terminated or clip_truncated or step_truncated:
            # Same-step auto-reset: the returned obs starts the next episode.
            self._reset_env(i)

        return self._get_obs(i), reward, terminated_final, truncated_final, components

    # -- pool protocol (what EnvPoolWrapper consumes) ------------------------

    def _run_chunked(self, worker) -> None:
        if self._executor is None:
            for chunk in self._env_chunks:
                worker(chunk)
        else:
            # list() re-raises worker exceptions instead of dropping them.
            list(self._executor.map(worker, self._env_chunks))

    def reset(self) -> tuple[np.ndarray, dict]:
        obs = np.empty((self._num_envs, self._obs_size), dtype=np.float32)

        def worker(chunk):
            for i in chunk:
                self._reset_env(i)
                obs[i] = self._get_obs(i)

        self._run_chunked(worker)
        return obs, {}

    def step(self, actions: Any) -> tuple[np.ndarray, ...]:
        actions = np.asarray(actions, dtype=np.float64)
        n = self._num_envs
        obs = np.empty((n, self._obs_size), dtype=np.float32)
        reward = np.empty(n, dtype=np.float32)
        terminated = np.empty(n, dtype=bool)
        truncated = np.empty(n, dtype=bool)
        metrics = {k: np.empty(n, dtype=np.float32) for k in _METRIC_KEYS}

        def worker(chunk):
            for i in chunk:
                o, r, term, trunc, comps = self._step_env(i, actions[i])
                obs[i] = o
                reward[i] = r
                terminated[i] = term
                truncated[i] = trunc
                for k in _METRIC_KEYS:
                    metrics[k][i] = comps[k]

        self._run_chunked(worker)
        return obs, reward, terminated, truncated, {"metrics": metrics}


def build_mocap_envpool_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for the CPU EnvPool-style mocap env (see ``env.builder``).

    Reads from cfg_env (beyond the shared config/reward groups):
      parallel_envs (int):   training pool size; throughput saturates around
                             the physical core count, unlike the GPU backends.
      num_threads (int?):    stepping threads; null = min(parallel_envs, cores).
      test_episodes (int):   eval pool size; must match trainer.test_episodes.
      seed, clip_ids, self_collisions: as in the GPU builder.

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
    _configure_collisions(mj_model, cfg_env.get("self_collisions", True))

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
    test_pool = MocapCpuPool(
        mj_model, dataset, config,
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
