"""CPU EnvPool-style mirror of the CMU mocap-tracking task.

This is a faithful re-implementation of ``MocapTrackingEnv`` (+ its
``TerminationWrapper``) on native MuJoCo (``mujoco.mj_step``) and numpy,
exposed through the same pool protocol as EnvPool's gymnasium API so the
Trainer's CPU loop (``_run_envpool``) drives it unchanged. Its purpose is a
clean CPU-vs-GPU comparison against the MJX ("jax") and Warp backends: same
observations, rewards, termination/truncation semantics and config — only the
physics backend and the vectorization strategy differ.

Semantics mirrored from the GPU path (see ``mocap_tracking.py``):
  - actuation mode (torque motors / dm_control PD position servos) and the
    reset filter state that holds the reset pose under position control;
  - obs layout, look-ahead reference deltas, reset noise, random clip/start,
    and the negative-mining start distribution;
  - reward components (pose/vel/ee/root_pos/root_quat/root_vel/torque +
    alive), the retained pre-split ``reward/root`` logging term, the
    termination-cause indicators and all their metric keys;
  - termination = NaN | tracking collapse | root drift; truncation = clip end
    (look_ahead guard) | episode step-limit, with clip-end truncation clearing
    the termination flag exactly like ``TerminationWrapper`` does.

``examples/mocap/check_envpool_parity.py`` is the executable statement of that
contract — extend it whenever a term is added on either side.

Auto-reset is same-step (EnvPool convention): the obs returned on a done step
is the *reset* obs. This matches the JAX path, whose trainer acts from the
auto-reset state, so the entry following a done in the trajectory buffer is the
reset obs on both backends. The true final obs differs in exactly one place —
it feeds the observation-normalization statistics on the JAX path, while the
reset obs feeds them here.

Parallelism: two interchangeable steppers, selected by ``env.stepper``.

``rollout`` (default) hands the whole batch to ``mujoco.rollout``, which steps it
in MuJoCo's own C++ thread pool and returns the post-step state and sensordata as
batched arrays. Physics, the per-env control filtering and the state capture all
collapse into that one call. Because rollout exposes only ``state`` and
``sensordata``, the fields the obs/reward read off MjData (``xpos``, ``xmat``,
``actuator_force``) are routed through added sensors — see
``cmu_mocap_data.add_rollout_sensors``.

``threads`` is the original stepper: one ``MjData`` per env driven by a Python
``ThreadPoolExecutor`` (MuJoCo's bindings release the GIL inside ``mj_step``, so
threads scale across cores without pickling). Kept as the reference
implementation and the fallback; the two agree bit-for-bit on obs, reward and
metrics apart from a ~1e-9 solver drift (see ``check_envpool_parity.py``).

Measured on the 12-core 7900X at 1000 envs: 14.2k sps threaded, 34.3k rollout.

Unlike the GPU backends there are no contact budgets to size (native MuJoCo
allocates contacts dynamically) and no GPU clip budget — the full clip dataset
always lives in host RAM.

Usage via experiment YAML::

    env:
      builder: examples.mocap.mocap_envpool.build_mocap_envpool_env
      parallel_envs: 20
      num_threads: null   # default: min(parallel_envs, cpu cores)
      stepper: rollout    # or "threads" for the per-env mj_step loop
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from typing import Any, Optional

import mujoco
import numpy as np
from ml_collections import config_dict
from mujoco import rollout as mj_rollout

from examples.mocap.mocap_tracking import (
    CMU_BODY_NAMES,
    FOOT_TOUCH_SENSORS,
    _configure_actuation,
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
# `metrics` dict field-for-field: the reward kernels (including the split root
# terms and the retained pre-split `reward/root`), the two penalties, the
# root-drift diagnostic and the three termination-cause indicators.
_METRIC_KEYS = (
    "reward/pose", "reward/vel", "reward/ee", "reward/root",
    "reward/root_pos", "reward/root_quat", "reward/root_vel",
    "reward/torque", "reward/action_rate", "root_dist",
    "term/nan", "term/tracking", "term/root",
)


# Work chunks handed to the thread pool per worker thread. >1 gives the
# executor something to load-balance with (see the note in __init__); the value
# is a throughput knob, not a semantic one.
_CHUNKS_PER_THREAD = 2

# Workers the observation assembly is split across, and the pool size below
# which it stays single-threaded. See _gather_obs.
_OBS_WORKERS = 6
_OBS_THREAD_MIN_ENVS = 256

# The state vector mujoco.rollout round-trips: (time, qpos, qvel, act).
_FULL_PHYSICS = mujoco.mjtState.mjSTATE_FULLPHYSICS
_CTRL_SPEC = int(mujoco.mjtState.mjSTATE_CTRL)


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
        actuation: str = "torque",
        reset_pool_size: int = 0,
        rollout_model: Optional[mujoco.MjModel] = None,
        rollout_addrs: Optional[tuple] = None,
    ):
        self._model = mj_model
        self._config = config
        # Physics stepper. `rollout_model` is the sensor-augmented twin of
        # `mj_model` (see cmu_mocap_data.add_rollout_sensors); when given, the
        # per-env `mj_step` loop and the `_capture` gather are replaced by one
        # `mujoco.rollout` call. `self._model` stays the BASE model throughout —
        # obs sizing, foot-sensor addresses and the reward all read it — and the
        # augmented model is used only where physics is actually stepped, which
        # is `self._sim_model`. The two are bit-identical dynamically; the
        # augmented one merely reports more sensors.
        self._rollout_model = rollout_model
        self._sim_model = rollout_model if rollout_model is not None else mj_model
        self._use_rollout = rollout_model is not None
        self._num_envs = int(num_envs)
        self._n_substeps = int(round(config.ctrl_dt / config.sim_dt))
        # Mirrors MocapTrackingEnv._actuation. The MODEL is converted by the
        # builder (_configure_actuation rewrites the actuator arrays before any
        # MjData exists); this flag only selects the reset filter state, exactly
        # as it does on the GPU side.
        self._actuation = actuation

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
        tc = float(config.get("target_filter_tc", 0.0))
        self._filter_alpha = (
            float(np.exp(-float(config.ctrl_dt) / tc)) if tc > 0 else 0.0
        )
        # ctrl -> joint-angle inverse map, for initializing the filter state to
        # a pose-holding command at reset under position actuation (mirror
        # _post_init / reset): the actuated joint's qpos address, range low and
        # half-range per actuator.
        jid = mj_model.actuator_trnid[:, 0]
        self._act_qadr = mj_model.jnt_qposadr[jid]
        self._act_q_lo = mj_model.jnt_range[jid, 0].astype(np.float64)
        self._act_slope = (
            (mj_model.jnt_range[jid, 1] - mj_model.jnt_range[jid, 0]) / 2.0
        ).astype(np.float64)

        # Reference data stays float32 to match the GPU env's on-device arrays.
        self._ref_qpos = np.asarray(dataset["qpos"], dtype=np.float32)
        self._ref_qvel = np.asarray(dataset["qvel"], dtype=np.float32)
        self._ref_body_pos = np.asarray(dataset["body_pos"], dtype=np.float32)
        self._clip_starts = np.asarray(dataset["clip_starts"], dtype=np.int64)
        self._clip_lengths = np.asarray(dataset["clip_lengths"], dtype=np.int64)
        self._num_clips = len(self._clip_starts)

        nq, nv, nu = mj_model.nq, mj_model.nv, mj_model.nu
        # Term-for-term with MocapTrackingEnv._get_obs: root position (3), root
        # orientation as the 6D rep (6, not the raw 4-quat), root twist (6),
        # joint pos/vel, the two binary foot-contact flags, the per-body
        # proprioception block (nbody-1 bodies x [3 root-relative pos + 6
        # orientation-6d]), the last raw action, and one reference-delta block
        # per look-ahead frame (joints, jvel, rot6d, root position).
        n_proprio = (mj_model.nbody - 1) * 9
        self._obs_size = (
            3 + 6 + 6 + (nq - 7) + (nv - 6) + 2 + n_proprio + nu
            + int(config.look_ahead) * ((nq - 7) + (nv - 6) + 6 + 3)
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
        # the trace-time constant in the GPU env). Uses the SPLIT root weights,
        # like _get_termination: `w_root` is a logging-only leftover and must
        # not enter the floor, even though the two happen to sum alike today.
        cfg = config.reward_config
        rt = config.reward_termination
        self._max_tracking = (
            cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root_pos + cfg.w_root_quat
        )
        self._track_floor = (
            rt.min_tracking_frac * self._max_tracking if rt.enabled else None
        )

        # --- Negative mining over start phases (mirror _post_init) ---
        mining = config.get("negative_mining", None)
        self._mining_enabled = bool(
            mining is not None and mining.get("enabled", False)
        )
        if self._mining_enabled:
            self._mining_bins = int(mining.get("bins", 64))
            self._mining_alpha = float(mining.get("alpha", 0.5))
            self._mining_ema = float(mining.get("ema", 0.8))
            self._mining_lead_in = int(mining.get("lead_in", 0))
        else:
            self._mining_bins = 0
            self._mining_alpha = 0.0
            self._mining_ema = 0.0
            self._mining_lead_in = 0
        # Start-phase weights the pool draws from, owned by the trainer on the
        # GPU path (a traced reset argument) but by the pool here — the CPU
        # reset is plain Python, so there is nothing to keep out of a trace.
        # `set_mining_weights` is how the trainer pushes each epoch's refresh.
        self._mining_weights = (
            np.full(self._mining_bins, 1.0 / self._mining_bins)
            if self._mining_enabled else None
        )
        self._mining_visit = np.zeros(max(self._mining_bins, 1))
        self._mining_fail = np.zeros(max(self._mining_bins, 1))

        # One MjData per env is the threaded stepper's whole state. The rollout
        # stepper keeps its state in a plain (num_envs, nstate) array instead and
        # only needs one scratch MjData per THREAD, so the per-env allocation is
        # skipped entirely — at 4000 envs that is 4000 MjData not built.
        self._datas = (
            [] if self._use_rollout
            else [mujoco.MjData(mj_model) for _ in range(self._num_envs)]
        )
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

        # Auto-reset draws from a precomputed pool of reset states rather than
        # building each one on demand, exactly as _run_jax does (`reset_pool` +
        # a gather in `_step_and_autoreset`). Two reasons, and the parity one
        # came first: on the GPU path an auto-reset IS a gather from a finite
        # pool regenerated once per epoch, so sampling fresh states here was a
        # genuine behavioural difference. It is also where the CPU time went —
        # an on-demand reset needs `mj_forward` (~133 us, comparable to a whole
        # control step) purely to produce the observation, and early in training
        # ~20-25% of envs reset every step, which measured as roughly half of
        # `env.step`. Pooled, a reset is a memcpy: the forward was already paid
        # once when the pool was built. 0 disables pooling (the eval pool wants
        # its exact deterministic reset, and `_test`/`v_test_reset` on the GPU
        # side does not use the pool either).
        self._reset_pool_size = int(reset_pool_size)
        self._reset_pool = None
        self._pool_rng = np.random.default_rng(seed + 7919)
        # Envs whose MjData still has to be restored from the pool; consumed by
        # `_advance_env` on the worker thread at the start of the next step.
        self._needs_reset = np.zeros(self._num_envs, dtype=bool)

        # Preallocated (num_envs, ...) mirrors of the MjData fields the obs and
        # reward read, filled by `_capture`. This is what lets `step` skip the
        # `np.stack([x.qpos for x in datas])`-style gathers entirely: those ran
        # on the main thread and at 1000 envs were the largest non-physics item
        # in the profile after the resets.
        self._b_qpos = np.zeros((self._num_envs, nq))
        self._b_qvel = np.zeros((self._num_envs, nv))
        self._b_xpos = np.zeros((self._num_envs, mj_model.nbody, 3))
        self._b_xmat = np.zeros((self._num_envs, mj_model.nbody, 9))
        self._b_sensor = np.zeros((self._num_envs, mj_model.nsensordata))
        self._b_afrc = np.zeros((self._num_envs, nu))

        if num_threads is None:
            num_threads = min(self._num_envs, os.cpu_count() or 1)
        self._num_threads = max(1, int(num_threads))
        # More chunks than threads so `executor.map` hands them out as workers
        # free up. Per-env cost is uneven (contact count varies with pose, and a
        # resetting env pays an unwarmstarted forward solve), so equal-sized
        # static chunks finish at very different times and every step waits on
        # the slowest one. Oversubscribing the queue costs one dispatch each and
        # buys dynamic load balancing.
        n_chunks = min(self._num_envs, self._num_threads * _CHUNKS_PER_THREAD)
        self._env_chunks = [
            chunk for chunk in
            np.array_split(np.arange(self._num_envs), max(n_chunks, 1))
            if len(chunk)
        ]
        self._executor = (
            ThreadPoolExecutor(max_workers=self._num_threads)
            if self._num_threads > 1 else None
        )

        # Env ranges the observation assembly is split over (see _gather_obs).
        # Far fewer, far larger slices than the physics chunks: this is
        # memory-bandwidth-bound numpy, so it stops scaling once the workers
        # saturate DRAM — measured flat from 6 workers up at 5000 envs, and
        # more slices only add per-slice numpy overhead. Disabled for small
        # pools, where a single pass is already cheaper than the dispatch.
        if self._executor is not None and self._num_envs >= _OBS_THREAD_MIN_ENVS:
            n_sl = min(_OBS_WORKERS, self._num_threads)
            edges = np.linspace(0, self._num_envs, n_sl + 1).astype(int)
            self._obs_slices = [
                (int(edges[i]), int(edges[i + 1])) for i in range(n_sl)
                if edges[i + 1] > edges[i]
            ]
            self._obs_out = np.zeros(
                (self._num_envs, self._obs_size), dtype=np.float32
            )
        else:
            self._obs_slices = None
            self._obs_out = None

        self._rollout = None
        if self._use_rollout:
            self._init_rollout(rollout_addrs)

    # -- rollout stepper -----------------------------------------------------

    def _init_rollout(self, rollout_addrs) -> None:
        """Allocate the buffers ``mujoco.rollout`` reads and writes.

        Why this is faster than the thread pool it replaces, measured on the
        12-core 7900X at 1000 envs stepping the same trajectory:

          - the physics itself: 32.4 ms of bare `ThreadPoolExecutor` mj_step ->
            19.6 ms in rollout's C++ pool. Same work; the executor loses ~40% to
            per-chunk dispatch and to workers queueing for the GIL between
            mj_step calls, which a native pool never touches.
          - the ~16 ms of GIL-held Python glue that used to run per env inside
            the stepping worker (clip, filter EMA, `d.ctrl[:] =`, phase advance)
            becomes a handful of batched numpy ops on (num_envs, nu) arrays.
          - `_capture` disappears: the state and sensordata come back already
            batched, so the ~5 ms of per-env MjData reads are simply not paid.

        End to end that is 14.2k -> 34.3k sps at 1000 envs, 13.7k -> 38.5k at
        4000, with obs/reward/metrics bit-identical to the threaded stepper (see
        the stepper-parity check in check_envpool_parity.py).
        """
        m = self._sim_model
        self._blk_adr, self._frc_adr = rollout_addrs
        self._nstate = mujoco.mj_stateSize(m, _FULL_PHYSICS)
        n, ns, nsub = self._num_envs, self._nstate, self._n_substeps

        self._rollout = mj_rollout.Rollout(nthread=self._num_threads)
        # rollout wants one scratch MjData per thread, not per env. A second set
        # is kept for sampling resets (`_reset_rollout`), so that path never
        # allocates an MjData in the hot loop nor disturbs the stepper's own.
        self._rl_datas = [mujoco.MjData(m) for _ in range(self._num_threads)]
        self._rl_scratch = [mujoco.MjData(m) for _ in range(self._num_threads)]
        # With skip_checks the C++ side does no singleton tiling, so the model
        # list has to be nbatch long. Same object repeated: it is a list of
        # references, built once.
        self._rl_models = [m] * n

        self._state = np.zeros((n, ns))
        self._st_out = np.zeros((n, nsub, ns))
        self._sd_out = np.zeros((n, nsub, m.nsensordata))
        self._control = np.zeros((n, nsub, m.nu))
        # qacc_warmstart is an INPUT to rollout with no matching output, so it
        # cannot be carried across control steps the way the threaded stepper's
        # MjData carries it. Feeding a fixed zero array keeps the stepper
        # deterministic regardless of how the thread pool happens to schedule
        # envs onto scratch data; the alternative (passing None, i.e. inheriting
        # whatever env last used that scratch) would not be. The cost is that
        # the Newton solver re-converges from cold each control step, which
        # moves results by ~1e-9 in the float32 obs — the tolerance the parity
        # check allows for, and the reason it is a tolerance and not equality.
        self._warmstart = np.zeros((n, m.nv))

        # Cached fancy-index helpers for scattering a reset's rows (see
        # _apply_pooled_reset); built once because np.ix_ on every reset showed
        # up in the profile.
        nb = self._model.nbody
        self._body_rows = np.arange(1, nb)
        self._xmat_c0 = np.arange(0, 7, 3)   # xmat is row-major: col 0 = 0,3,6
        self._xmat_c1 = np.arange(1, 8, 3)   # col 1 = 1,4,7

    def _unpack_rollout(self, st, sd, idxs=None) -> None:
        """Scatter rollout's (state, sensordata) rows into the batched buffers.

        This is what `_capture` did, minus the per-env MjData reads: the same
        six arrays, filled from two contiguous blocks. `idxs=None` writes every
        env (the hot path); an index array writes just those rows (auto-reset).
        """
        nb = self._model.nbody
        nq, nv = self._model.nq, self._model.nv
        rows = slice(None) if idxs is None else idxs
        n = self._num_envs if idxs is None else len(idxs)

        self._b_qpos[rows] = st[:, 1:1 + nq]
        self._b_qvel[rows] = st[:, 1 + nq:1 + nq + nv]
        # [pos(3), xaxis(3), yaxis(3)] per body, bodies 1..nbody-1, contiguous.
        blk = sd[:, self._blk_adr:self._blk_adr + 9 * (nb - 1)].reshape(n, nb - 1, 9)
        if idxs is None:
            self._b_xpos[:, 1:] = blk[:, :, 0:3]
            self._b_xmat[:, 1:, self._xmat_c0] = blk[:, :, 3:6]
            self._b_xmat[:, 1:, self._xmat_c1] = blk[:, :, 6:9]
        else:
            self._b_xpos[np.ix_(idxs, self._body_rows)] = blk[:, :, 0:3]
            self._b_xmat[np.ix_(idxs, self._body_rows, self._xmat_c0)] = blk[:, :, 3:6]
            self._b_xmat[np.ix_(idxs, self._body_rows, self._xmat_c1)] = blk[:, :, 6:9]
        self._b_afrc[rows] = sd[:, self._frc_adr:self._frc_adr + self._model.nu]
        # The task's own sensors keep their addresses: the rollout sensors are
        # APPENDED, so the base model's block is still sensordata[:nsensordata]
        # (asserted in add_rollout_sensors).
        self._b_sensor[rows] = sd[:, :self._model.nsensordata]

    def _advance_rollout(self, actions: np.ndarray) -> None:
        """Physics + capture for every env: one native call."""
        cfg = self._config
        # The per-env control work of `_advance_env`, batched. Held the GIL once
        # per env before; now three numpy ops for the whole pool.
        ctrl = np.clip(actions * cfg.action_scale, self._lowers, self._uppers)
        if self._filter_alpha > 0.0:
            ctrl = (
                self._filter_alpha * self._filtered_ctrl
                + (1.0 - self._filter_alpha) * ctrl
            )
        self._filtered_ctrl = ctrl
        # One command held across all substeps == mj_step(nstep=n_substeps).
        self._control[...] = ctrl[:, None, :]

        # skip_checks bypasses the wrapper's per-call shape validation and
        # ascontiguousarray pass; every array here is preallocated, contiguous
        # and float64, which is exactly the contract that check enforces.
        self._rollout.rollout(
            self._rl_models, self._rl_datas, self._state, self._control,
            skip_checks=True, nstep=self._n_substeps,
            initial_warmstart=self._warmstart,
            state=self._st_out, sensordata=self._sd_out,
        )
        self._state[...] = self._st_out[:, -1]
        self._unpack_rollout(self._state, self._sd_out[:, -1])
        self._phase_idx = (self._phase_idx + 1) % self._clip_len

    # -- per-env logic (mirrors MocapTrackingEnv) ----------------------------

    def _capture(self, idxs) -> None:
        """Copy the given envs' MjData fields into the batched buffers.

        Deliberately runs on the CALLING (main) thread, not inside the stepping
        workers. These are ~6 small numpy assignments per env and every one of
        them holds the GIL, so spreading them across the pool only makes 24
        threads queue for the same lock: measured at 1000 envs the threaded
        variant costs 3.6 ms against 3.1 ms serial, and it drags the physics
        down with it by stealing the GIL from the workers between mj_step calls.
        Iterating env-major (all six fields per env) rather than field-major
        keeps each MjData's memory hot.
        """
        q, v = self._b_qpos, self._b_qvel
        xp, xm = self._b_xpos, self._b_xmat
        se, af = self._b_sensor, self._b_afrc
        datas = self._datas
        for i in idxs:
            d = datas[i]
            q[i] = d.qpos
            v[i] = d.qvel
            xp[i] = d.xpos
            xm[i] = d.xmat
            se[i] = d.sensordata
            af[i] = d.actuator_force

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
        if not cfg.random_start:
            start_idx = 0
        elif self._mining_weights is not None:
            start_idx = self._mined_start(rng, start_high)
        else:
            start_idx = int(rng.integers(start_high))

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
        # Filter state starts at the command that HOLDS the reset pose (position
        # mode), so the first filtered steps don't yank the character toward
        # mid-range targets; torque mode starts at zero force. Mirrors the
        # `hold_ctrl` branch in MocapTrackingEnv.reset. Read from the SAMPLED
        # qpos (pre-normalization), which is what the GPU side inverts too.
        if self._actuation == "position":
            self._filtered_ctrl[i] = np.clip(
                (qpos[self._act_qadr] - self._act_q_lo) / self._act_slope - 1.0,
                -1.0, 1.0,
            )
        else:
            self._filtered_ctrl[i] = 0.0
        self._step_count[i] = 0

    # -- reset pool ----------------------------------------------------------

    def refresh_reset_pool(self) -> None:
        """(Re)build the pool of reset states auto-reset gathers from.

        Called once per epoch by the trainer, matching the JAX loop's per-epoch
        `reset_pool = jit_v_reset(...)`, and after `mining_refresh` so the new
        pool already reflects the updated start distribution. Each entry is a
        fully forwarded reset state: the raw sampled qpos/qvel plus the derived
        kinematics and sensor readings the observation needs, so applying one
        later costs no solve.
        """
        if self._reset_pool_size <= 0:
            return
        n = self._reset_pool_size
        m = self._model
        pool = {
            "qpos": np.zeros((n, m.nq)),
            "qvel": np.zeros((n, m.nv)),
            "xpos": np.zeros((n, m.nbody, 3)),
            "xmat": np.zeros((n, m.nbody, 9)),
            "sensor": np.zeros((n, m.nsensordata)),
            "afrc": np.zeros((n, m.nu)),
            "filtered_ctrl": np.zeros((n, m.nu)),
            "phase_idx": np.zeros(n, dtype=np.int64),
            "clip_start": np.zeros(n, dtype=np.int64),
            "clip_len": np.ones(n, dtype=np.int64),
        }
        if self._use_rollout:
            pool["state"] = np.zeros((n, self._nstate))
        # One scratch MjData per worker thread, not per entry: building the pool
        # is O(pool_size) forward solves and allocating that many MjData would
        # dwarf the work itself.
        n_workers = self._num_threads
        scratch = [mujoco.MjData(self._sim_model) for _ in range(n_workers)]
        rngs = [
            np.random.default_rng(s)
            for s in np.random.SeedSequence(
                int(self._pool_rng.integers(2 ** 31))
            ).spawn(n_workers)
        ]

        def worker(w):
            d, rng = scratch[w], rngs[w]
            for k in range(w, n, n_workers):
                self._sample_reset_into(d, rng, pool, k)

        if self._executor is None:
            for w in range(n_workers):
                worker(w)
        else:
            list(self._executor.map(worker, range(n_workers)))
        self._reset_pool = pool

    def _sample_reset_into(self, d, rng, out, k: int) -> None:
        """Draw one reset state into row ``k`` of ``out`` using scratch data ``d``.

        The sampling is `_reset_env`'s, factored out so the on-demand and pooled
        paths cannot drift apart. ``out`` is any dict of (n, ...) arrays keyed as
        below — the reset pool, or the live buffers themselves with ``k`` an env
        index (see `_reset_all_rollout`). An optional ``"state"`` key receives
        the packed rollout state; ``d`` must then be a `self._sim_model` MjData.
        """
        cfg = self._config
        clip_idx = int(rng.integers(self._num_clips))
        clip_start = int(self._clip_starts[clip_idx])
        clip_len = int(self._clip_lengths[clip_idx])
        start_high = (
            clip_len if cfg.cyclic else max(clip_len - int(cfg.look_ahead), 1)
        )
        if not cfg.random_start:
            start_idx = 0
        elif self._mining_weights is not None:
            start_idx = self._mined_start(rng, start_high)
        else:
            start_idx = int(rng.integers(start_high))

        abs_idx = clip_start + start_idx
        noise = cfg.reset_noise_scale
        qpos = self._ref_qpos[abs_idx] + noise * rng.standard_normal(self._model.nq)
        qvel = self._ref_qvel[abs_idx] + noise * rng.standard_normal(self._model.nv)

        m = self._sim_model
        mujoco.mj_resetData(m, d)
        d.qpos[:] = qpos
        d.qvel[:] = qvel
        # See _reset_env: native mj_forward normalizes the noisy root quat in
        # qpos IN PLACE while MJX leaves it as sampled, so the raw quat is
        # restored afterwards — the derived xpos/xmat come from the same
        # normalized xquat both backends' kinematics use.
        q_root = d.qpos[3:7].copy()
        mujoco.mj_forward(m, d)
        d.qpos[3:7] = q_root

        out["qpos"][k] = d.qpos
        out["qvel"][k] = d.qvel
        out["xpos"][k] = d.xpos
        out["xmat"][k] = d.xmat
        # Truncate rather than assign whole: under the rollout stepper `d` is an
        # augmented-model MjData whose sensordata carries the extra rollout
        # sensors after the task's own block.
        out["sensor"][k] = d.sensordata[:self._model.nsensordata]
        out["afrc"][k] = d.actuator_force
        out["phase_idx"][k] = start_idx
        out["clip_start"][k] = clip_start
        out["clip_len"][k] = clip_len
        if "state" in out:
            # The rollout stepper's physics state is this packed row, not the
            # MjData — which is why an auto-reset under it is a pure array copy
            # with no deferred `mj_resetData` to run on the next step.
            mujoco.mj_getState(m, d, out["state"][k], _FULL_PHYSICS)
        if self._actuation == "position":
            out["filtered_ctrl"][k] = np.clip(
                (qpos[self._act_qadr] - self._act_q_lo) / self._act_slope - 1.0,
                -1.0, 1.0,
            )

    def _apply_pooled_reset(self, idxs: np.ndarray) -> None:
        """Auto-reset ``idxs`` by gathering from the reset pool.

        Mirrors `_step_and_autoreset`'s `pool_leaf[idx]` gather. Pure array
        copies: both the MjData that physics will step from next and the batched
        buffers the observation is read out of are written directly, so no
        forward solve and no `_capture` is needed for these envs.
        """
        p = self._reset_pool
        sel = self._pool_rng.integers(0, self._reset_pool_size, len(idxs))

        self._b_qpos[idxs] = p["qpos"][sel]
        self._b_qvel[idxs] = p["qvel"][sel]
        self._b_xpos[idxs] = p["xpos"][sel]
        self._b_xmat[idxs] = p["xmat"][sel]
        self._b_sensor[idxs] = p["sensor"][sel]
        self._b_afrc[idxs] = p["afrc"][sel]

        self._phase_idx[idxs] = p["phase_idx"][sel]
        self._clip_start[idxs] = p["clip_start"][sel]
        self._clip_len[idxs] = p["clip_len"][sel]
        self._filtered_ctrl[idxs] = p["filtered_ctrl"][sel]
        self._last_act[idxs] = 0.0
        self._step_count[idxs] = 0

        if self._use_rollout:
            # Under rollout the physics state IS an array row, so restoring it is
            # the same kind of copy as the buffers above and there is nothing to
            # defer: no per-env `mj_resetData`, no `_needs_reset` bookkeeping.
            self._state[idxs] = p["state"][sel]
            return

        # The MjData itself is NOT touched here — only flagged. Nothing reads it
        # between now and the next `mj_step` (the observation and reward come
        # from the buffers above), and mj_step runs its own forward pass from
        # qpos/qvel, so the restore can be deferred into `_advance_env` where it
        # happens on a worker thread instead of this one. That matters: the
        # gathers above cost 0.84 ms for ~1700 envs while this loop cost 22 ms,
        # because `mj_resetData` plus two slice assignments is ~13 us of
        # GIL-held Python per env and it was the largest serial stage in the
        # step. `_b_qpos`/`_b_qvel` already hold the pooled state to restore.
        self._needs_reset[idxs] = True

    # -- negative mining over start phases -----------------------------------
    #
    # Same three-stage scheme as MocapTrackingEnv (mining_init / mining_observe
    # / mining_refresh): accumulate where episodes DIE per clip-relative phase
    # bin, and once per epoch fold those failure RATES into the start
    # distribution. The split of work differs only in where the state lives —
    # on the GPU path the weights are a traced reset argument threaded by the
    # trainer (so refreshing them cannot retrigger a recompile), whereas the CPU
    # reset is plain Python and the pool can just own them. The sampled
    # distribution is identical; the RNG streams are not (numpy Generator vs
    # jax.random), so parity here is distributional, not bit-for-bit.

    @property
    def mining_bins(self) -> int:
        return self._mining_bins

    def _mined_start(self, rng, start_high: int) -> int:
        """Draw a bin from the difficulty weights, then a frame within it."""
        w = self._mining_weights
        b = int(rng.choice(self._mining_bins, p=w / w.sum()))
        # Bins span the clip, so convert to a frame range and pick uniformly
        # inside the bin — the bin is the unit of *estimation*, not of start
        # granularity, so starts stay spread over every frame.
        lo = (b * start_high) // self._mining_bins
        hi = max(((b + 1) * start_high) // self._mining_bins, lo + 1)
        idx = int(rng.integers(lo, hi))
        # Back up so the policy runs INTO the hard region with context rather
        # than being dropped at the failure point cold.
        return int(np.clip(idx - self._mining_lead_in, 0, start_high - 1))

    def mining_observe(self, phase_idx, clip_len, terminated) -> None:
        """Accumulate this step's visits and genuine failures per phase bin.

        `terminated` must be GENUINE failure, not `done` — see
        MocapTrackingEnv.mining_observe for why a clip that merely ran out must
        not be mined for. Called from `step` with the post-step phase, which is
        what the GPU trainer feeds `mining_observe` as well.
        """
        b = np.clip(
            (phase_idx * self._mining_bins) // np.maximum(clip_len, 1),
            0, self._mining_bins - 1,
        )
        np.add.at(self._mining_visit, b, 1.0)
        np.add.at(self._mining_fail, b, terminated.astype(np.float64))

    def mining_refresh(self) -> None:
        """Fold this epoch's failure rates into the start distribution."""
        # Rate, not count: a bin reached rarely (because we die before it) would
        # otherwise look easy purely for lack of visits.
        rate = self._mining_fail / np.maximum(self._mining_visit, 1.0)
        total = rate.sum()
        uniform = 1.0 / self._mining_bins
        # All-zero rate (nothing failed anywhere) => fall back to uniform rather
        # than dividing by zero and mining noise.
        hard = rate / max(total, 1e-12) if total > 0 else np.full_like(rate, uniform)
        target = (1.0 - self._mining_alpha) * uniform + self._mining_alpha * hard
        w = self._mining_ema * self._mining_weights + (1.0 - self._mining_ema) * target
        self._mining_weights = w / w.sum()
        self._mining_visit[:] = 0.0
        self._mining_fail[:] = 0.0

    def mining_stats(self) -> dict:
        """Loggable scalars: is mining actually concentrating, and on what."""
        w = self._mining_weights
        u = 1.0 / self._mining_bins
        visits = max(float(self._mining_visit.sum()), 1.0)
        return {
            "mining/max_weight_ratio": float(w.max()) / u,
            "mining/hardest_bin": float(np.argmax(w)),
            "mining/fail_rate": float(self._mining_fail.sum()) / visits,
            "mining/effective_bins": float(
                np.exp(-np.sum(w * np.log(w + 1e-12)))
            ),
        }

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
        d_pos = ref_qpos[:, :, :3] - qpos[:, None, :3]
        # Flatten frame-by-frame: [frame1 block, frame2 block, ...].
        ref_delta = np.concatenate(
            [d_joints, d_jvel, d_rot6d, d_pos], axis=-1
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

        # Field order is load-bearing: it must match MocapTrackingEnv._get_obs
        # element-for-element, since a checkpoint trained on one backend is
        # replayed on the other and the normalizer statistics are per-index.
        return np.concatenate([
            qpos[:, :3],                          # root position (x, y, z)
            _quat_to_rot6d_np(qpos[:, 3:7]),      # root orientation (6D rep)
            qvel[:, :6],                          # root linear + angular vel
            qpos[:, 7:],                          # joint positions
            qvel[:, 6:],                          # joint velocities
            feet,                                 # (left, right) contact flags
            body_pos,                             # root-relative body positions
            body_rot6d,                           # body orientations (6D rep)
            last_act,                             # last raw action (pre-filter)
            ref_delta,                            # look-ahead reference deltas
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

        # Root position and orientation get SEPARATE kernels (see the GPU env
        # for why: the shared term was 82-90% orientation while root_termination
        # fires on position alone). `r_root` is retained purely so the logged
        # `reward/root` stays comparable with pre-split runs — it feeds neither
        # the total nor `tracking`.
        root_pos_err = np.sum(np.square(qpos[:, :3] - ref_qpos[:, :3]), axis=1)
        root_quat_err = _quaternion_distance_np(qpos[:, 3:7], ref_qpos[:, 3:7])
        r_root_pos = np.exp(-root_pos_err / cfg.sigma_root_pos)
        r_root_quat = np.exp(-root_quat_err / cfg.sigma_root_quat)
        r_root = np.exp(-(root_pos_err + root_quat_err) / cfg.sigma_root)
        root_dist = np.sqrt(root_pos_err)

        # Root VELOCITY tracking (qvel[:6]). `r_vel` starts at qvel[6:], so
        # without this the root's own velocity appears in no reward term. Kept
        # OUT of `tracking` (like the penalties) so it cannot move the
        # tracking-collapse floor.
        root_vel_err = np.sum(np.square(qvel[:, :6] - ref_qvel[:, :6]), axis=1)
        r_root_vel = np.exp(-root_vel_err / cfg.sigma_root_vel)

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
            + cfg.w_root_pos * r_root_pos
            + cfg.w_root_quat * r_root_quat
        )
        components = {
            "reward/pose": r_pose,
            "reward/vel": r_vel,
            "reward/ee": r_ee,
            "reward/root": r_root,
            "reward/root_pos": r_root_pos,
            "reward/root_quat": r_root_quat,
            "reward/root_vel": r_root_vel,
            "reward/torque": r_torque,
            "reward/action_rate": r_action_rate,
            "root_dist": root_dist,
        }
        return (
            tracking
            + cfg.w_alive
            + cfg.w_root_vel * r_root_vel
            + r_torque
            + r_action_rate,
            tracking,
            root_dist,
            components,
        )

    # -- single-env adapters (used only by check_envpool_parity.py) ----------
    #
    # These read ``self._datas[i]`` DIRECTLY, not the batched buffers, so they
    # are only valid when that MjData is the env's current state. Mid-rollout
    # that is no longer guaranteed: an env auto-reset by `_apply_pooled_reset`
    # has its new state in the buffers and its MjData restored lazily on the
    # next `_advance_env` (see `_needs_reset`). The parity checker places state
    # into the MjData itself before calling these, so it is unaffected — but do
    # not reach for them to inspect a live pool.

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
        # Deferred half of a pooled auto-reset (see _apply_pooled_reset): restore
        # this env's MjData from the state already published to the buffers. Done
        # here so the per-env `mj_resetData` runs on the worker thread rather
        # than serially on the caller's.
        if self._needs_reset[i]:
            mujoco.mj_resetData(self._model, d)
            d.qpos[:] = self._b_qpos[i]
            d.qvel[:] = self._b_qvel[i]
            self._needs_reset[i] = False
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
        """Assemble the batched observation from the captured state buffers.

        Split across the thread pool by env range. This threads where `_capture`
        deliberately does not, and the difference is the op size: `_capture` is
        thousands of tiny per-env assignments that each hold the GIL, whereas
        each slice here is a handful of large numpy ufuncs and concatenates,
        which release it. Measured at 5000 envs: 22.7 ms serial -> 10.0 ms on 6
        workers, bit-identical output (the slices are disjoint and each writes
        only its own rows).
        """
        if self._obs_slices is None:
            return self._obs_batch(
                self._b_qpos, self._b_qvel, self._b_xpos, self._b_xmat,
                self._b_sensor, self._phase_idx, self._clip_start,
                self._clip_len, self._last_act,
            )

        out = self._obs_out

        def worker(bounds):
            lo, hi = bounds
            out[lo:hi] = self._obs_batch(
                self._b_qpos[lo:hi], self._b_qvel[lo:hi], self._b_xpos[lo:hi],
                self._b_xmat[lo:hi], self._b_sensor[lo:hi],
                self._phase_idx[lo:hi], self._clip_start[lo:hi],
                self._clip_len[lo:hi], self._last_act[lo:hi],
            )

        list(self._executor.map(worker, self._obs_slices))
        return out

    def _live_view(self) -> dict:
        """The live per-env buffers, keyed the way `_sample_reset_into` writes.

        Lets the rollout stepper reset straight into the arrays the obs is read
        from, with no MjData in between — the same trick `refresh_reset_pool`
        uses, pointed at the pool's own state instead of at a side table.
        """
        return {
            "qpos": self._b_qpos, "qvel": self._b_qvel,
            "xpos": self._b_xpos, "xmat": self._b_xmat,
            "sensor": self._b_sensor, "afrc": self._b_afrc,
            "filtered_ctrl": self._filtered_ctrl,
            "phase_idx": self._phase_idx, "clip_start": self._clip_start,
            "clip_len": self._clip_len, "state": self._state,
        }

    def _reset_rollout(self, idxs) -> None:
        """Freshly sample a reset for ``idxs`` under the rollout stepper.

        Same per-env sampling and the same per-env RNG streams as `_reset_env`
        (so the two steppers reset to identical states from identical seeds),
        landing in the live buffers and the packed state array rather than in a
        per-env MjData. Used for the initial `reset()` and, when no reset pool is
        configured, for auto-reset.
        """
        view = self._live_view()
        chunks = [c for c in np.array_split(
            np.asarray(idxs), min(len(idxs), self._num_threads)) if len(c)]

        def worker(w):
            d = self._rl_scratch[w]
            for i in chunks[w]:
                self._sample_reset_into(d, self._rngs[i], view, int(i))

        if self._executor is None or len(chunks) == 1:
            for w in range(len(chunks)):
                worker(w)
        else:
            list(self._executor.map(worker, range(len(chunks))))
        self._last_act[idxs] = 0.0
        self._step_count[idxs] = 0

    def reset(self) -> tuple[np.ndarray, dict]:
        if self._use_rollout:
            self._reset_rollout(np.arange(self._num_envs))
        else:
            def worker(chunk):
                for i in chunk:
                    self._reset_env(i)

            self._run_chunked(worker)
            # `_reset_env` wrote the MjData directly, so nothing is deferred.
            self._needs_reset[:] = False
            self._capture(range(self._num_envs))
        # Build the auto-reset pool off the same distribution, on first reset.
        # `refresh_reset_pool` rebuilds it every epoch thereafter.
        if self._reset_pool_size > 0 and self._reset_pool is None:
            self.refresh_reset_pool()
        return self._gather_obs(), {}

    def _advance_threads(self, actions: np.ndarray) -> None:
        """Physics + phase advance, per env across the thread pool (mj_step
        releases the GIL, so this is where the cores get used)."""
        def worker(chunk):
            for i in chunk:
                self._advance_env(i, actions[i])

        self._run_chunked(worker)
        self._capture(range(self._num_envs))

    def step(self, actions: Any) -> tuple[np.ndarray, ...]:
        actions = np.asarray(actions, dtype=np.float64)
        cfg = self._config

        # 1. Advance physics and publish the post-step state into the batched
        #    buffers. The two steppers differ ONLY here — everything below reads
        #    the buffers and is shared, which is what keeps them in parity.
        if self._use_rollout:
            self._advance_rollout(actions)
        else:
            self._advance_threads(actions)

        # 2. Reward + termination from the post-step state, batched over envs.
        #    The buffers still hold the post-step (pre-reset) state here — the
        #    auto-reset in step 4 re-captures only the done envs' rows, and
        #    everything below has consumed them into fresh arrays by then. That
        #    is why one full gather suffices per step where the obvious
        #    structure (gather for the reward, gather again for the obs) needs
        #    two: only the handful of envs that actually reset change state
        #    between the two reads.
        qpos, qvel = self._b_qpos, self._b_qvel
        xpos, afrc = self._b_xpos, self._b_afrc
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
        # Which rule fired, gated exactly like the return value: with
        # early_termination off, low_track/root_too_far are computed but never
        # end an episode, so reporting them as causes would mislead. Mirrors
        # MocapTrackingEnv._get_termination's metrics block.
        gate = bool(cfg.early_termination)
        comps["term/nan"] = nan_check.astype(np.float64)
        comps["term/tracking"] = (low_track & gate).astype(np.float64)
        comps["term/root"] = (root_too_far & gate).astype(np.float64)

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

        # 3. Negative mining reads the POST-step phase and the genuine
        #    termination flag — the same two quantities, at the same point in the
        #    step, that the GPU trainer hands MocapTrackingEnv.mining_observe.
        #    Must run BEFORE the auto-reset overwrites phase_idx.
        if self._mining_weights is not None:
            self.mining_observe(
                self._phase_idx, self._clip_len, terminated_final
            )

        # 4. last_act obs field = the current action (mirror step's info update),
        #    then auto-reset done envs (which zeroes their last_act/filter/phase).
        self._last_act = actions.copy()
        done_idx = np.nonzero(done)[0]
        if self._reset_pool is not None:
            self._apply_pooled_reset(done_idx)
        elif len(done_idx):
            if self._use_rollout:
                self._reset_rollout(done_idx)
            else:
                self._reset_many(done_idx)
                self._capture(done_idx)

        # 5. Observation from the post-reset state, batched over envs.
        obs = self._gather_obs()
        metrics = {k: comps[k].astype(np.float32) for k in _METRIC_KEYS}
        return (
            obs, reward.astype(np.float32), terminated_final, truncated_final,
            {"metrics": metrics},
        )


def build_mocap_envpool_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for the CPU EnvPool-style mocap env (see ``env.builder``).

    Reads from cfg_env (beyond the shared config/reward groups):
      parallel_envs (int):   training pool size. Throughput is near-flat in this
                             number (per-env cost, not dispatch, is the wall):
                             measured 34.3k sps at 1000 and 38.5k at 4000.
      num_threads (int?):    stepping threads; null = min(parallel_envs, cores).
      stepper (str):         "rollout" (default, native batched stepping) or
                             "threads" (per-env mj_step over a ThreadPoolExecutor,
                             the reference implementation). See the module
                             docstring.
      test_episodes (int):   eval pool size; must match trainer.test_episodes.
      seed, clip_ids, collisions, actuation, actuation_kp_scale,
      actuation_kv_ratio: as in the GPU builder, and read with the SAME
      defaults — these change what task is being solved, so a silent
      divergence here would make a CPU-vs-GPU comparison meaningless.

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

    from examples.mocap.cmu_mocap_data import (
        add_rollout_sensors, build_cmu_humanoid, load_cmu_clips,
    )
    from examples.mocap.loader import build_env_config

    config = build_env_config(cfg_env)

    stepper = str(cfg_env.get("stepper", "rollout"))
    if stepper not in ("rollout", "threads"):
        raise ValueError(
            f"env.stepper must be 'rollout' or 'threads', got {stepper!r}"
        )

    # Build model -> load clips -> configure, in that order, because that is
    # what load_mocap_env does: clip retargeting reads the model, so doing it
    # on either side of the model rewrites is a difference worth not having.
    mj_model, xml_path = build_cmu_humanoid()
    mj_model.opt.timestep = config.sim_dt

    clip_ids = list(cfg_env.clip_ids) if cfg_env.get("clip_ids") else None
    dataset = load_cmu_clips(mj_model, clip_ids=clip_ids, ctrl_dt=config.ctrl_dt)

    _configure_collisions(mj_model, resolve_collision_mode(cfg_env))
    # "torque" (raw motors) or "position" (PD-target servos, dm_control tuned
    # gains). Rewrites the actuator arrays in place, so it must run before any
    # MjData is built — the same ordering constraint the GPU env has relative to
    # put_model. Defaults track load_mocap_env's exactly (kv_ratio 0.1).
    actuation = str(cfg_env.get("actuation", "torque"))
    _configure_actuation(
        mj_model,
        actuation,
        float(cfg_env.get("actuation_kp_scale", 1.0)),
        float(cfg_env.get("actuation_kv_ratio", 0.1)),
    )

    seed = int(cfg_env.get("seed", 0))
    num_threads = cfg_env.get("num_threads", None)
    test_episodes = int(cfg_env.get("test_episodes", 5))

    # The sensor-augmented twin the rollout stepper steps. Built from the SAME
    # cached XML and then given the same rewrites in the same order, so its
    # dynamics are bit-identical to `mj_model` — only its sensor block is wider.
    rollout_model = None
    rollout_addrs = None
    if stepper == "rollout":
        rollout_model, blk, frc = add_rollout_sensors(xml_path)
        rollout_model.opt.timestep = config.sim_dt
        _configure_collisions(rollout_model, resolve_collision_mode(cfg_env))
        _configure_actuation(
            rollout_model,
            actuation,
            float(cfg_env.get("actuation_kp_scale", 1.0)),
            float(cfg_env.get("actuation_kv_ratio", 0.1)),
        )
        rollout_addrs = (blk, frc)

    train_pool = MocapCpuPool(
        mj_model, dataset, config,
        num_envs=int(cfg_env.parallel_envs),
        seed=seed,
        num_threads=num_threads,
        actuation=actuation,
        # POOL_SIZE == NUM_ENVS, the same sizing _run_jax uses.
        reset_pool_size=int(cfg_env.parallel_envs),
        rollout_model=rollout_model,
        rollout_addrs=rollout_addrs,
    )
    # Canonical eval protocol (mirrors the GPU loader): start at frame 0, no
    # reset noise, run each clip to its end. See the eval-env note in
    # examples/mocap/loader.py for why this is fixed and must not be changed to
    # random-phase / fixed-horizon sampling. The training pool keeps the original.
    import copy

    eval_config = copy.deepcopy(config)
    eval_config.random_start = False
    eval_config.reset_noise_scale = 0.0
    # Run to the clip END, not to `episode_length`: capping eval at 1000 makes
    # "tracked the whole clip" and "hit the cap" indistinguishable. +1 so the
    # final frame stays reachable past the look_ahead cutoff.
    eval_horizon = int(max(dataset["clip_lengths"])) + 1
    eval_config.episode_length = eval_horizon
    # The eval pool stays on the threaded stepper regardless of `stepper`. It is
    # `test_episodes` envs (single digits), where rollout's advantage — amortizing
    # dispatch over a large batch — does not exist, and it keeps the per-env
    # MjData that `_get_obs`/`_get_reward` and the parity checker read. The two
    # steppers' physics is the same model with the same rewrites, so an eval
    # score is not measuring anything different from what training stepped.
    test_pool = MocapCpuPool(
        mj_model, dataset, eval_config,
        num_envs=test_episodes,
        seed=seed + 1,
        num_threads=num_threads,
        actuation=actuation,
    )

    max_steps = int(config.episode_length)
    return EnvBundle(
        env=EnvPoolWrapper(train_pool, max_episode_steps=max_steps),
        test_env=EnvPoolWrapper(test_pool, max_episode_steps=eval_horizon),
        env_cfg=None,
    )
