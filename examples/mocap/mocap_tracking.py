from typing import Any, Optional

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src import reward

from roxie.environment import functional
from roxie.environment.loader import MuJoCoFuncEnv
from roxie.utils.math import (
    batched_quat_diff,
    mat_to_rot6d,
    quat_to_rot6d,
    quaternion_distance,
)

CMU_BODY_NAMES = {
    "torso": "thorax",
    "end_effectors": ["lhand", "rhand", "lfoot", "rfoot"],
}


GROUND_CONTACT_GEOMS = {
    "lfoot", "lfoot_ch", "ltoes0", "ltoes1", "ltoes2",
    "rfoot", "rfoot_ch", "rtoes0", "rtoes1", "rtoes2",
    "lhand", "rhand", "head",
}


# Touch sensors (added in build_cmu_humanoid) whose reading defines the binary
# foot-contact obs, grouped by physical foot. Order is (left, right); the toe
# sensor is grouped with its ankle so a toe- or heel-only contact still reads
# as that foot being planted.
FOOT_TOUCH_SENSORS = (
    ("lfoot_touch", "ltoes_touch"),
    ("rfoot_touch", "rtoes_touch"),
)


COLLISION_MODES = ("full", "ground", "feet")


def _configure_collisions(m: mujoco.MjModel, mode: str = "full") -> None:
    """Set up the collision filter for the CMU humanoid.

    MuJoCo admits a geom pair when ``contype1 & conaffinity2`` or
    ``contype2 & conaffinity1`` is nonzero (on top of its automatic same-body /
    welded-parent-child and ``<contact exclude>`` filtering). The three modes
    set those two fields to pick which pairs survive:

    ``full`` — the native dm_control model: contype=1/conaffinity=1 everywhere,
      so limbs collide with each other *and* with the floor. Cheap at runtime on
      real mocap poses (~15 simultaneous contacts, bounded by the Warp
      ``naconmax``/``njmax`` budgets), but see the caveat below.

    ``ground`` — no self-contact at all, full ground contact: every humanoid
      geom gets contype=0/conaffinity=1 and the floor contype=1/conaffinity=0,
      so humanoid-vs-humanoid never matches (0 & 1 both ways) while every
      humanoid geom still collides with the floor. Use this when the *reference*
      is the thing under test: retargeted mocap routinely interpenetrates its
      own limbs (arm through torso, thigh through thigh), and with ``full`` the
      solver shoves the body out of exactly the pose the tracking reward is
      asking for — an unreachable target that no policy can fix. It also cuts
      MJX's statically-sized contact arrays from all ~980 potential pairs to the
      ~45 geom-vs-floor ones.

    ``feet`` — collisions restricted to feet/hands/head vs the floor
      (``GROUND_CONTACT_GEOMS``). The leanest contact budget; the torso and
      limbs pass through the ground as well as through each other, so use it
      only when contacts must be as cheap as possible.
    """
    if mode not in COLLISION_MODES:
        raise ValueError(f"collisions must be one of {COLLISION_MODES}, got {mode!r}")

    if mode == "full":
        # Native model already has full self-collision + ground; nothing to do.
        return

    floor_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    if mode == "ground":
        # Humanoid geoms are pure "receivers" (contype 0), the floor a pure
        # "emitter" (conaffinity 0): the only bit that can match is the floor's
        # contype against a humanoid geom's conaffinity.
        m.geom_contype[:] = 0
        m.geom_conaffinity[:] = 1
        m.geom_contype[floor_id] = 1
        m.geom_conaffinity[floor_id] = 0
        return

    m.geom_contype[:] = 0
    m.geom_conaffinity[:] = 0

    m.geom_conaffinity[floor_id] = 1

    for name in GROUND_CONTACT_GEOMS:
        gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid >= 0:
            m.geom_contype[gid] = 1


def resolve_collision_mode(cfg_env: Any) -> str:
    """Read the collision mode off a run config (``env.collisions``).

    Shared by every builder (MJX/Warp loader, CPU envpool) so they cannot drift.
    Falls back to the superseded ``env.self_collisions`` bool when a config
    predates the three-way key — that is what checkpoints saved by older runs
    carry, and play.py replays them from their own saved config.
    """
    mode = cfg_env.get("collisions", None)
    if mode is not None:
        return str(mode)
    legacy = cfg_env.get("self_collisions", None)
    if legacy is None:
        return "full"
    return "full" if bool(legacy) else "feet"


def _configure_actuation(
    m: mujoco.MjModel, mode: str, kp_scale: float = 1.0, kv_ratio: float = 0.0
) -> None:
    """Rewrite the model's actuators in place for the requested control mode.

    ``torque`` (the raw dm_control CMU model): leave the plain motors —
    force = ctrl * gear, ctrl in [-1, 1].

    ``position``: convert every motor into MuJoCo's scaled position servo
    (PD-target control), replicating dm_control's CMUHumanoidPositionControlled
    V2020 (scaled_actuators.add_position_actuator + the _POSITION_ACTUATORS_V2020
    per-joint kp/damping/forcerange table), but applied to the compiled MjModel
    so no second XML is needed. ctrl stays in [-1, 1] and maps affinely onto the
    joint's range — so the policy/noise/action pipeline is untouched; actions
    just *mean* "target pose" instead of "torque":
        force = kp*slope*ctrl + kp*(q_lo + slope) - kp*q      (slope = range/2)
    i.e. force = kp * (target(ctrl) - q), with dm_control's forcerange as the
    strength limit (gear folds to 1).

    D-term = JOINT damping, faithful to V2020: dm_control sets
    `associated_joint.damping = params.damping` and builds the position servo
    with biasprm[2] == 0 (see scaled_actuators.add_position_actuator, which
    hardcodes b2 = 0). So the velocity-proportional force lives on the joint
    (`dof_damping`), NOT in the actuator. That matters under the model's Euler
    integrator: `dof_damping` is integrated implicitly (stable at large values)
    and is exempt from the actuator forcerange, whereas an explicit actuator
    `-kv*qvel` bias shares the force limit and resonates — MuJoCo itself advises
    the implicit/implicitfast integrators for actuator kv. We therefore overwrite
    the joint damping and keep the servo a pure P-term.

    `kv_ratio` is an OPTIONAL extra explicit actuator-damping knob (kv =
    kp*kv_ratio in biasprm[2]); 0.0 = pure V2020, its default. Only reach for it
    if you also switch integrators. `kp_scale` remains the gain knob if the
    dm_control gains prove hot/soft.

    Deliberate deviation: V2020 also puts a 30 ms first-order activation filter
    on the targets (dyntype='filter'). Activation states must exist at model
    COMPILE time, so a post-compile conversion cannot add it — the env's
    ctrl-rate EMA (`target_filter_tc`) is the 40 Hz stand-in for it.
    """
    if mode == "torque":
        return
    if mode != "position":
        raise ValueError(f"unknown actuation mode: {mode!r}")

    from dm_control.locomotion.walkers.cmu_humanoid import _POSITION_ACTUATORS_V2020

    params = {p.name: p for p in _POSITION_ACTUATORS_V2020}
    for i in range(m.nu):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        p = params[name]  # KeyError => model/table mismatch: fail loudly
        jid = m.actuator_trnid[i, 0]
        assert m.jnt_limited[jid], f"joint for actuator {name} has no range"
        q_lo, q_hi = m.jnt_range[jid]
        kp = float(p.kp) * kp_scale
        slope = (q_hi - q_lo) / 2.0

        # V2020 damping is a JOINT property, integrated implicitly by the Euler
        # integrator. Overwrite (not add) to match `associated_joint.damping`.
        m.dof_damping[m.jnt_dofadr[jid]] = float(p.damping)

        m.actuator_gaintype[i] = mujoco.mjtGain.mjGAIN_FIXED
        m.actuator_gainprm[i, :] = 0.0
        m.actuator_gainprm[i, 0] = kp * slope
        m.actuator_biastype[i] = mujoco.mjtBias.mjBIAS_AFFINE
        m.actuator_biasprm[i, :] = 0.0
        m.actuator_biasprm[i, 0] = kp * (q_lo + slope)
        m.actuator_biasprm[i, 1] = -kp
        # Pure P-term by default (b2 = 0, like V2020); kv_ratio > 0 opts into an
        # extra explicit actuator D-term (use an implicit integrator if so).
        m.actuator_biasprm[i, 2] = -kp * kv_ratio
        m.actuator_gear[i, :] = 0.0
        m.actuator_gear[i, 0] = 1.0
        m.actuator_forcerange[i, :] = p.forcerange
        m.actuator_forcelimited[i] = 1
        m.actuator_ctrlrange[i, :] = (-1.0, 1.0)
        m.actuator_ctrllimited[i] = 1


class MocapTrackingEnv(mjx_env.MjxEnv, MuJoCoFuncEnv):
    """CMU mocap tracking, as a ``FuncEnv``.

    Two bases, each for one thing. ``mjx_env.MjxEnv`` supplies the playground
    physics conveniences (``dt``/``sim_dt``/``n_substeps``/``observation_size``)
    and is what ``reset``/``step`` below implement; ``MuJoCoFuncEnv`` supplies
    the functional accessors that read observation, reward, termination and
    truncation back off the ``mjx_env.State`` those two produce. ``initial`` and
    ``transition`` are the FuncEnv entry points and delegate straight to them —
    the env is genuinely both, rather than one wrapped in the other.

    ``params`` is this env's negative-mining table (see ``init_params``). It
    must stay TRACED: the table is regenerated every epoch, and re-tracing the
    reset each time would cost more than the mining buys.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        dataset: dict,
        config: config_dict.ConfigDict,
        body_names: Optional[dict] = None,
        gpu_clip_budget: int = 0,
        impl: str = "jax",
        naconmax: Optional[int] = None,
        njmax: Optional[int] = None,
        naccdmax: Optional[int] = None,
        collisions: str = "full",
        graph_mode: Optional[str] = None,
        clip_swap: bool = True,
        clip_seed: int = 0,
        actuation: str = "torque",
        actuation_kp_scale: float = 1.0,
        actuation_kv_ratio: float = 0.0,
    ):
        super().__init__(config)

        self._mj_model = mj_model
        self._mj_model.opt.timestep = self.sim_dt
        self._actuation = actuation

        # "full" (self-collision + ground), "ground" (no self-contact, every
        # geom still hits the floor) or "feet" — see _configure_collisions.
        _configure_collisions(self._mj_model, collisions)
        # "torque" (raw motors) or "position" (PD-target servos, dm_control
        # tuned gains). Must run before put_model — it rewrites actuator arrays.
        _configure_actuation(
            self._mj_model, actuation, actuation_kp_scale, actuation_kv_ratio
        )

        # Physics backend: "jax" is the classic MJX implementation; "warp" is
        # the NVIDIA-Warp backend (mujoco_warp). Both are dispatched through the
        # same mjx API, so only data construction differs (Warp needs explicit
        # contact/constraint budgets).
        self._impl = impl
        self._naconmax = naconmax
        self._njmax = njmax
        self._naccdmax = naccdmax
        # CUDA-graph capture mode for the Warp backend. mjx defaults to
        # GraphMode.WARP, whose capture cache is keyed on per-step input/output buffer
        # addresses; under JAX those change every step, so a new CUDA graph is
        # captured each step and the evicted ones' native host descriptors are never
        # reclaimed — a steady host-RAM leak that eventually OOMs a long run.
        # GraphMode.WARP_STAGED_EX captures the graph ONCE on fixed staging buffers
        # and replays it, adding only a device->staging memcpy per step.
        # GraphMode.JAX/NONE also avoid the leak but launch kernels eagerly, far
        # slower for a step with this many kernels. Ignored by the "jax" backend.
        put_kwargs = {}
        if impl == "warp" and graph_mode is not None:
            from warp._src.jax_experimental.ffi import GraphMode
            put_kwargs["graph_mode"] = getattr(GraphMode, graph_mode.upper())
        self._mjx_model = mjx.put_model(self._mj_model, impl=impl, **put_kwargs)

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
        # When False, the initial `gpu_clip_budget`-sized subset is kept for the
        # whole run (swap_clips() no-ops) — a fixed-subset/curriculum knob rather
        # than a VRAM-rotation one. The initial pick is seeded by `clip_seed` so
        # the same subset is reproducible across runs (e.g. algorithm A/Bs).
        self._clip_swap = bool(clip_swap)

        self._load_gpu_chunk(seed=clip_seed)

        self._xml_path = None
        self._body_names = body_names or CMU_BODY_NAMES
        self._post_init()

    def _post_init(self) -> None:
        bn = self._body_names
        self._torso_body_id = self._mj_model.body(bn["torso"]).id

        self._ee_body_ids = jp.array(
            [self._mj_model.body(name).id for name in bn["end_effectors"]]
        )

        # Proprioception: every non-world body's frame relative to the root.
        # `_root_body_id` is the body carrying the single free joint (the frame
        # `data.qpos[:3]` positions), used to re-express body xpos root-relative.
        # `_proprio_body_ids` is all bodies but the worldbody (id 0) — its frame
        # is the fixed global origin and carries no proprioceptive signal.
        free_jnts = np.nonzero(
            self._mj_model.jnt_type == mujoco.mjtJoint.mjJNT_FREE
        )[0]
        assert len(free_jnts) == 1, "expected exactly one free (root) joint"
        self._root_body_id = int(self._mj_model.jnt_bodyid[free_jnts[0]])
        self._proprio_body_ids = jp.arange(1, self._mj_model.nbody)

        # Foot ground-contact obs: the sensordata addresses of the per-foot
        # touch sensors (ankle + toe), shape (2, 2), and the normal-force
        # threshold that binarizes them. Read through sensordata rather than the
        # contact list because sensordata is a per-world field on every backend,
        # whereas the Warp backend keeps contacts in a non-vmapped global arena a
        # per-world obs cannot index.
        self._foot_sensor_adr = jp.array(
            [[self._mj_model.sensor(s).adr[0] for s in foot]
             for foot in FOOT_TOUCH_SENSORS]
        )
        self._foot_contact_force_thresh = float(
            self._config.get("foot_contact_force_thresh", 1.0)
        )

        self._lowers = self._mj_model.actuator_ctrlrange[:, 0]
        self._uppers = self._mj_model.actuator_ctrlrange[:, 1]

        # Per-actuator effort normalizer for the reward's effort penalty:
        # the force limit in position mode (forcerange), the motor strength
        # (|gear|) in torque mode — so |actuator_force| / limit is ~[0, 1]
        # under either actuation and the penalty keeps one meaning.
        self._force_limit = jp.array(
            np.where(
                self._mj_model.actuator_forcelimited.astype(bool),
                self._mj_model.actuator_forcerange[:, 1],
                np.abs(self._mj_model.actuator_gear[:, 0]),
            ),
            dtype=jp.float32,
        )

        # First-order target filter (see step). dm_control's position-controlled
        # CMU walker runs a 30 ms activation filter on the servos that our
        # post-compile actuator conversion cannot replicate (activation states
        # are sized at compile time); this ctrl-rate EMA is its 40 Hz stand-in.
        # alpha = weight on the previous filtered command; 0 disables.
        tc = float(self._config.get("target_filter_tc", 0.0))
        self._filter_alpha = (
            float(np.exp(-float(self._config.ctrl_dt) / tc)) if tc > 0 else 0.0
        )
        # ctrl -> joint-angle inverse map, for initializing the filter state to
        # a pose-holding command at reset (position mode only): the actuated
        # joint's qpos address, range low and half-range per actuator.
        jid = self._mj_model.actuator_trnid[:, 0]
        self._act_qadr = jp.array(self._mj_model.jnt_qposadr[jid])
        self._act_q_lo = jp.array(self._mj_model.jnt_range[jid, 0], dtype=jp.float32)
        self._act_slope = jp.array(
            (self._mj_model.jnt_range[jid, 1] - self._mj_model.jnt_range[jid, 0]) / 2.0,
            dtype=jp.float32,
        )

        # --- Negative mining over start phases (see cmu.yaml) ---
        mining = self._config.get("negative_mining", None)
        self._mining_enabled = bool(mining is not None and mining.get("enabled", False))
        if self._mining_enabled:
            self._mining_bins = int(mining.get("bins", 64))
            self._mining_alpha = float(mining.get("alpha", 0.5))
            self._mining_ema = float(mining.get("ema", 0.8))
            self._mining_lead_in = int(mining.get("lead_in", 0))
            print(f"Negative mining ON: {self._mining_bins} phase bins",
                  flush=True)
        else:
            self._mining_bins = 0
            self._mining_alpha = 0.0
            self._mining_ema = 0.0
            self._mining_lead_in = 0

    def _load_gpu_chunk(self, clip_indices=None, seed=None):
        if clip_indices is None:
            if self._gpu_clip_budget >= self._total_clips:
                clip_indices = np.arange(self._total_clips)
            else:
                clip_indices = np.random.default_rng(seed).choice(
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

    # ------------------------------------------------------------------
    # Negative mining over start phases
    #
    # The env adapts its OWN start distribution: phases of a clip that episodes
    # keep dying in get sampled more often. It rides entirely on the FuncEnv
    # `params` hooks, so nothing mutable lives on the env and the whole thing
    # stays jit-friendly. `params` is the difficulty table:
    #
    #   init_params()                       -> params  uniform, empty counters
    #   observe_params(params, info, term)  -> params  per step, on device
    #   epoch_refresh(params)               -> params  once per epoch
    #
    # The per-step piece is two scatter-adds on a (bins,) array; the expensive
    # part (normalisation, EMA) happens once per epoch on the epoch boundary.
    # The diagnostics ride out with the ordinary per-step metrics — see
    # `_mining_metrics` — so the driver needs no separate channel for them.
    # ------------------------------------------------------------------

    def init_params(self):
        """Uniform start distribution and empty counters — or None when mining
        is off, which is also what any other env hands the driver."""
        if not self._mining_enabled:
            return None
        n = self._mining_bins
        return {
            "weights": jp.full((n,), 1.0 / n, dtype=jp.float32),
            "fail": jp.zeros((n,), dtype=jp.float32),
            "visit": jp.zeros((n,), dtype=jp.float32),
        }

    def _phase_bin(self, phase_idx, clip_len):
        """Which difficulty bin a clip-relative phase falls in."""
        return jp.clip(
            (phase_idx * self._mining_bins) // jp.maximum(clip_len, 1),
            0, self._mining_bins - 1,
        )

    def observe_params(self, params, info, terminated):
        """Accumulate where episodes DIE, per clip-relative phase bin.

        `terminated` is GENUINE failure, not `done` — that is the driver's
        contract and this is why it matters: a clip that simply ran out (or hit
        the step limit) is not a tracking failure, and counting it would make
        the end of every clip look maximally hard and soak up the whole start
        budget. It is deliberately the same flag the critic treats as terminal —
        mining and bootstrapping must never disagree about what a failure is.
        """
        if params is None:
            return None
        b = self._phase_bin(info["phase_idx"], info["clip_len"])
        return {
            "weights": params["weights"],
            "visit": params["visit"].at[b].add(1.0),
            "fail": params["fail"].at[b].add(
                jp.asarray(terminated, dtype=jp.bool_).astype(jp.float32)
            ),
        }

    def epoch_refresh(self, params):
        """Fold this epoch's failure rates into the start distribution, then
        reshuffle the on-GPU clip subset (which is what can invalidate the live
        envs, so it is what decides the flag)."""
        if params is not None:
            # Rate, not count: a bin reached rarely (because we die before it)
            # would otherwise look easy purely for lack of visits.
            rate = params["fail"] / jp.maximum(params["visit"], 1.0)
            total = jp.sum(rate)
            # All-zero rate (nothing failed anywhere) => fall back to uniform
            # rather than dividing by zero and mining noise.
            hard = jp.where(
                total > 0, rate / jp.maximum(total, 1e-12),
                1.0 / self._mining_bins,
            )
            uniform = 1.0 / self._mining_bins
            target = (1.0 - self._mining_alpha) * uniform + self._mining_alpha * hard
            w = self._mining_ema * params["weights"] + (1.0 - self._mining_ema) * target
            params = {
                "weights": w / jp.sum(w),
                "fail": jp.zeros_like(params["fail"]),
                "visit": jp.zeros_like(params["visit"]),
            }
        return params, self.swap_clips()

    def _mining_metrics(self, params) -> dict:
        """Loggable scalars: is mining actually concentrating, and on what.

        Emitted every step, as ordinary metrics, because the weights are
        CONSTANT within an epoch — the trainer's epoch mean of each is therefore
        exactly its value. (How often episodes actually fail is already reported
        by the `term/*` metrics.)
        """
        u = 1.0 / self._mining_bins
        w = params["weights"]
        return {
            # 1.0 = uniform; higher = more concentrated on hard bins.
            "mining/max_weight_ratio": jp.max(w) / u,
            "mining/hardest_bin": jp.argmax(w).astype(jp.float32),
            # Effective number of bins actually being sampled (exp of entropy);
            # if this collapses toward 1 the start distribution has degenerated.
            "mining/effective_bins": jp.exp(-jp.sum(w * jp.log(w + 1e-12))),
        }

    def swap_clips(self, seed=None):
        """Reshuffle the on-GPU clip subset. Returns True if clips actually
        changed (callers must reset in-progress envs, whose stored clip
        indices reference the previous chunk), False for a no-op swap.
        No-ops when `clip_swap=False`: the initial subset is pinned for the
        whole run (fixed-subset training / coverage diagnostics)."""
        if not self._clip_swap or self._gpu_clip_budget >= self._total_clips:
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

    # ------------------------------------------------------------------
    # FuncEnv entry points. The accessors (observation/reward/terminal/truncal)
    # come from MuJoCoFuncEnv unchanged — sharing them with the playground
    # adapter is what keeps the two backends from drifting on what a
    # termination is.
    # ------------------------------------------------------------------

    @property
    def observation_space(self):
        # Cached: `observation_size` traces a whole reset via `jax.eval_shape`.
        if getattr(self, "_obs_space", None) is None:
            self._obs_space = functional.unbounded_box(self.observation_size)
        return self._obs_space

    @property
    def action_space(self):
        # Written by `_configure_actuation`: (-1, 1) per joint under position
        # control, the raw motor range under torque control.
        ctrl_range = self._mj_model.actuator_ctrlrange
        return functional.box(ctrl_range[:, 0], ctrl_range[:, 1])

    def initial(self, rng: jax.Array, params: jax.Array | None = None):
        return self.reset(rng, params)

    def transition(self, state, action, rng, params=None):
        return self.step(state, action)

    def transition_info(self, state, action, next_state, params=None) -> dict:
        # `metrics` is what the trainer logs as `train/<key>`; the two phase
        # fields are what `observe_params` bins failures by. Nothing else about
        # the env's internal info (rng, filter state, last action) is anyone
        # else's business, so it stays inside the state.
        metrics = next_state.metrics
        if params is not None:
            metrics = {**metrics, **self._mining_metrics(params)}
        return {
            "metrics": metrics,
            "phase_idx": next_state.info["phase_idx"],
            "clip_len": next_state.info["clip_len"],
        }

    def _abs_idx(self, info: dict[str, Any]) -> jax.Array:
        return info["clip_start"] + info["phase_idx"]

    def reset(self, rng: jax.Array, params: dict | None = None) -> mjx_env.State:
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

        def _uniform_start(r):
            return jax.random.randint(r, (), 0, start_high)

        def _mined_start(r):
            """Draw a bin from the difficulty weights, then a frame within it.

            `params` is a TRACED argument, not a closed-over constant, so the
            driver can refresh the table every epoch without retriggering a
            recompile of the reset (which would cost more than the mining buys).
            """
            bin_rng, frac_rng = jax.random.split(r)
            b = jax.random.categorical(bin_rng, jp.log(params["weights"] + 1e-12))
            # Bins span the clip, so convert to a frame range and pick uniformly
            # inside the bin — the bin is the unit of *estimation*, not of start
            # granularity, so starts stay spread over every frame.
            lo = (b * start_high) // self._mining_bins
            hi = jp.maximum(((b + 1) * start_high) // self._mining_bins, lo + 1)
            idx = jax.random.randint(frac_rng, (), lo, hi)
            # Back up so the policy runs INTO the hard region with context rather
            # than being dropped at the failure point cold.
            return jp.clip(idx - self._mining_lead_in, 0, start_high - 1)

        if params is not None and self._mining_enabled:
            start_sampler = _mined_start
        else:
            start_sampler = _uniform_start

        start_idx = jax.lax.cond(
            self._config.random_start,
            start_sampler,
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

        # Filter state starts at the command that HOLDS the reset pose (position
        # mode), so the first filtered steps don't yank the character toward
        # mid-range targets; torque mode starts at zero force. Carried in info
        # so reset/step pytrees stay structurally identical.
        if self._actuation == "position":
            hold_ctrl = jp.clip(
                (qpos[self._act_qadr] - self._act_q_lo) / self._act_slope - 1.0,
                -1.0,
                1.0,
            )
        else:
            hold_ctrl = jp.zeros(self.mjx_model.nu)

        info = {
            "rng": rng,
            "phase_idx": start_idx,
            "clip_start": clip_start,
            "clip_len": clip_len,
            "last_act": jp.zeros(self.mjx_model.nu),
            "filtered_ctrl": hold_ctrl,
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
            "reward/root_pos": jp.zeros(()),
            "reward/root_quat": jp.zeros(()),
            "reward/root_vel": jp.zeros(()),
            "reward/torque": jp.zeros(()),
            "reward/action_rate": jp.zeros(()),
            "root_dist": jp.zeros(()),
            # Must be present here too: `State.metrics` is part of the pytree, so
            # reset and step have to agree on its keys or the structures mismatch
            # under jit/scan. See _get_termination for what these mean.
            "term/nan": jp.zeros(()),
            "term/tracking": jp.zeros(()),
            "term/root": jp.zeros(()),
        }

        reward_val, done = jp.zeros(2)
        obs = self._get_obs(data, info)
        return mjx_env.State(data, obs, reward_val, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        rng, _ = jax.random.split(state.info["rng"])

        ctrl = jp.clip(
            action * self._config.action_scale, self._lowers, self._uppers
        )
        # First-order smoothing of the applied command (target_filter_tc > 0):
        # high-frequency target chatter becomes physically inert, so the policy
        # gains nothing from it. The filter state rides in info; alpha is a
        # trace-time constant (0 = pass-through).
        if self._filter_alpha > 0.0:
            ctrl = (
                self._filter_alpha * state.info["filtered_ctrl"]
                + (1.0 - self._filter_alpha) * ctrl
            )
        data = mjx_env.step(self.mjx_model, state.data, ctrl, self.n_substeps)

        clip_len = state.info["clip_len"]
        phase_idx = (state.info["phase_idx"] + 1) % clip_len

        abs_idx = state.info["clip_start"] + phase_idx
        reward_val, tracking, root_dist = self._get_reward(
            data, abs_idx, ctrl, action, state.info["last_act"], state.metrics
        )

        # Genuine termination: fall / NaN / tracking collapse / root drift.
        # `state.metrics` is mutated in place (same as _get_reward above) to
        # record which rule fired.
        terminated = self._get_termination(data, tracking, root_dist, state.metrics)

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
            "filtered_ctrl": ctrl,
            "truncation": clip_truncated,
        }

        obs = self._get_obs(data, info)
        done = done.astype(jp.float32)
        return mjx_env.State(data, obs, reward_val, done, state.metrics, info)

    def _feet_contacts(self, data: mjx.Data) -> jax.Array:
        """Binary ground-contact flag per foot, ordered (left, right).

        Reads the per-foot ``touch`` sensors (each sums the normal force of
        contacts under its foot/toe zone; ~0 when airborne): a foot reads as 1.0
        when the summed force over its ankle and toe sensors exceeds
        ``foot_contact_force_thresh`` newtons. Uses ``sensordata`` — a per-world
        field on every backend (jax / warp / native MjData) — rather than the
        raw contact list, which the Warp backend stores in a non-vmapped global
        arena a per-world obs cannot demux. Returns float32 (2,).
        """
        touch = data.sensordata[self._foot_sensor_adr]  # (2, 2), >= 0
        planted = jp.sum(touch, axis=-1) > self._foot_contact_force_thresh
        return planted.astype(jp.float32)

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

        # Orientation delta as the 6D continuous rotation rep of the relative
        # rotation (ref^-1 * current), not the raw quaternion diff: continuous
        # network input, no double-cover discontinuity. (look_ahead, 6)
        d_rot6d = quat_to_rot6d(batched_quat_diff(ref_qpos[:, 3:7], data.qpos[3:7]))
        d_pos = ref_qpos[:, :3] - data.qpos[:3]

        # Flatten frame-by-frame: [frame1 block, frame2 block, ...].
        ref_delta = jp.concatenate(
            [d_joints, d_jvel, d_rot6d, d_pos], axis=-1
        ).reshape(-1)

        # Proprioception: each body's global frame (xpos/xmat), re-expressed
        # relative to the root. Positions are root-relative (root xpos subtracted,
        # so the whole body drops out of the absolute world position the policy
        # can't affect); orientations are the global xmat as a 6D continuous
        # rotation rep. This hands the policy the end-effector / limb geometry
        # directly instead of forcing it to reconstruct forward kinematics from
        # qpos. (nbody-1, 3) and (nbody-1, 6) flattened.
        root_pos = data.xpos[self._root_body_id]
        body_pos = (data.xpos[self._proprio_body_ids] - root_pos).reshape(-1)
        body_rot6d = mat_to_rot6d(data.xmat[self._proprio_body_ids]).reshape(-1)

        obs = jp.concatenate([
            data.qpos[:3], # root position (x, y, z)
            quat_to_rot6d(data.qpos[3:7]), # root orientation (6D continuous rotation rep)
            data.qvel[:6], # root linear and angular velocity (vx, vy, vz, wx, wy, wz)
            data.qpos[7:], # joint positions (excluding root)
            data.qvel[6:], # joint velocities (excluding root)
            # Binary (left, right) foot ground-contact flags — an explicit
            # gait-phase cue the policy would otherwise have to infer from the
            # full contact-rich dynamics.
            self._feet_contacts(data),
            body_pos, # root-relative body positions (flattened)
            body_rot6d, # root-relative body orientations (flattened)
            info["last_act"], # last raw action (pre-filter) for action-rate penalty
            ref_delta, # look-ahead reference trajectory deltas (flattened)
        ])
        return obs

    def _get_reward(
        self,
        data: mjx.Data,
        abs_idx: jax.Array,
        ctrl: jax.Array,
        action: jax.Array,
        last_action: jax.Array,
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

        # Root position and orientation get SEPARATE kernels. A shared exponential
        # is dominated by the orientation error, while `root_termination` fires on
        # `root_dist = sqrt(root_pos_err)` — position alone — so a single `w_root`
        # puts most of its pressure on a quantity that never terminates anything.
        # Split, `w_root_pos` weights exactly what the termination measures.
        root_pos_err = jp.sum(jp.square(data.qpos[:3] - ref_qpos[:3]))
        root_quat_err = quaternion_distance(data.qpos[3:7], ref_qpos[3:7])

        r_root_pos = jp.exp(-root_pos_err / cfg.sigma_root_pos)
        r_root_quat = jp.exp(-root_quat_err / cfg.sigma_root_quat)
        # The combined kernel, logged as `reward/root` alongside the split terms.
        r_root = jp.exp(-(root_pos_err + root_quat_err) / cfg.sigma_root)

        root_dist = jp.sqrt(root_pos_err)

        # Root VELOCITY tracking (qvel[:6] = 3 linear + 3 angular), deliberately its
        # own term: `r_vel` above starts at qvel[6:], so the root's own velocity is
        # otherwise absent from every reward component — and root drift is the
        # integral of exactly this error. Without it, pressure on the root only
        # arrives once drift has already accumulated into position error.
        #
        # Kept OUT of the `tracking` sum below, like the torque/action-rate
        # penalties: `tracking` sets the collapse threshold via
        # `min_tracking_frac * sum(weights)`, so folding this in would move the
        # termination floor and the reward gradient at once. It shapes behaviour only.
        root_vel_err = jp.sum(jp.square(data.qvel[:6] - ref_qvel[:6]))
        r_root_vel = jp.exp(-root_vel_err / cfg.sigma_root_vel)

        # Effort penalty on the ACTUAL actuator force, normalized by each actuator's
        # strength limit so it means "effort" under either actuation mode: in torque
        # mode actuator_force = gear * ctrl, so force/limit reduces to ctrl, while in
        # position mode ctrl is a target pose and penalizing it would be wrong — the
        # servo's realized force is the effort. Negative, and kept OUT of `tracking`
        # so it never feeds the tracking-collapse termination.
        effort = data.actuator_force / self._force_limit
        r_torque = -cfg.w_torque * jp.mean(jp.square(effort))

        # Action-rate penalty on the RAW policy output, pre-filter, so it charges the
        # source of chatter rather than its filtered echo. In normalized [-1, 1]
        # units, so a persistent full-range flip costs w_action_rate * 4 per step. The
        # first step of an episode compares against last_act = 0, a bounded artifact.
        # Kept OUT of `tracking` so it never feeds the tracking-collapse termination.
        r_action_rate = -cfg.w_action_rate * jp.mean(jp.square(action - last_action))

        metrics["reward/pose"] = r_pose
        metrics["reward/vel"] = r_vel
        metrics["reward/ee"] = r_ee
        metrics["reward/root"] = r_root
        metrics["reward/root_pos"] = r_root_pos
        metrics["reward/root_quat"] = r_root_quat
        metrics["reward/root_vel"] = r_root_vel
        metrics["reward/torque"] = r_torque
        metrics["reward/action_rate"] = r_action_rate
        metrics["root_dist"] = root_dist

        # Weighted tracking reward (everything but the constant alive bonus and
        # the torque penalty). Returned alongside the total (with the root
        # drift) so termination can gate on both without recomputing.
        tracking = (
            cfg.w_pose * r_pose
            + cfg.w_vel * r_vel
            + cfg.w_ee * r_ee
            + cfg.w_root_pos * r_root_pos
            + cfg.w_root_quat * r_root_quat
        )
        return (
            tracking
            + cfg.w_alive
            + cfg.w_root_vel * r_root_vel
            + r_torque
            + r_action_rate,
            tracking,
            root_dist,
        )

    def _get_termination(
        self,
        data: mjx.Data,
        tracking: jax.Array,
        root_dist: jax.Array,
        metrics: dict[str, Any] | None = None,
    ) -> jax.Array:
        """Genuine termination: NaN / tracking collapse / root drift.

        When `metrics` is given, records WHICH cause fired as `term/nan`,
        `term/tracking` and `term/root`. Episodes here end almost entirely by
        early termination rather than the 1000-step cap, so knowing which rule
        binds is what tells you whether to retune the reward (tracking floor) or
        the geometry (root drift) -- previously both were invisible.

        These are per-step indicators, and the trainer averages them over the
        epoch, so the logged value is `terminations_of_this_cause / env_steps`.
        Multiply by the epoch's mean episode `length` to read it as a fraction of
        episodes. The causes are NOT mutually exclusive -- both can fire on the
        same step -- so they need not sum to the overall termination rate.
        """
        nan_check = jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

        # Tracking collapse: terminate once the weighted tracking reward drops
        # below a fraction of its max (= sum of component weights). Weights and
        # the fraction are static config, so this floor is a trace-time constant.
        cfg = self._config.reward_config
        rt = self._config.reward_termination
        if rt.enabled:
            max_track = (
                cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root_pos + cfg.w_root_quat
            )
            low_track = tracking < rt.min_tracking_frac * max_track
        else:
            low_track = jp.bool_(False)

        # Root-drift termination: end the episode once the root wanders more
        # than `max_dist` metres from the reference root position (position
        # only; orientation stays the reward's job). Geometric, so unlike the
        # tracking-fraction rule it is decoupled from reward shaping — sigma/
        # weight retunes don't silently move this floor. `root_dist` comes from
        # _get_reward, which already holds the phase-indexed reference frame.
        rrt = self._config.root_termination
        if rrt.enabled:
            root_too_far = root_dist > rrt.max_dist
        else:
            root_too_far = jp.bool_(False)

        if metrics is not None:
            # Gate the per-cause indicators the same way the return value is
            # gated: with early_termination off, low_track/root_too_far are
            # computed but never actually end an episode, so reporting them as
            # causes would be misleading.
            gate = jp.asarray(self._config.early_termination, dtype=bool)
            metrics["term/nan"] = nan_check.astype(jp.float32)
            metrics["term/tracking"] = (low_track & gate).astype(jp.float32)
            metrics["term/root"] = (root_too_far & gate).astype(jp.float32)

        return jp.where(
            self._config.early_termination,
            nan_check | low_track | root_too_far,
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

    # ------------------------------------------------------------------
    # Native-CPU playback hooks (roxie.utils.native_player.NativePlayer)
    #
    # These let the env be stepped on native ``mujoco.mj_step`` for fast,
    # GPU-free playback. They REUSE the MJX ``_get_obs``/``_get_reward``/
    # ``_get_termination`` above (those read only qpos/qvel/xpos/actuator_force,
    # which a native MjData also carries), so observation/reward/termination stay
    # bit-for-bit identical to training — only the integrator differs. The
    # generic player owns the MjData and mj_step; the task-specific reset,
    # ctrl pipeline and phase bookkeeping live here.
    # ------------------------------------------------------------------

    @property
    def native_n_substeps(self) -> int:
        return int(self.n_substeps)

    def native_reset(self, data: mujoco.MjData, key: jax.Array) -> dict[str, Any]:
        # Deterministic playback: frame 0 of a clip (no random start, no reset
        # noise), mirroring the eval env's reproducible reset. The clip is picked
        # from `key` so successive episodes cycle through the pool.
        clip_idx = int(jax.random.randint(key, (), 0, self._num_clips))
        clip_start = int(self._clip_starts[clip_idx])
        clip_len = int(self._clip_lengths[clip_idx])
        abs_idx = clip_start  # phase_idx 0

        mujoco.mj_resetData(self._mj_model, data)
        data.qpos[:] = np.asarray(self._ref_qpos[abs_idx])
        data.qvel[:] = np.asarray(self._ref_qvel[abs_idx])
        mujoco.mj_forward(self._mj_model, data)

        # Filter state holds the reset pose (position) / zero force (torque),
        # mirroring reset() so the first filtered steps don't yank the character.
        if self._actuation == "position":
            filtered = np.clip(
                (data.qpos[np.asarray(self._act_qadr)] - np.asarray(self._act_q_lo))
                / np.asarray(self._act_slope) - 1.0,
                -1.0, 1.0,
            )
        else:
            filtered = np.zeros(self._mj_model.nu)

        return {
            "clip_start": clip_start,
            "phase_idx": 0,
            "clip_len": clip_len,
            "last_act": np.zeros(self._mj_model.nu),
            "filtered_ctrl": filtered,
        }

    def native_obs(self, data: mujoco.MjData, info: dict[str, Any]) -> np.ndarray:
        return np.asarray(self._get_obs(data, info))

    def native_control(
        self, action: Any, info: dict[str, Any]
    ) -> tuple[np.ndarray, dict[str, Any]]:
        action = np.asarray(action, dtype=np.float64)
        ctrl = np.clip(
            action * self._config.action_scale,
            np.asarray(self._lowers), np.asarray(self._uppers),
        )
        if self._filter_alpha > 0.0:
            ctrl = (
                self._filter_alpha * info["filtered_ctrl"]
                + (1.0 - self._filter_alpha) * ctrl
            )
        info["filtered_ctrl"] = ctrl
        return ctrl, info

    def native_post(
        self, data: mujoco.MjData, action: Any, ctrl: np.ndarray, info: dict[str, Any]
    ) -> tuple[float, bool, dict[str, Any], dict[str, Any]]:
        clip_len = info["clip_len"]
        phase = (info["phase_idx"] + 1) % clip_len
        abs_idx = info["clip_start"] + phase

        metrics: dict[str, Any] = {}
        reward, tracking, root_dist = self._get_reward(
            data, abs_idx, ctrl, np.asarray(action), info["last_act"], metrics
        )
        terminated = bool(self._get_termination(data, tracking, root_dist, metrics))
        clip_truncated = (not self._config.cyclic) and (
            phase >= clip_len - self._config.look_ahead
        )
        done = terminated or clip_truncated

        info["phase_idx"] = phase
        info["last_act"] = np.asarray(action)
        return float(reward), done, metrics, info
