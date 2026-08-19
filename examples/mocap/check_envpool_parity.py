"""Parity check: CPU EnvPool mirror vs the MJX mocap-tracking env.

Verifies that the numpy re-implementation in ``mocap_envpool.py`` computes the
same observations and reward components as ``MocapTrackingEnv`` (impl="jax")
when both are placed in the *identical* state, and that the pool's two physics
steppers agree with each other. Five checks:

  1. **Observation parity** — for random (clip, phase, noisy-state) tuples,
     both backends must emit the same obs vector (float32 tolerance).

  2. **Reward parity** — same states: all five reward components and the total
     must agree. The end-effector term goes through each backend's own forward
     kinematics (mjx.forward vs mj_forward), so this also catches model or
     body-indexing mismatches.

  3. **One-control-step divergence** (informational) — both backends step the
     same ctrl from the same state. MJX (float32, its own solver) and native
     MuJoCo (float64) legitimately diverge, so this only reports the gap; it
     is not a pass/fail check.

  4. **Rollout sensor fidelity** — the sensors added by ``add_rollout_sensors``
     to carry MjData out of ``mujoco.rollout`` must read back *exactly* the
     fields they stand in for (xpos, xmat columns 0/1, actuator_force). This is
     the check that catches the ``mjOBJ_BODY`` / ``mjOBJ_XBODY`` trap: BODY is
     the body's inertial frame, and picking it produces observations that look
     entirely reasonable and are wrong.

  5. **Stepper parity** — the ``rollout`` and ``threads`` steppers must agree.
     Split in two, because ``qacc_warmstart`` is an input to ``mujoco.rollout``
     with no corresponding output: the rollout stepper cannot carry the solver's
     warm start across control steps the way a per-env MjData does, so it
     re-converges Newton from cold every step. (a) *Single-step equivalence*,
     the pass/fail half: the rollout pool is re-synced to the threaded pool's
     state — warmstart included — before each step, and the two must then agree
     BIT-FOR-BIT. (b) *Free-running divergence*, informational: released from a
     common state the two separate exponentially (~1e-5 by step 30, ~1e-1 by
     step 200), because a humanoid on a floor is chaotic and the cold warmstart
     plants a ~1e-9 seed. Compare check 3, where MJX and native MuJoCo separate
     by 1e-3 after a single step. A trajectory-matching test would therefore
     pass or fail on nothing but its horizon, which is why (a) is the verdict
     and (b) only reports the curve.

Run from the repo root::

    python examples/mocap/check_envpool_parity.py
    python examples/mocap/check_envpool_parity.py --frames 20 --no-check-physics
    python examples/mocap/check_envpool_parity.py --no-check-stepper
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.4")

import sys

import click
import jax
import jax.numpy as jp
import numpy as np

from roxie.utils import hydra_searchpath

# examples/ is not part of the installed roxie package; put the repo root on the
# path so ``examples.mocap`` is importable when run as a script.
sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))

from mujoco_playground._src import mjx_env  # noqa: E402

from examples.mocap.cmu_mocap_data import (  # noqa: E402
    add_rollout_sensors, build_cmu_humanoid, load_cmu_clips,
)
from examples.mocap.loader import load_default_config  # noqa: E402
from examples.mocap.mocap_envpool import MocapCpuPool  # noqa: E402
from examples.mocap.mocap_tracking import (  # noqa: E402
    MocapTrackingEnv, _configure_actuation, _configure_collisions,
)

_COMPONENT_KEYS = (
    "reward/pose", "reward/vel", "reward/ee", "reward/root",
    "reward/root_pos", "reward/root_quat", "reward/root_vel",
    "reward/torque", "reward/action_rate", "root_dist",
)


def _sample_state(pool, rng, noise):
    """Random (clip, phase) plus a noisy state around the reference."""
    clip_idx = int(rng.integers(pool._num_clips))
    clip_start = int(pool._clip_starts[clip_idx])
    clip_len = int(pool._clip_lengths[clip_idx])
    look_ahead = int(pool._config.look_ahead)
    phase_idx = int(rng.integers(max(clip_len - look_ahead - 1, 1)))
    abs_idx = clip_start + phase_idx

    qpos = pool._ref_qpos[abs_idx].astype(np.float64)
    qvel = pool._ref_qvel[abs_idx].astype(np.float64)
    qpos = qpos + noise * rng.standard_normal(qpos.shape)
    qvel = qvel + noise * rng.standard_normal(qvel.shape)
    # Keep the root quat unit: post-step states always carry unit quats on both
    # backends (both integrators normalize), and native mj_forward normalizes
    # qpos in place while MJX does not — an unnormalized test quat would
    # compare CPU-normalized against MJX-raw, a mismatch that never occurs on
    # the training path.
    qpos[3:7] /= np.linalg.norm(qpos[3:7]) + 1e-12
    return clip_start, clip_len, phase_idx, abs_idx, qpos, qvel


def _place_pool_env(
    pool, i, clip_start, clip_len, phase_idx, qpos, qvel, last_act, ctrl=None
):
    import mujoco

    d = pool._datas[i]
    mujoco.mj_resetData(pool._model, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    # Set the applied ctrl before forwarding so d.actuator_force reflects it —
    # the reward's effort term reads actuator_force, not the raw command.
    if ctrl is not None:
        d.ctrl[:] = ctrl
    mujoco.mj_forward(pool._model, d)
    pool._clip_start[i] = clip_start
    pool._clip_len[i] = clip_len
    pool._phase_idx[i] = phase_idx
    pool._last_act[i] = last_act
    return d


def _build_rollout_twin(xml_path, config, actuation):
    """The sensor-augmented model, with the SAME rewrites MocapTrackingEnv applies.

    Must mirror this checker's ``MocapTrackingEnv(...)`` construction argument for
    argument (its defaults: collisions="full", kp_scale=1.0, kv_ratio=0.0) or the
    two models would differ in dynamics and every downstream check would be
    measuring the wrong thing.
    """
    model, blk, frc = add_rollout_sensors(xml_path)
    model.opt.timestep = config.sim_dt
    _configure_collisions(model, "full")
    _configure_actuation(model, actuation, 1.0, 0.0)
    return model, blk, frc


def _check_rollout_sensors(pool, mj_model, aug_model, blk, frc, rng, frames, noise):
    """Do the rollout sensors read back exactly the MjData fields they mirror?"""
    import mujoco

    nb, nu = mj_model.nbody, mj_model.nu
    d_base = mujoco.MjData(mj_model)
    d_aug = mujoco.MjData(aug_model)
    worst = {"framepos vs xpos": 0.0, "framexaxis vs xmat[:,0]": 0.0,
             "frameyaxis vs xmat[:,1]": 0.0, "actuatorfrc vs actuator_force": 0.0,
             "task sensor block": 0.0}

    for _ in range(frames):
        _, _, _, _, qpos, qvel = _sample_state(pool, rng, noise)
        ctrl = rng.uniform(-1, 1, nu)
        for m, d in ((mj_model, d_base), (aug_model, d_aug)):
            mujoco.mj_resetData(m, d)
            d.qpos[:] = qpos
            d.qvel[:] = qvel
            d.ctrl[:] = ctrl
            mujoco.mj_forward(m, d)

        sd = d_aug.sensordata
        block = sd[blk:blk + 9 * (nb - 1)].reshape(nb - 1, 9)
        xmat = d_base.xmat[1:].reshape(nb - 1, 3, 3)
        for key, got, want in (
            ("framepos vs xpos", block[:, 0:3], d_base.xpos[1:]),
            ("framexaxis vs xmat[:,0]", block[:, 3:6], xmat[:, :, 0]),
            ("frameyaxis vs xmat[:,1]", block[:, 6:9], xmat[:, :, 1]),
            ("actuatorfrc vs actuator_force", sd[frc:frc + nu],
             d_base.actuator_force),
            ("task sensor block", sd[:mj_model.nsensordata], d_base.sensordata),
        ):
            worst[key] = max(worst[key], float(np.abs(got - want).max()))

    print(f"\nRollout sensor fidelity ({frames} states, expect exact):")
    for k, v in worst.items():
        print(f"  {k:<30} max |diff| = {v:.2e}")
    ok = all(v == 0.0 for v in worst.values())
    print(f"  sensor fidelity: {'OK' if ok else 'FAIL'} (tol 0)")
    return ok


def _sync_rollout_from_threads(p_rollout, p_threads):
    """Place the rollout pool in the threaded pool's exact state, warmstart included.

    ``qacc_warmstart`` is the whole reason this is needed: rollout accepts it as
    an input but returns no updated value, so a free-running rollout pool cannot
    carry it and its solver re-converges from cold. Handed the threaded pool's
    warmstart, rollout starts Newton from precisely where ``mj_step`` would — and
    the step becomes bit-for-bit comparable.
    """
    import mujoco

    for i, d in enumerate(p_threads._datas):
        mujoco.mj_getState(
            p_threads._model, d, p_rollout._state[i],
            mujoco.mjtState.mjSTATE_FULLPHYSICS,
        )
        p_rollout._warmstart[i] = d.qacc_warmstart
    for attr in ("_phase_idx", "_clip_start", "_clip_len", "_last_act",
                 "_filtered_ctrl", "_step_count", "_b_qpos", "_b_qvel",
                 "_b_xpos", "_b_xmat", "_b_sensor", "_b_afrc"):
        getattr(p_rollout, attr)[...] = getattr(p_threads, attr)


def _check_stepper_parity(
    mj_model, aug_model, addrs, dataset, config, actuation, seed, steps, envs
):
    """Do the rollout and threads steppers agree?

    Two measurements, because they answer different questions and only one of
    them is a defensible pass/fail.

    **A. Single-step equivalence (pass/fail, exact).** The rollout pool is
    re-synced to the threaded pool's state before every step, so each comparison
    is one step from identical inputs and nothing accumulates. Handed the same
    warmstart, the two steppers must agree BIT-FOR-BIT — there is no tolerance to
    argue about, and any nonzero result is a genuine defect.

    **B. Free-running divergence (informational).** Both pools are then released
    from a common state and stepped independently. They separate, and fast: the
    ~1e-9 seed that the un-carried warmstart plants gets amplified exponentially,
    because a humanoid on a floor is chaotic. Measured here: ~1e-5 by step 30,
    ~1e-1 by step 200. That is physics, not a bug — check 3 shows MJX and native
    MuJoCo separating by 1e-3 after a SINGLE step — but it does mean a
    trajectory-comparison test would pass or fail purely on how long it ran, so
    this half only reports the curve.

    Auto-reset is the one place the two may legitimately differ (each draws from
    its own reset pool), so both halves run under a config where no env can
    reset: no early termination, cyclic clips, no step limit.
    """
    import copy

    cfg = copy.deepcopy(config)
    cfg.early_termination = False
    cfg.root_termination.enabled = False
    cfg.reward_termination.enabled = False
    cfg.cyclic = True
    cfg.episode_length = 10 ** 9

    common = dict(
        num_envs=envs, seed=seed, num_threads=4, actuation=actuation,
        reset_pool_size=0,
    )
    p_threads = MocapCpuPool(mj_model, dataset, cfg, **common)
    p_rollout = MocapCpuPool(
        mj_model, dataset, cfg, rollout_model=aug_model, rollout_addrs=addrs,
        **common,
    )
    o1, _ = p_threads.reset()
    o2, _ = p_rollout.reset()
    # The two reset paths share `_sample_reset_into` and the per-env RNG streams,
    # so identical seeds must place both pools in identical states. If this trips,
    # nothing after it means anything.
    reset_gap = float(np.abs(o1 - o2).max())

    nu = mj_model.nu

    def _one_step(a):
        o1, r1, te1, tr1, i1 = p_threads.step(a)
        o2, r2, te2, tr2, i2 = p_rollout.step(a)
        if te1.any() or tr1.any() or te2.any() or tr2.any():
            raise AssertionError(
                "an env reset during the stepper check; the no-reset config is "
                "wrong and the comparison would be meaningless"
            )
        met = max(float(np.abs(i1["metrics"][k] - i2["metrics"][k]).max())
                  for k in i1["metrics"])
        return (float(np.abs(o1 - o2).max()), float(np.abs(r1 - r2).max()), met,
                np.array_equal(te1, te2) and np.array_equal(tr1, tr2))

    # --- A. re-synced single steps -----------------------------------------
    rng = np.random.default_rng(seed + 101)
    a_obs = a_rew = a_met = 0.0
    a_term = True
    for _ in range(steps):
        _sync_rollout_from_threads(p_rollout, p_threads)
        d_obs, d_rew, d_met, t_ok = _one_step(
            rng.uniform(-0.4, 0.4, size=(envs, nu))
        )
        a_obs, a_rew, a_met = max(a_obs, d_obs), max(a_rew, d_rew), max(a_met, d_met)
        a_term &= t_ok

    print(f"\nStepper parity A: single steps from identical state "
          f"({envs} envs x {steps} steps):")
    print(f"  obs after reset  max |diff| = {reset_gap:.2e}")
    print(f"  obs              max |diff| = {a_obs:.2e}")
    print(f"  reward           max |diff| = {a_rew:.2e}")
    print(f"  metrics          max |diff| = {a_met:.2e}")
    print(f"  termination flags identical = {a_term}")
    ok = (reset_gap == 0.0 and a_term
          and a_obs == 0.0 and a_rew == 0.0 and a_met == 0.0)
    print(f"  single-step equivalence: {'OK' if ok else 'FAIL'} (tol 0)")

    # --- B. free-running divergence (informational) -------------------------
    _sync_rollout_from_threads(p_rollout, p_threads)
    # Cold warmstart is what the stepper actually runs with; restore it so the
    # divergence reported is the one training would see, not an idealized one.
    p_rollout._warmstart[...] = 0.0
    rng = np.random.default_rng(seed + 202)
    print("\nStepper parity B: free-running divergence (informational — chaotic "
          "amplification of the un-carried warmstart, not a defect):")
    marks = {1, 5, 10, 30, 100, 200, steps}
    worst = 0.0
    for t in range(1, max(steps, 1) + 1):
        d_obs, _, _, _ = _one_step(rng.uniform(-0.4, 0.4, size=(envs, nu)))
        worst = max(worst, d_obs)
        if t in marks:
            print(f"  after {t:>4} steps: obs max |diff| = {worst:.2e}")
    return ok


@click.command()
@click.option("--frames", default=10, help="Number of random states to compare.")
@click.option("--noise", default=0.05, help="State noise around the reference.")
@click.option("--seed", default=0, help="RNG seed.")
@click.option(
    "--clip-ids", default="CMU_016_22",
    help="Comma-separated clip IDs (small default keeps the check fast).",
)
@click.option(
    "--check-physics/--no-check-physics", default=True,
    help="Also report one-control-step divergence (jits an MJX step).",
)
@click.option(
    "--actuation", default="position",
    type=click.Choice(["torque", "position"]),
    help="Actuation mode to check parity under (the sweeps run 'position').",
)
@click.option(
    "--check-stepper/--no-check-stepper", default=True,
    help="Also check the rollout stepper against the threaded one.",
)
@click.option(
    "--stepper-envs", default=32, help="Envs used by the stepper-parity check.",
)
@click.option(
    "--stepper-steps", default=30,
    help="Control steps used by the stepper-parity check.",
)
def main(frames, noise, seed, clip_ids, check_physics, actuation,
         check_stepper, stepper_envs, stepper_steps):
    config = load_default_config()
    clip_list = [c.strip() for c in clip_ids.split(",") if c.strip()]

    # The MJX env configures collisions, actuation and timestep on the shared
    # MjModel in its constructor; build it first and hand the same (rewritten)
    # model to the CPU pool, so any divergence that shows up is in the obs /
    # reward code rather than in the two builders' model setup.
    mj_model, xml_path = build_cmu_humanoid()
    dataset = load_cmu_clips(mj_model, clip_ids=clip_list, ctrl_dt=config.ctrl_dt)
    menv = MocapTrackingEnv(
        mj_model=mj_model, dataset=dataset, config=config, impl="jax",
        actuation=actuation,
    )
    pool = MocapCpuPool(
        mj_model, dataset, config, num_envs=1, seed=seed, num_threads=1,
        actuation=actuation,
    )
    print(f"actuation = {actuation}")

    rng = np.random.default_rng(seed)
    nu = mj_model.nu

    obs_max_diff = 0.0
    comp_max_diff = {k: 0.0 for k in _COMPONENT_KEYS}
    total_max_diff = 0.0

    from mujoco import mjx  # forward mjx.Data with a ctrl for the effort term

    for f in range(frames):
        clip_start, clip_len, phase_idx, abs_idx, qpos, qvel = _sample_state(
            pool, rng, noise,
        )
        # `last_act` is the obs's last-action field and the action-rate baseline;
        # `action` is the current raw policy output; the applied ctrl is the
        # scaled/clipped command that drives the effort (actuator_force) term.
        last_act = rng.uniform(-1, 1, nu)
        action = rng.uniform(-1, 1, nu)
        applied = np.clip(
            action * config.action_scale, pool._lowers, pool._uppers
        )

        # --- CPU side ---
        d = _place_pool_env(
            pool, 0, clip_start, clip_len, phase_idx, qpos, qvel, last_act,
            ctrl=applied,
        )
        cpu_obs = pool._get_obs(0)
        cpu_total, _, _, cpu_comps = pool._get_reward(
            d, abs_idx, action, last_act,
        )

        # --- MJX side ---
        mdata = menv._init_data(jp.array(qpos, jp.float32), jp.array(qvel, jp.float32))
        # Re-forward with the applied ctrl so mdata.actuator_force matches the
        # CPU side's effort input (the initial forward runs at ctrl=0).
        mdata = mdata.replace(ctrl=jp.array(applied, jp.float32))
        mdata = mjx.forward(menv.mjx_model, mdata)
        info = {
            "clip_start": jp.int32(clip_start),
            "clip_len": jp.int32(clip_len),
            "phase_idx": jp.int32(phase_idx),
            "last_act": jp.array(last_act, jp.float32),
        }
        mjx_obs = np.array(menv._get_obs(mdata, info))
        metrics = {k: jp.zeros(()) for k in _COMPONENT_KEYS}
        mjx_total, _, _ = menv._get_reward(
            mdata, jp.int32(abs_idx), jp.array(applied, jp.float32),
            jp.array(action, jp.float32), jp.array(last_act, jp.float32), metrics,
        )

        obs_max_diff = max(obs_max_diff, float(np.abs(cpu_obs - mjx_obs).max()))
        total_max_diff = max(
            total_max_diff, abs(float(cpu_total) - float(mjx_total))
        )
        for k in _COMPONENT_KEYS:
            comp_max_diff[k] = max(
                comp_max_diff[k], abs(float(cpu_comps[k]) - float(metrics[k]))
            )

    print(f"\nCompared {frames} random states (noise={noise}):")
    print(f"  obs      max |diff| = {obs_max_diff:.2e}")
    for k in _COMPONENT_KEYS:
        print(f"  {k:<14} max |diff| = {comp_max_diff[k]:.2e}")
    print(f"  total    max |diff| = {total_max_diff:.2e}")

    obs_ok = obs_max_diff < 1e-3
    rew_ok = total_max_diff < 1e-3 and all(
        v < 1e-3 for v in comp_max_diff.values()
    )
    print(f"\n  obs parity:    {'OK' if obs_ok else 'FAIL'} (tol 1e-3)")
    print(f"  reward parity: {'OK' if rew_ok else 'FAIL'} (tol 1e-3)")

    if check_physics:
        import mujoco

        print("\nOne-control-step divergence (informational; backends use "
              "different precision/solvers):")

        @jax.jit
        def mjx_step(data, ctrl):
            return mjx_env.step(menv.mjx_model, data, ctrl, menv.n_substeps)

        qpos_gap = 0.0
        for f in range(min(frames, 5)):
            clip_start, clip_len, phase_idx, abs_idx, qpos, qvel = _sample_state(
                pool, rng, noise,
            )
            ctrl = rng.uniform(-1, 1, nu)

            d = _place_pool_env(
                pool, 0, clip_start, clip_len, phase_idx, qpos, qvel,
                np.zeros(nu),
            )
            d.ctrl[:] = np.clip(ctrl * config.action_scale, pool._lowers, pool._uppers)
            mujoco.mj_step(pool._model, d, nstep=pool._n_substeps)

            mdata = menv._init_data(
                jp.array(qpos, jp.float32), jp.array(qvel, jp.float32)
            )
            mctrl = jp.clip(
                jp.array(ctrl, jp.float32) * config.action_scale,
                menv._lowers, menv._uppers,
            )
            mdata = mjx_step(mdata, mctrl)

            qpos_gap = max(
                qpos_gap, float(np.abs(d.qpos - np.array(mdata.qpos)).max())
            )
        print(f"  qpos max |diff| after 1 ctrl step: {qpos_gap:.2e}")

    stepper_ok = True
    if check_stepper:
        # Built AFTER MocapTrackingEnv, which rewrote mj_model in place — the twin
        # has to receive the same rewrites, not the pre-rewrite defaults.
        aug_model, blk, frc = _build_rollout_twin(xml_path, config, actuation)
        sensors_ok = _check_rollout_sensors(
            pool, mj_model, aug_model, blk, frc, rng, frames, noise,
        )
        parity_ok = _check_stepper_parity(
            mj_model, aug_model, (blk, frc), dataset, config, actuation, seed,
            stepper_steps, stepper_envs,
        )
        stepper_ok = sensors_ok and parity_ok

    if not (obs_ok and rew_ok and stepper_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
