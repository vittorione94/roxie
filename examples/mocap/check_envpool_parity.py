"""Parity check: CPU EnvPool mirror vs the MJX mocap-tracking env.

Verifies that the numpy re-implementation in ``mocap_envpool.py`` computes the
same observations and reward components as ``MocapTrackingEnv`` (impl="jax")
when both are placed in the *identical* state. Three checks:

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

Run from the repo root::

    python examples/mocap/check_envpool_parity.py
    python examples/mocap/check_envpool_parity.py --frames 20 --no-check-physics
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

from examples.mocap.cmu_mocap_data import build_cmu_humanoid, load_cmu_clips  # noqa: E402
from examples.mocap.loader import load_default_config  # noqa: E402
from examples.mocap.mocap_envpool import MocapCpuPool  # noqa: E402
from examples.mocap.mocap_tracking import MocapTrackingEnv  # noqa: E402

_COMPONENT_KEYS = (
    "reward/pose", "reward/vel", "reward/ee", "reward/root", "reward/torque",
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


def _place_pool_env(pool, i, clip_start, clip_len, phase_idx, qpos, qvel, last_act):
    import mujoco

    d = pool._datas[i]
    mujoco.mj_resetData(pool._model, d)
    d.qpos[:] = qpos
    d.qvel[:] = qvel
    mujoco.mj_forward(pool._model, d)
    pool._clip_start[i] = clip_start
    pool._clip_len[i] = clip_len
    pool._phase_idx[i] = phase_idx
    pool._last_act[i] = last_act
    return d


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
def main(frames, noise, seed, clip_ids, check_physics):
    config = load_default_config()
    clip_list = [c.strip() for c in clip_ids.split(",") if c.strip()]

    # The MJX env configures collisions + timestep on the shared MjModel in its
    # constructor; build it first and hand the same model to the CPU pool.
    mj_model, _ = build_cmu_humanoid()
    dataset = load_cmu_clips(mj_model, clip_ids=clip_list, ctrl_dt=config.ctrl_dt)
    menv = MocapTrackingEnv(
        mj_model=mj_model, dataset=dataset, config=config, impl="jax",
    )
    pool = MocapCpuPool(
        mj_model, dataset, config, num_envs=1, seed=seed, num_threads=1,
    )

    rng = np.random.default_rng(seed)
    nu = mj_model.nu

    obs_max_diff = 0.0
    comp_max_diff = {k: 0.0 for k in _COMPONENT_KEYS}
    total_max_diff = 0.0

    for f in range(frames):
        clip_start, clip_len, phase_idx, abs_idx, qpos, qvel = _sample_state(
            pool, rng, noise,
        )
        last_act = rng.uniform(-1, 1, nu)
        ctrl = rng.uniform(-1, 1, nu)

        # --- CPU side ---
        d = _place_pool_env(
            pool, 0, clip_start, clip_len, phase_idx, qpos, qvel, last_act,
        )
        cpu_obs = pool._get_obs(0)
        cpu_total, _, cpu_comps = pool._get_reward(d, abs_idx, ctrl)

        # --- MJX side ---
        mdata = menv._init_data(jp.array(qpos, jp.float32), jp.array(qvel, jp.float32))
        info = {
            "clip_start": jp.int32(clip_start),
            "clip_len": jp.int32(clip_len),
            "phase_idx": jp.int32(phase_idx),
            "last_act": jp.array(last_act, jp.float32),
        }
        mjx_obs = np.array(menv._get_obs(mdata, info))
        metrics = {k: jp.zeros(()) for k in _COMPONENT_KEYS}
        mjx_total, _ = menv._get_reward(
            mdata, jp.int32(abs_idx), jp.array(ctrl, jp.float32), metrics,
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

    if not (obs_ok and rew_ok):
        sys.exit(1)


if __name__ == "__main__":
    main()
