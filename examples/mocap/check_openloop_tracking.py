"""Open-loop reference-tracking probe for the mocap task.

Unlike ``check_mocap_reward.py`` (which teleports the state onto the reference
and only checks the reward *function*), this actually *steps the physics* with
the best possible causal command and measures how well the PD servos can follow
the clip. It answers one question: is the ~1 s tracking ceiling seen in training
a *learning/exploration* problem, or a *plant* (actuation-bandwidth) problem?

The "expert" action at each step is the control that maps exactly onto the next
reference pose. In position mode the actuator applies ``force = kp*(target - q)``
with ``target = q_lo + slope*(ctrl + 1)``, so the ctrl that requests reference
pose ``q_ref`` is the same inverse map the env uses for its reset hold command::

    ctrl = clip((q_ref - q_lo) / slope - 1, -1, 1)

Feeding that every step exercises the full deployed pipeline — the 30 ms target
EMA, the servo gains, the (now V2020-faithful) joint damping, forcerange clips,
contacts — so the resulting tracking is the *upper bound* any policy could reach
on this plant. The gap from perfect tracking (== the reward's theoretical max) is
pure actuation loss; if the expert itself can't hold tracking, no agent will.

Early termination is disabled so the whole clip is rolled out; the configured
tracking-collapse / root-drift floors are then re-evaluated in Python to report
the step at which the episode *would* have terminated.

Run from the repo root::

    python examples/mocap/check_openloop_tracking.py --clip-ids CMU_016_22
    python examples/mocap/check_openloop_tracking.py --clip-ids CMU_016_22 --kv-ratio 0.1
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

sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))

from examples.mocap.loader import load_default_config, load_mocap_env  # noqa: E402


@click.command()
@click.option(
    "--clip-ids",
    type=str,
    default=None,
    help="Comma-separated clip ids to load. Selects the clip to probe; load a "
    "single id for a deterministic clip choice.",
)
@click.option("--impl", type=str, default="jax", help="Physics backend (jax|warp).")
@click.option("--kp-scale", type=float, default=1.0, help="Servo gain scale.")
@click.option(
    "--kv-ratio",
    type=float,
    default=0.0,
    help="Optional extra explicit actuator D-term (kv = kp*kv_ratio); 0 = pure "
    "V2020 (joint damping only).",
)
@click.option("--seed", type=int, default=0)
def main(clip_ids, impl, kp_scale, kv_ratio, seed):
    clip_id_list = [c.strip() for c in clip_ids.split(",")] if clip_ids else None

    # Roll out the whole clip: disable early termination so a mid-clip collapse
    # doesn't cut the trace (we re-apply the floors in Python afterwards).
    config = load_default_config()
    config.early_termination = False
    config.random_start = False
    config.reset_noise_scale = 0.0

    _, test_wrapper, _ = load_mocap_env(
        config=config,
        clip_ids=clip_id_list,
        impl=impl,
        actuation="position",
        actuation_kp_scale=kp_scale,
        actuation_kv_ratio=kv_ratio,
    )
    menv = test_wrapper.env  # bare env: manual rollout, no autoreset

    cfg = menv._config.reward_config
    max_reward = float(cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root + cfg.w_alive)
    max_track = float(cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root)
    track_floor = menv._config.reward_termination.min_tracking_frac * max_track
    max_dist = float(menv._config.root_termination.max_dist)
    look_ahead = int(menv._config.look_ahead)

    @jax.jit
    def expert_action(ref_qpos):
        # Inverse of the affine servo map: ctrl that requests reference pose.
        return jp.clip(
            (ref_qpos[menv._act_qadr] - menv._act_q_lo) / menv._act_slope - 1.0,
            -1.0,
            1.0,
        )

    jit_reset = jax.jit(menv.reset)
    jit_step = jax.jit(menv.step)

    state = jit_reset(jax.random.PRNGKey(seed))
    clip_start = int(state.info["clip_start"])
    clip_len = int(state.info["clip_len"])
    n_steps = clip_len - look_ahead - 1 if not menv._config.cyclic else clip_len

    print(
        f"Clip: start={clip_start} len={clip_len}; rolling out {n_steps} steps "
        f"(kp_scale={kp_scale}, kv_ratio={kv_ratio}, impl={impl})."
    )
    print(f"Perfect-tracking reward (== state on reference): {max_reward:.4f}")
    print(f"Tracking-collapse floor: tracking < {track_floor:.4f}  "
          f"(={100 * menv._config.reward_termination.min_tracking_frac:.0f}% of "
          f"{max_track:.3f});  root-drift floor: root_dist > {max_dist:.3f} m\n")

    rows = []  # (pose, vel, ee, root, root_dist, tracking)
    term_step = None
    term_reason = None
    for t in range(n_steps):
        next_phase = (int(state.info["phase_idx"]) + 1) % clip_len
        ref_qpos = menv._ref_qpos[clip_start + next_phase]
        state = jit_step(state, expert_action(ref_qpos))

        m = state.metrics
        pose = float(m["reward/pose"])
        vel = float(m["reward/vel"])
        ee = float(m["reward/ee"])
        root = float(m["reward/root"])
        rdist = float(m["root_dist"])
        tracking = cfg.w_pose * pose + cfg.w_vel * vel + cfg.w_ee * ee + cfg.w_root * root
        rows.append((pose, vel, ee, root, rdist, tracking))

        if term_step is None:
            if tracking < track_floor:
                term_step, term_reason = t + 1, "tracking-collapse"
            elif rdist > max_dist:
                term_step, term_reason = t + 1, "root-drift"

    arr = np.array(rows)  # (T, 6)
    names = ["pose", "vel", "ee", "root", "root_dist", "tracking"]

    def summarize(a, label):
        m = a.mean(axis=0)
        print(f"  {label:<28s} "
              + "  ".join(f"{n}={v:.3f}" for n, v in zip(names, m))
              + f"   track%max={100 * m[5] / max_track:5.1f}")

    print("== Open-loop expert tracking ==")
    if term_step is None:
        print(f"  Survives the FULL clip ({n_steps} steps) without hitting any "
              f"termination floor.")
    else:
        secs = term_step * float(menv._config.ctrl_dt)
        print(f"  WOULD terminate at step {term_step}/{n_steps} "
              f"(~{secs:.2f}s) via {term_reason}.")
    print()
    summarize(arr, "full clip (mean/step):")
    if term_step is not None and term_step > 1:
        summarize(arr[:term_step], "up to termination:")
    print(f"\n  worst-step tracking={arr[:, 5].min():.3f} "
          f"(={100 * arr[:, 5].min() / max_track:.1f}% of max)   "
          f"max root_dist={arr[:, 4].max():.3f} m")


if __name__ == "__main__":
    main()
