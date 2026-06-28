"""Sanity-check the mocap tracking reward.

Unlike ``train.py``/``play.py`` this does not run a policy or step the
physics. It drives the humanoid *directly onto the reference trajectory* and
evaluates :meth:`MocapTrackingEnv._get_reward` frame by frame. Two things are
checked:

  1. **Perfect tracking** — setting ``qpos``/``qvel`` to the reference at every
     frame should give the maximum reward (``sum(weights) + w_alive``). Any
     gap exposes a bug in the reward, the reference data, or a mismatch between
     the MJX forward pass and the CPU forward pass used to bake ``body_pos``.

  2. **Noise sensitivity** — Gaussian noise of increasing scale is added to the
     reference state and the reward is re-evaluated. This shows how steep the
     reward landscape is (how fast the signal decays away from the reference),
     broken down per reward component so you can see which term dominates the
     gradient.

Run from the repo root::

    python examples/mocap/check_mocap_reward.py --clip-ids CMU_016_22
    python examples/mocap/check_mocap_reward.py --noise-scales 0,0.01,0.05,0.1
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
# Match train.py/play.py: cap JAX's pool so the warp backend (if used) has
# headroom. Harmless for the default JAX backend.
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

from examples.mocap.loader import load_mocap_env  # noqa: E402


def _make_reward_fn(menv):
    """Return a jitted ``(qpos, qvel, abs_idx) -> (reward, components)`` fn.

    Builds a fresh mjx.Data, runs a forward pass (for the end-effector world
    positions) and evaluates the env's own reward, so we exercise the exact
    code path used during training.
    """

    def reward_fn(qpos, qvel, abs_idx):
        data = menv._init_data(qpos, qvel)
        components = {
            "reward/pose": jp.zeros(()),
            "reward/vel": jp.zeros(()),
            "reward/ee": jp.zeros(()),
            "reward/root": jp.zeros(()),
        }
        total, _ = menv._get_reward(data, abs_idx, components)
        return total, components

    return jax.jit(jax.vmap(reward_fn, in_axes=(0, 0, 0)))


def _apply_noise(qpos, qvel, scale, key):
    """Add N(0, scale) noise to a reference state, keeping the root quat unit."""
    kq, kv = jax.random.split(key)
    qpos_n = qpos + scale * jax.random.normal(kq, qpos.shape)
    qvel_n = qvel + scale * jax.random.normal(kv, qvel.shape)
    quat = qpos_n[3:7]
    quat = quat / (jp.linalg.norm(quat) + 1e-8)
    qpos_n = qpos_n.at[3:7].set(quat)
    return qpos_n, qvel_n


def _summarize(rewards, components):
    rewards = np.asarray(rewards)
    parts = {k: float(np.mean(np.asarray(v))) for k, v in components.items()}
    return float(rewards.mean()), float(rewards.min()), float(rewards.max()), parts


@click.command()
@click.option(
    "--clip-ids",
    type=str,
    default=None,
    help="Comma-separated clip ids to load. Omit to load all (uses cache).",
)
@click.option(
    "--clip-index",
    type=int,
    default=0,
    help="Index (within the loaded set) of the clip to evaluate.",
)
@click.option(
    "--noise-scales",
    type=str,
    default="0,0.005,0.01,0.02,0.05,0.1,0.2,0.5",
    help="Comma-separated Gaussian noise std devs to sweep.",
)
@click.option(
    "--num-seeds",
    type=int,
    default=8,
    help="Noise realizations averaged per scale.",
)
@click.option("--impl", type=str, default="jax", help="Physics backend (jax|warp).")
@click.option("--seed", type=int, default=0)
def main(clip_ids, clip_index, noise_scales, num_seeds, impl, seed):
    clip_id_list = [c.strip() for c in clip_ids.split(",")] if clip_ids else None
    scales = [float(s) for s in noise_scales.split(",")]

    env, _, _ = load_mocap_env(clip_id_list, impl=impl)
    menv = env.env  # unwrap TerminationWrapper

    starts = np.asarray(menv._clip_starts)
    lengths = np.asarray(menv._clip_lengths)
    num_clips = len(starts)
    if not 0 <= clip_index < num_clips:
        raise SystemExit(
            f"--clip-index {clip_index} out of range (loaded {num_clips} clips)"
        )

    start = int(starts[clip_index])
    length = int(lengths[clip_index])
    abs_idx = jp.arange(start, start + length)
    ref_qpos = menv._ref_qpos[abs_idx]
    ref_qvel = menv._ref_qvel[abs_idx]

    cfg = menv._config.reward_config
    max_reward = float(
        cfg.w_pose + cfg.w_vel + cfg.w_ee + cfg.w_root + cfg.w_alive
    )

    reward_fn = _make_reward_fn(menv)

    print(
        f"Loaded {num_clips} clip(s); evaluating clip index {clip_index} "
        f"({length} frames, abs {start}..{start + length - 1})."
    )
    print(f"Theoretical max reward (perfect tracking): {max_reward:.4f}\n")

    # 1. Perfect tracking ---------------------------------------------------
    rewards, components = reward_fn(ref_qpos, ref_qvel, abs_idx)
    mean, lo, hi, parts = _summarize(rewards, components)
    gap = max_reward - mean
    print("== Perfect tracking (state == reference) ==")
    print(f"  reward  mean={mean:.5f}  min={lo:.5f}  max={hi:.5f}")
    print(f"  gap from theoretical max: {gap:.5e}")
    print(
        "  components: "
        + "  ".join(f"{k.split('/')[1]}={v:.4f}" for k, v in parts.items())
    )
    if gap > 1e-3:
        print(
            "  WARNING: reward is not perfect on the reference — check the "
            "reward terms or the reference body_pos vs MJX forward pass."
        )
    print()

    # 2. Noise sensitivity --------------------------------------------------
    print(f"== Noise sensitivity (avg over {num_seeds} seeds x {length} frames) ==")
    header = f"  {'scale':>7}  {'reward':>8}  {'%ofmax':>7}  {'pose':>6}  {'vel':>6}  {'ee':>6}  {'root':>6}"
    print(header)
    base_key = jax.random.PRNGKey(seed)
    for scale in scales:
        seed_rewards = []
        seed_parts = {k: [] for k in components}
        for s in range(num_seeds):
            base_key, k = jax.random.split(base_key)
            frame_keys = jax.random.split(k, length)
            qpos_n, qvel_n = jax.vmap(
                lambda q, v, fk: _apply_noise(q, v, scale, fk)
            )(ref_qpos, ref_qvel, frame_keys)
            r, comp = reward_fn(qpos_n, qvel_n, abs_idx)
            seed_rewards.append(float(np.mean(np.asarray(r))))
            for key_ in comp:
                seed_parts[key_].append(float(np.mean(np.asarray(comp[key_]))))
        r_mean = float(np.mean(seed_rewards))
        p = {k: float(np.mean(v)) for k, v in seed_parts.items()}
        print(
            f"  {scale:>7.3f}  {r_mean:>8.4f}  {100 * r_mean / max_reward:>6.1f}%  "
            f"{p['reward/pose']:>6.3f}  {p['reward/vel']:>6.3f}  "
            f"{p['reward/ee']:>6.3f}  {p['reward/root']:>6.3f}"
        )


if __name__ == "__main__":
    main()
