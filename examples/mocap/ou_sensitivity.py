"""Sensitivity analysis of the OU *exploration* process on the mocap task.

This drives the humanoid with the non-learning Ornstein-Uhlenbeck agent
(``roxie.agents.basic.OrnsteinUhlenbeck``) and measures how good the resulting
correlated-random exploration is at tracking the reference motion, as a function
of the OU hyperparameters (``scale``, ``theta``, ``dt``, ``clip``, ``mu``).

It is *not* training: nothing learns. The OU process is the policy. The point is
to find the exploration hyperparameters that produce the highest-quality random
behaviour on this task -- a sensible floor / prior for the learned agents' noise
modules (which share the exact same parameterization, see
``roxie.exploration.noisy.OrnsteinUhlenbeckNoise``).

Method: a one-at-a-time (OAT) sweep around a baseline. For each parameter we
vary it over a grid while holding the others at baseline, evaluate each setting
over many parallel episodes (random clip + random start), and report:

  * reward/step -- mean per-step tracking reward until the episode ends
                   (max 1.0; the 0.05 alive bonus is the rough floor).
  * length      -- mean steps survived before falling (cap = horizon).
  * return      -- mean undiscounted episode return (reward/step x length).

All (param-value x repetitions) combinations are packed into ONE batched rollout
so the physics step compiles a single time. The OU update is reimplemented inline
(identically to the agent) so the hyperparameters can be per-env arrays.

Run from the repo root::

    uv run python examples/mocap/ou_sensitivity.py --impl warp
    uv run python examples/mocap/ou_sensitivity.py --impl jax --self-collisions false
"""

import os

os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.6")

import sys
import time

import click
import jax
import jax.numpy as jp
import numpy as np
from omegaconf import OmegaConf

from roxie.utils import hydra_searchpath

sys.path.insert(0, str(hydra_searchpath.REPO_ROOT))

from examples.mocap.loader import build_mocap_env  # noqa: E402


# Baseline OU hyperparameters (match roxie/configs/agent/ou.yaml). Each sweep
# varies one of these while the rest stay here.
BASELINE = {"scale": 1.0, "theta": 0.15, "dt": 1e-2, "clip": 2.0, "mu": 0.0}

# Grids swept one-at-a-time around the baseline.
SWEEPS = {
    "scale": [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0],
    "theta": [0.0, 0.05, 0.15, 0.3, 0.6, 1.0],
    "dt": [1e-3, 5e-3, 1e-2, 2e-2, 5e-2],
    "clip": [1.0, 2.0, 3.0, 5.0],
    "mu": [-0.2, 0.0, 0.2],
}


def build_combos():
    """Return (combos, index map). combos: list of dicts, one per OAT point.

    The baseline point is shared across sweeps, so it is de-duplicated: each
    sweep records which combo index each of its grid values maps to.
    """
    combos = []
    seen = {}

    def add(params):
        key = tuple(round(params[k], 8) for k in BASELINE)
        if key not in seen:
            seen[key] = len(combos)
            combos.append(params)
        return seen[key]

    index_map = {}
    for param, values in SWEEPS.items():
        index_map[param] = []
        for v in values:
            params = dict(BASELINE)
            params[param] = v
            index_map[param].append(add(params))
    return combos, index_map


def make_rollout(env, horizon, action_low, action_high):
    """Compile a batched OU rollout returning (return, length) per env.

    Hyperparameters are passed as (N, 1) arrays so each of the N envs runs its
    own OU process. The update mirrors roxie.agents.basic.OrnsteinUhlenbeck.step
    exactly: clip the gaussian, integrate dx = theta*dt*(mu - x) + scale*sqrt(dt)*noise,
    clamp the state to [-1, 1], then scale to env units.
    """
    v_step = jax.vmap(env.step)
    action_size = env.action_size

    @jax.jit
    def rollout(states, scale, theta, dt, clip, mu, key):
        n = scale.shape[0]

        def body(carry, _):
            states, x, dones, ret, length, key = carry
            key, nkey = jax.random.split(key)
            noise = jp.clip(jax.random.normal(nkey, (n, action_size)), -clip, clip)
            x = x + theta * dt * (mu - x) + scale * jp.sqrt(dt) * noise
            x = jp.clip(x, -1.0, 1.0)
            action = action_low + 0.5 * (x + 1.0) * (action_high - action_low)

            next_states = v_step(states, action)
            not_done = ~dones
            ret = ret + next_states.env_state.reward * not_done.astype(jp.float32)
            length = length + not_done.astype(jp.int32)
            dones = dones | next_states.env_state.done.astype(bool)
            return (next_states, x, dones, ret, length, key), None

        init = (
            states,
            jp.zeros((n, action_size)),
            jp.zeros((n,), dtype=bool),
            jp.zeros((n,), dtype=jp.float32),
            jp.zeros((n,), dtype=jp.int32),
            key,
        )
        (_, _, _, ret, length, _), _ = jax.lax.scan(
            body, init, None, length=horizon
        )
        return ret, length

    return rollout


def evaluate(combos, env, horizon, reps, seed):
    """Roll out every combo x reps envs in one batch; return per-combo stats."""
    n_combos = len(combos)
    total = n_combos * reps

    ctrl_range = jp.array(env.mj_model.actuator_ctrlrange)
    action_low, action_high = ctrl_range[:, 0], ctrl_range[:, 1]

    # Per-env hyperparameter arrays: combo i repeated `reps` times.
    def col(name):
        vals = np.array([c[name] for c in combos], dtype=np.float32)
        return jp.asarray(np.repeat(vals, reps)[:, None])

    scale, theta, dt = col("scale"), col("theta"), col("dt")
    clip, mu = col("clip"), col("mu")

    rollout = make_rollout(env, horizon, action_low, action_high)

    key = jax.random.PRNGKey(seed)
    key, reset_key, roll_key = jax.random.split(key, 3)
    v_reset = jax.jit(jax.vmap(env.reset))

    print(f"Resetting {total} envs ({n_combos} combos x {reps} reps)...", flush=True)
    t0 = time.time()
    states = v_reset(jax.random.split(reset_key, total))
    jax.block_until_ready(states.env_state.obs)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    print(f"Rolling out {horizon} steps (compiling)...", flush=True)
    t0 = time.time()
    ret, length = rollout(states, scale, theta, dt, clip, mu, roll_key)
    jax.block_until_ready(ret)
    print(f"  {time.time() - t0:.1f}s", flush=True)

    ret = np.asarray(ret).reshape(n_combos, reps)
    length = np.asarray(length).reshape(n_combos, reps)
    rps = ret / np.maximum(length, 1)
    return {
        "return": ret.mean(axis=1),
        "length": length.mean(axis=1),
        "rps": rps.mean(axis=1),
        "return_sem": ret.std(axis=1) / np.sqrt(reps),
    }


@click.command()
@click.option("--impl", default="warp", help="Physics backend (jax|warp).")
@click.option("--self-collisions", default=True, type=bool)
@click.option("--gpu-clip-budget", default=50, type=int)
@click.option("--reps", default=32, type=int, help="Envs averaged per combo.")
@click.option("--horizon", default=500, type=int, help="Rollout steps (cap=1000).")
@click.option("--seed", default=0, type=int)
def main(impl, self_collisions, gpu_clip_budget, reps, horizon, seed):
    combos, index_map = build_combos()
    total = len(combos) * reps

    cfg_env = OmegaConf.create(
        {
            "parallel_envs": total,
            "impl": impl,
            "self_collisions": self_collisions,
            "gpu_clip_budget": gpu_clip_budget,
            "clip_ids": [],
            "naconmax": None,
            "njmax": None,
            "naccdmax": None,
        }
    )
    print(f"Building mocap env (impl={impl}, {len(combos)} combos)...", flush=True)
    bundle = build_mocap_env(cfg_env, mode="train")
    env = bundle.env

    stats = evaluate(combos, env, horizon, reps, seed)

    print("\n" + "=" * 64)
    print("OU EXPLORATION SENSITIVITY ON MOCAP TRACKING")
    print(f"horizon={horizon}  reps={reps}  impl={impl}  "
          f"self_collisions={self_collisions}")
    print(f"baseline: {BASELINE}")
    print("reward/step: max 1.0, ~0.05 = falling/idle floor")
    print("=" * 64)

    best = {}
    for param, idxs in index_map.items():
        print(f"\n-- {param} "
              f"(others at baseline) {'-' * (40 - len(param))}")
        print(f"  {'value':>8}  {'rew/step':>9}  {'length':>7}  {'return':>8}")
        best_i, best_rps = None, -1.0
        for v, ci in zip(SWEEPS[param], idxs):
            rps = stats["rps"][ci]
            mark = ""
            if rps > best_rps:
                best_rps, best_i = rps, v
            base = abs(v - BASELINE[param]) < 1e-9
            if base:
                mark = "  <- baseline"
            print(f"  {v:>8.4g}  {rps:>9.4f}  {stats['length'][ci]:>7.1f}  "
                  f"{stats['return'][ci]:>8.2f}{mark}")
        best[param] = best_i
        print(f"  best {param}: {best_i:.4g}  (rew/step={best_rps:.4f})")

    print("\n" + "=" * 64)
    print("RECOMMENDED (per-parameter OAT optima):")
    for k in BASELINE:
        print(f"  {k}: {best[k]:.4g}")
    print("\nNote: OAT ignores interactions. Verifying the assembled combo "
          "against\nthe baseline in a second rollout...")
    print("=" * 64)

    verify = [dict(BASELINE), {**BASELINE, **best}]
    vstats = evaluate(verify, env, horizon, reps, seed + 1)
    for name, ci in (("baseline", 0), ("recommended", 1)):
        print(f"  {name:>12}: rew/step={vstats['rps'][ci]:.4f}  "
              f"length={vstats['length'][ci]:.1f}  "
              f"return={vstats['return'][ci]:.2f} "
              f"(+/-{vstats['return_sem'][ci]:.2f})")


if __name__ == "__main__":
    main()
