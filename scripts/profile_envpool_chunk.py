#!/usr/bin/env python
"""Where a `Rollout.collect` chunk's time goes on the EnvPool (C++ pool) path.

`EnvPoolRollout.advance` steps the pool through an ordered
`jax.experimental.io_callback` once per chunk step, inside the fused
`lax.scan`. Whether those T host crossings are worth restructuring the chunk to
avoid is an empirical question, and this script is the measurement: it runs the
real benchmark config (TD3 / dm_control / `envpool_cpu`) three times, swapping
only `advance`, and subtracts the runs from each other.

    real      the pool, through the ordered io_callback -- what the repo does
    freecb    the callback still fires, but the host side returns cached arrays
              instead of stepping the pool
    nocb      no callback at all; `advance` emits device-side constants

Everything else -- the policy forward, the replay write, the running
observation statistics, the episode bookkeeping, the learning pass -- is the
real thing in all three, so the differences isolate one term each:

    device   = nocb                    the chunk minus the environment
    callback = freecb - nocb           the host crossings and their transfers
    physics  = real - freecb           the C++ pool itself

Only PART of the `callback` term is removable by batching the crossings (a
host-side step loop, or one bulk callback per chunk): the transfers happen
either way, and it is the fixed per-crossing cost that a batch would save. The
two are separated by running more than one width -- the fixed part scales with
T, the transfer part with T * num_envs -- which `--envs` does by default.

Usage:
    uv run python scripts/profile_envpool_chunk.py
    uv run python scripts/profile_envpool_chunk.py --envs 16,64,1024 --task CheetahRun
    uv run python scripts/profile_envpool_chunk.py --mode real --envs 1024   # one cell
"""

import argparse
import collections
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parent.parent
MODES = ("real", "freecb", "nocb")

# Chunks to time. The first is dropped: it carries the loop's compile.
WARMUP_CHUNKS = 1


def _child(mode: str, envs: int, task: str, chunks: int) -> None:
    """One cell: patch `advance` for `mode`, then run the real trainer.

    A fresh process per cell rather than one process swapping implementations:
    `collect_chunk` caches a trace per `advance` function, the pool is
    stateful, and the replay buffer is sized at construction, so nothing here
    is safely re-entrant.
    """
    sys.path.insert(0, str(REPO))

    import numpy as np
    import jax
    import jax.numpy as jnp
    from jax.experimental import io_callback

    from roxie.agents.td3 import TD3
    from roxie.environment.vector import Timestep, VecState
    from roxie.utils.rollout import EnvPoolRollout, Rollout

    if mode == "freecb":
        def stepper(environment):
            """The callback, minus the pool. Returns the arrays `step_spec`
            promises, allocated once, so the crossing costs exactly what it
            costs and nothing is measured behind it."""
            spec = environment.step_spec
            cached = tuple(np.zeros(s.shape, s.dtype) for s in spec[:4]) + (
                {"nonfinite": np.zeros(spec[-1]["nonfinite"].shape, np.float32)},
            )
            return lambda action: cached

        def advance(environment, state, action, key, reset_pool, params):
            obs, reward, terminated, truncated, metrics = io_callback(
                stepper(environment), environment.step_spec, action,
                ordered=True,
            )
            return VecState(env_state=None, obs=obs), Timestep(
                obs=obs, reward=reward, terminated=terminated,
                truncated=truncated, info={"metrics": metrics},
            )

        EnvPoolRollout.advance = staticmethod(advance)

    elif mode == "nocb":
        def advance(environment, state, action, key, reset_pool, params):
            # Device-side stand-in physics, not constants: the next observation
            # has to depend on this one AND on the action, or the chunk being
            # timed is not the chunk. Feed the scan a loop-invariant body and
            # XLA collapses the policy forward across unrolled steps, and the
            # "floor" comes out FASTER than a run that does strictly less work.
            prev = state.obs
            obs = prev * 0.99 + jnp.mean(action, axis=-1, keepdims=True)
            reward = jnp.mean(obs, axis=-1)
            # Never true, but traced -- a constant `False` would let the episode
            # bookkeeping fold away.
            never = reward > 1e30
            return VecState(env_state=None, obs=obs), Timestep(
                obs=obs, reward=reward, terminated=never, truncated=never,
                info={"metrics": {"nonfinite": reward * 0}},
            )

        EnvPoolRollout.advance = staticmethod(advance)

    totals, counts = collections.Counter(), collections.Counter()

    def timed(name, fn, ready):
        """Wrap `fn` so its dispatch is charged to `name` and BLOCKED on.

        Without the block the two halves of the loop would report nonsense:
        `collect` returns as soon as it is enqueued, so an unblocked `learn`
        absorbs the acting it overlapped with.
        """
        def wrapper(self, *args, **kwargs):
            t0 = time.perf_counter()
            out = fn(self, *args, **kwargs)
            jax.block_until_ready(ready(out))
            totals[name] += time.perf_counter() - t0
            counts[name] += 1
            if counts[name] == WARMUP_CHUNKS:
                totals[name] = 0.0  # the compile is not work
            return out
        return wrapper

    Rollout.collect = timed(
        "collect", Rollout.collect, lambda out: (out[0], out[2].obs),
    )
    TD3.learn = timed("learn", TD3.learn, lambda out: out)

    def report():
        n = max(counts["collect"] - WARMUP_CHUNKS, 1)
        m = max(counts["learn"] - WARMUP_CHUNKS, 1)
        print(f"@@ {mode} {envs} {totals['collect'] / n * 1e3:.3f} "
              f"{totals['learn'] / m * 1e3:.3f} {n}", flush=True)

    import atexit
    atexit.register(report)

    # One chunk is `steps_between_updates` env steps whatever the width, so the
    # step budget buys the same number of chunks at every `--envs`.
    import runpy
    sys.argv = [
        str(REPO / "roxie" / "train.py"),
        "--config-name", "dmc/bench_td3",
        f"release.task={task}",
        f"env.parallel_envs={envs}",
        f"trainer.steps={2048 * (chunks + WARMUP_CHUNKS)}",
        "trainer.epoch_steps=1000000000",  # no epoch boundary inside the window
        "trainer.test_episodes=2",
        "trainer.show_progress=false",
        "logging.wandb.enabled=false",
        "agent.hyperparams.memory_warmup=8192",
        f"hydra.run.dir={REPO}/outputs/profile_envpool_chunk/{task}/{envs}/{mode}",
        "hydra/job_logging=disabled",
        "hydra/hydra_logging=disabled",
    ]
    runpy.run_path(str(REPO / "roxie" / "train.py"), run_name="__main__")


def _parent(widths, task, chunks, modes):
    cells = {}
    for envs in widths:
        for mode in modes:
            print(f"  {task} {envs:>5} envs  {mode}...", flush=True)
            proc = subprocess.run(
                [sys.executable, __file__, "--child", mode, str(envs),
                 task, str(chunks)],
                capture_output=True, text=True,
            )
            line = next(
                (l for l in proc.stdout.splitlines() if l.startswith("@@ ")),
                None,
            )
            if line is None:
                print(proc.stdout[-2000:], proc.stderr[-2000:], sep="\n")
                raise SystemExit(f"{mode} @ {envs} envs produced no result")
            _, _, _, collect, learn, n = line.split()
            cells[(envs, mode)] = (float(collect), float(learn), int(n))
            print(f"    collect {float(collect):7.2f} ms   learn "
                  f"{float(learn):7.2f} ms   over {n} chunks", flush=True)

    print(f"\n{task}: ms per 2048-env-step chunk, TD3 / envpool_cpu\n")
    head = (f"{'envs':>6} {'T':>4} {'device':>8} {'callback':>9} "
            f"{'physics':>8} {'collect':>8} {'learn':>8} {'cb % of iter':>13}")
    print(head)
    print("-" * len(head))
    for envs in widths:
        if not all((envs, m) in cells for m in MODES):
            continue
        real = cells[(envs, "real")][0]
        freecb = cells[(envs, "freecb")][0]
        nocb = cells[(envs, "nocb")][0]
        learn = cells[(envs, "real")][1]
        callback, physics = freecb - nocb, real - freecb
        print(f"{envs:>6} {2048 // envs:>4} {nocb:>8.2f} {callback:>9.2f} "
              f"{physics:>8.2f} {real:>8.2f} {learn:>8.2f} "
              f"{callback / (real + learn) * 100:>12.1f}%")

    if len(widths) >= 2 and all(
        (e, m) in cells for e in widths for m in MODES
    ):
        _split_callback(cells, widths)


def _split_callback(cells, widths):
    """Fit `callback = T * fixed + T * envs * per_env` over the widths run.

    The point of the fit: only the FIXED term is what batching the crossings
    could remove. The per-env term is the host<->device transfer, and a bulk
    callback moves the same bytes as T separate ones do.
    """
    rows = []
    for envs in widths:
        cb = cells[(envs, "freecb")][0] - cells[(envs, "nocb")][0]
        rows.append((2048 // envs, envs, cb))
    # Two-point solve on the extreme widths: cb = T*a + 2048*b, since
    # T * envs == 2048 for every row.
    (t_hi, _, cb_hi), (t_lo, _, cb_lo) = rows[0], rows[-1]
    if t_hi == t_lo:
        return
    a = (cb_hi - cb_lo) / (t_hi - t_lo)
    b = (cb_lo - t_lo * a) / 2048
    print(f"\n  callback ~ T x {a * 1e3:.0f}us (fixed, per crossing) "
          f"+ T x envs x {b * 1e3:.2f}us (transfer)")
    print("  batching the crossings can only remove the fixed term:")
    for t, envs, _cb in rows:
        real, learn, _ = cells[(envs, "real")]
        print(f"    {envs:>5} envs (T={t:>3}): {t * a:>6.2f} ms "
              f"= {t * a / (real + learn) * 100:>4.1f}% of an iteration")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        _, _, mode, envs, task, chunks = sys.argv
        return _child(mode, int(envs), task, int(chunks))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--envs", default="16,64,1024",
                        help="comma-separated parallel_envs widths")
    parser.add_argument("--task", default="WalkerWalk")
    parser.add_argument("--chunks", type=int, default=30,
                        help="timed chunks per cell, after the compile")
    parser.add_argument("--mode", choices=MODES, default=None,
                        help="run one implementation instead of all three")
    args = parser.parse_args()
    _parent(
        [int(e) for e in args.envs.split(",")], args.task, args.chunks,
        (args.mode,) if args.mode else MODES,
    )


if __name__ == "__main__":
    main()
