# The release benchmark

Every launchable config in this repo belongs to one benchmark, which answers two
questions for the v1 release:

1. **Does every agent work?** — all seven (DDPG, TD3, TD4, D4PG, SAC, MPO, PPO)
   on a simple task and on a hard one, at matched hyperparameters and a matched
   step budget.
2. **Does it work on CPU as well as GPU?** — the same agent, the same task, the
   same budget, with the physics and the learner moved between devices.

```bash
scripts/run_release_benchmark.sh --dry-run   # the grid, with time estimates
scripts/run_release_benchmark.sh --smoke     # tiny budgets: does it all launch?
scripts/run_release_benchmark.sh             # the real thing (~14 h, sequential)
uv run python roxie/report.py                # assemble the W&B report
```

## The grid

| Suite | Task | Cells | Runs |
|---|---|---|---|
| `walker_walk` | mujoco_playground WalkerWalk, 256 envs, 5M steps | `warp_gpu`, `mjx_gpu`, `mjx_cpu` | 7 agents × 3 |
| `mocap_cmu_006_13` | CMU humanoid tracking, single clip, 1000 envs, 50M steps | `warp_gpu`, `envpool_cpu`, `envpool_gpu` | 7 agents × 1, + 2 agents × 2 |

A **cell** is a (physics device, learner device) placement:

| Cell | Physics | Learner | Notes |
|---|---|---|---|
| `warp_gpu` | GPU (mujoco_warp) | GPU | The headline configuration. |
| `mjx_gpu` | GPU (MJX) | GPU | Same card, physics traced into XLA instead of Warp kernels. |
| `mjx_cpu` | CPU (MJX) | CPU | **Fully GPU-free.** Same env class, same trainer loop as the two above — only the device changes. |
| `envpool_cpu` | CPU (native MuJoCo pool) | CPU | **Fully GPU-free.** A genuinely different implementation of the task; see [docs/backends.md](../docs/backends.md). |
| `envpool_gpu` | CPU (native MuJoCo pool) | GPU | The hybrid, with `trainer.async_learner` on so the two devices overlap. |

Between the two suites that covers all four physics × learner placements. The
walker suite carries the full-width agent cross because its runs are minutes;
the mocap CPU cells run a subset (`MOCAP_CPU_AGENTS`, default `td3 ppo`) because
each mocap run is about an hour.

The mocap task has no `mjx_*` cell on purpose: MJX sizes contact arrays
statically to *all* potential geom pairs (~980 on this humanoid, ~75× heavier
than Warp's budgeted arena), so it is not a sensible cell for that body.

## Layout

```
experiments/
  walker/
    bench_<agent>.yaml     launchable, one per agent — this is what you run
    agent/<agent>_bench.yaml   matched hyperparameters, one per agent
    backend/<cell>.yaml    the three device cells
    bench/walker_walk.yaml shared env + trainer + logging block
    noise/bench_gaussian.yaml  exploration noise for the deterministic arms
  mocap/
    bench_<agent>.yaml     launchable, one per agent
    agent/<agent>_bench.yaml
    backend/<cell>.yaml
    bench/cmu_006_13.yaml  shared env + trainer + logging block
    env_config/cmu.yaml    physics/obs/termination settings
    reward/cmu_tracking.yaml   reward weights and kernel bandwidths
    noise/bench_gaussian.yaml
```

Run one arm directly:

```bash
uv run python roxie/train.py --config-name walker/bench_td3
uv run python roxie/train.py --config-name mocap/bench_ppo
```

Switch the cell — note the override syntax, because the group lives in a
subdirectory of the search path:

```bash
uv run python roxie/train.py --config-name walker/bench_td3 walker/backend@backend=mjx_cpu
uv run python roxie/train.py --config-name mocap/bench_ppo  mocap/backend@backend=envpool_cpu
```

A bare `backend=mjx_cpu` is rejected (`Key 'backend' is not in struct`) and
`+backend=mjx_cpu` appends a second entry instead of replacing the default;
`<dir>/backend@backend=` is the form that works.

## Held identical across every arm of a suite

Verified by composing all seven configs and diffing the resolved blocks.

| Held fixed | Where |
|---|---|
| env, parallel_envs, seed | `bench/<suite>.yaml` |
| step budget, epoch size, eval protocol | `bench/<suite>.yaml` |
| matmul precision (global — reaches networks *and* physics) | `bench/<suite>.yaml` |
| actor + critic MLP shape, layer norm | every `agent/*_bench.yaml` |
| batch size, replay ratio, buffer capacity, warmup | every off-policy `agent/*_bench.yaml` |
| gamma, tau, learning rates, grad-norm clip | every `agent/*_bench.yaml` |
| n-step horizon (where the agent takes one) | every `agent/*_bench.yaml` |
| exploration noise + its anneal (deterministic arms) | `noise/bench_gaussian.yaml` |
| actor saturation penalty (deterministic arms) | every deterministic `agent/*_bench.yaml` |

`walker` runs nets `[256, 256]`, replay ratio 5.0, 2 048-step update boundary;
`mocap` runs nets `[1024, 512, 256]`, replay ratio ~4, 8 000-step boundary. The
two suites differ from each other — they are different tasks — but never within
themselves.

## What necessarily differs (algorithmic, not tuning)

| Arm | Critic | Returns | Exploration |
|---|---|---|---|
| ddpg | 1× Q | n-step 5 | gaussian noise |
| td3 | 2× Q, policy delay 2 | n-step 5 | gaussian noise |
| td4 | 2× categorical (101 atoms) | n-step 5 | gaussian noise |
| d4pg | 1× categorical (101 atoms) | n-step 5 | gaussian noise, no target smoothing |
| sac | 2× Q | **1-step** (no `n_step` in sac.py) | entropy, auto α |
| mpo | 1× Q | **1-step** (no `n_step` in mpo.py) | policy sampling + KL duals |
| ppo | V critic | GAE(0.95), on-policy | policy entropy |

Twin vs single critic is part of the algorithm, so **per-network width** is what
is equalized, not total parameter count. The categorical support (`v_min`/
`v_max`) is task-dependent and shared by the two distributional arms within a
suite.

## What gets logged

Every metric is namespaced by what produced it. `Trainer._store_epoch_metrics`
is the single place that assigns these names — agents and envs stay unaware of
the logging scheme, and both training loops (`_run_jax`, `_run_envpool`) funnel
through it so the two paths cannot drift apart.

| Prefix | Contents |
| --- | --- |
| `epoch`, `steps` | the run axes, ungrouped. `steps` is also the wandb x-axis. |
| `train/` | behaviour policy and learner: `score`, `length`, `episodes/`, `gradient_steps`, `loss/`, `reward/` (env components), `noise/`, `mining/`, and per-agent diagnostics (`train/td3/`, `train/ppo/`). |
| `test/` | held-out eval: `score`, `length`, `distinct_starts`, `score_per_step`. Fixed reset keys, so a change here is a change in the policy. |
| `sys/` | `sps`, `time/`, `mem/`, and `gpu/` — throughput and health, never a result. |

Two rules the logging itself enforces, both learned from the v1 grid:

- **An absent metric is logged as absent, not as zero.** `train/loss/*` is
  omitted for epochs with no gradient burst, so wandb shows a gap. A logged
  `0.0` is indistinguishable from a converged loss, and that ambiguity hid a
  bug in which all six off-policy arms ran 5M steps at zero gradient steps
  through a full overnight sweep.
- **`sys/gpu/*` is only logged by runs actually on the GPU.** `nvidia-smi`
  reports the whole card, so the GPU-free `mjx_cpu` cell previously published
  another process's memory and utilisation as its own.

Runs logged before this scheme carry bare names (`score`, `sps`, `loss/actor`).
`roxie/plot.py` maps them forward on load, so an `outputs/` tree holding both
still plots; `roxie/report.py` panels are written against the current names only.

## Reading the results

- **PPO is not replay-ratio comparable.** On-policy: it discards each rollout
  after its update. Judge it on score-vs-env-steps and score-vs-wall-clock,
  never on gradient steps.
- **MPO is ~20× more expensive per gradient step** (20 action samples per state).
  It buys its matched replay ratio in wall-clock.
- **SAC/MPO run 1-step returns** because their agents take no `n_step`. That is
  an implementation gap, not a chosen handicap.
- **Rank on the back-half mean ± std, never on a peak epoch.** Warp's atomic
  contact reductions are not bit-reproducible, and a single seed per arm is a
  demonstration, not a significance claim.
- On the mocap task, `test/score / test/length ≈ 0.60` historically — score is
  substantially survival time, so read `test/length` alongside it. Health checks
  per arm: `train/loss/critic` roughly 1–3, `|train/loss/actor|` (≈ mean Q) under
  ~90, and for the deterministic arms `train/td3/tanh_grad` ≈ 0.6 (collapse below 0.05 means
  the actor has saturated its tanh and stopped learning).
- Across cells, the *score* curves should land in the same place — it is the
  same algorithm on the same task — while wall-clock and throughput are where
  the cells genuinely differ. A score gap between cells is a bug, not a result;
  `examples/mocap/check_envpool_parity.py` is the executable statement of that
  contract for the mocap task.

## Budgets

Both budgets live in the shared block of their suite and are overridable per
invocation:

```bash
WALKER_STEPS=2000000 MOCAP_STEPS=20000000 scripts/run_release_benchmark.sh
```

Change one for a *whole suite*, never for a single cell — a per-cell budget
makes score-vs-steps incomparable, which is the one thing the grid exists to
compare. The exploration-noise anneal is expressed in env steps and is sized at
40% of the budget, so if you change a budget substantially, move
`noise/bench_gaussian.yaml` with it.
