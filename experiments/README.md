# The release benchmark

Every launchable config in this repo belongs to one benchmark, which answers two
questions for the v1 release:

1. **Does every agent work, everywhere?** — all seven (DDPG, TD3, TD4, D4PG,
   SAC, MPO, PPO) on the *whole* dm_control suite, at matched hyperparameters
   and a matched step budget.
2. **Does it work on CPU as well as GPU?** — the same 25 tasks, the same agents,
   the same budget, on two independent implementations of the physics: GPU
   through `mujoco_playground`, and fully GPU-free through EnvPool's native
   MuJoCo pool.

```bash
scripts/run_release_benchmark.sh --dry-run   # the grid, with time estimates
scripts/run_release_benchmark.sh --smoke     # tiny budgets: does it all launch?
scripts/run_release_benchmark.sh             # the real thing (WEEKS, sequential)
uv run python roxie/report.py                # assemble the W&B report
uv run python roxie/plot.py --grid --path outputs/release_v1 --output grid.pdf
uv run python scripts/export_release_weights.py   # package the policies
```

## The grid

**25 tasks × 7 agents × 2 cells = 350 runs**, every one of them at 50M env steps
and 256 parallel envs.

| Cell | Physics | Learner | In the default grid |
|---|---|---|---|
| `warp_gpu` | GPU — `mujoco_playground` + mujoco_warp | GPU | yes |
| `envpool_cpu` | CPU — EnvPool's native MuJoCo pool | CPU | yes — **fully GPU-free** |
| `mjx_gpu` | GPU — `mujoco_playground` + MJX | GPU | no, one flag away |
| `mjx_cpu` | CPU — `mujoco_playground` + MJX | CPU | no, one flag away |

The tasks are `roxie.environment.suites.DMC_TASKS` — the 25 that
`mujoco_playground.registry.dm_control_suite` registers. Every one of them has an
EnvPool counterpart, which is what makes the two cells a comparison:

```
AcrobotSwingup   AcrobotSwingupSparse  BallInCup       CartpoleBalance
CartpoleBalanceSparse  CartpoleSwingup CartpoleSwingupSparse  CheetahRun
FingerSpin       FingerTurnEasy        FingerTurnHard  FishSwim
HopperHop        HopperStand           HumanoidStand   HumanoidWalk
HumanoidRun      PendulumSwingup       PointMass       ReacherEasy
ReacherHard      SwimmerSwimmer6       WalkerRun       WalkerStand
WalkerWalk
```

### The two cells are not the same code

This is the point worth being precise about. `mujoco_playground` *reimplements*
the dm_control tasks as JAX/MJX programs; EnvPool *wraps* dm_control's own C++
physics. Same task specification, two independent implementations, one 0–1000
return scale.

- A **score** gap between cells on a task is a finding about one of the two
  implementations, not a hardware artefact.
- A **wall-clock** gap is the cells doing their job.
- Their **observation vectors are not always identical** — playground omits some
  of dm_control's observation groups on a few tasks (`HumanoidRun` is 67-dim vs
  EnvPool's 95, `FingerSpin` 9 vs 12; most tasks match exactly). Both are
  self-consistent, so each cell trains and evaluates against its own spec. It
  also means **a policy trained on one cell cannot be loaded against the other**,
  which is why the weights export publishes one cell and says which.

### Escape hatches

`--cells` is a filter *and* an override: naming a cell runs it even when the
default grid omits it.

```bash
# Warp vs MJX on the same card, on a couple of tasks
scripts/run_release_benchmark.sh --cells mjx_gpu --tasks CheetahRun,WalkerWalk
# The same JAX program with no card at all
scripts/run_release_benchmark.sh --cells mjx_cpu --tasks CheetahRun
```

## Wall-clock

Measured with a learner attached on a 12-core 7900X + RTX 5080 (CheetahRun,
TD3, 256 envs):

| Cell | sps | 50M steps | × 175 runs |
|---|---:|---:|---:|
| `warp_gpu` | 6 600 | ~2h06m | ~15 days |
| `envpool_cpu` | 13 700 | ~1h00m | ~7 days |

**The whole grid is ~23 days sequential**, and runs *are* sequential on purpose —
every cell wants either the whole card or every core, so overlapping two of them
measures contention rather than the backend. Run it in task batches with
`--tasks`; the manifest makes the whole thing resumable at run granularity.

Note the direction of that table, which is the least intuitive number in the
benchmark: **on this suite the GPU cell is the slower one.** dm_control bodies
are tiny, so at 256 envs the card is nowhere near saturated and Warp's per-step
dispatch dominates, while EnvPool runs 256 cheap envs across 24 threads very
happily. The GPU wins on env *count*, not env *size* — and `parallel_envs` is
held fixed at 256 precisely so that this is visible rather than tuned away.

## Layout

```
experiments/dmc/
  bench_<agent>.yaml         launchable, one per agent — this is what you run
  agent/<agent>_bench.yaml   matched hyperparameters, one per agent
  backend/<cell>.yaml        the four device cells
  bench/dmc.yaml             shared env + trainer + logging block
  noise/bench_gaussian.yaml  exploration noise for the deterministic arms
```

**The task is an override, not a file.** 25 × 7 × 2 launchables would be 350
yamls saying "same thing, different `env_name`". `release.task` names one of the
25 and the backend group turns it into either a playground registry name or an
EnvPool task id — those differ for two of the 25 (`BallInCup` →
`BallInCupCatch-v1`, `PointMass` → `PointMassEasy-v1`), which is why the mapping
lives in `roxie/environment/suites.py` with a test on it rather than being
spelled out in yaml.

```bash
uv run python roxie/train.py --config-name dmc/bench_td3                        # WalkerWalk, warp_gpu
uv run python roxie/train.py --config-name dmc/bench_td3 release.task=CheetahRun
uv run python roxie/train.py --config-name dmc/bench_ppo release.task=HumanoidRun \
    dmc/backend@backend=envpool_cpu
```

Note the backend override's syntax, because the group lives in a subdirectory of
the search path. A bare `backend=envpool_cpu` is rejected (`Key 'backend' is not
in struct`) and `+backend=envpool_cpu` appends a second entry instead of
replacing the default; `dmc/backend@backend=` is the form that works.

## Where the results go

**One W&B project per env** — `roxie-CheetahRun`, `roxie-WalkerWalk`, 25 of them
— with the run identity inside a project being `<agent>.<cell>`, `job_type` the
cell and `group` the grid (`release-v1`).

A project is the unit W&B gives a workspace, a run table and cross-run charts
to. Making it the env means a project's default view is already the comparison
that means something: same task, same budget, 7 agents × 2 cells. The previous
scheme — one project for the whole grid with the env carried as a *tag* — put
runs with incomparable score scales on shared axes by default and needed a
filter applied before any chart said anything.

Two figures come out of it:

- `roxie/report.py` — one W&B report, a section per task, reaching across all 25
  projects.
- `roxie/plot.py --grid` — the release figure, read from the local `log.csv`
  files rather than the API: **one panel per env, every agent's curve inside
  it**, colour by agent and stroke by cell, y-axis pinned to 0–1000 so the
  panels are comparable at a glance.

## Held identical across every arm

Verified by composing all seven configs and diffing the resolved blocks.

| Held fixed | Where |
|---|---|
| step budget, epoch size, eval protocol | `bench/dmc.yaml` |
| parallel_envs (256), seed | `bench/dmc.yaml` |
| actor + critic MLP `[256, 256]`, layer norm | every `agent/*_bench.yaml` |
| batch size, replay ratio, buffer capacity, warmup | every off-policy `agent/*_bench.yaml` |
| gamma, tau, learning rates, grad-norm clip | every `agent/*_bench.yaml` |
| n-step horizon (where the agent takes one) | every `agent/*_bench.yaml` |
| exploration noise + its anneal (deterministic arms) | `noise/bench_gaussian.yaml` |
| actor saturation penalty (deterministic arms) | every deterministic `agent/*_bench.yaml` |

**Nothing is tuned per task.** That is deliberate and it is the benchmark's
central methodological choice: the grid ranks *algorithms* at matched
hyperparameters, so a task where an arm would need a different learning rate or
a wider net is a task where it scores badly and says so. Per-task tuning would
turn the figure into a tuning result.

One hyperparameter is genuinely task-shaped and survives anyway: the categorical
critics' support (`v_min`/`v_max` in `d4pg_bench.yaml` / `td4_bench.yaml`). Every
dm_control reward is a normalised tolerance in [0, 1], so at gamma 0.99 the
discounted return is bounded by ~100 on all 25 — one support covers the suite by
a property of dm_control, not by luck. A task from outside dm_control needs it
rechecked.

### Settings that are tied to the step budget

Anything measured in **env steps** silently changes meaning when the budget
moves. Changing `trainer.steps` means changing these with it:

| Setting | Rule | @ 50M |
|---|---|---|
| `noise.decay_schedule.decay_steps` | 40% of the budget, so noise reaches its floor with more than half the run left to exploit it | 20 M |
| `trainer.epoch_steps` | ~100 points of curve; an epoch boundary pauses the learner and runs an eval, so it is not free | 500 k |
| `trainer.save_steps` | ~10 checkpoints per run, which is what the weights export picks the best of | 5 M |

`memory_warmup` and the replay capacity deliberately do **not** scale: they are
matched hyperparameters, so a longer budget means more turnover through the same
buffer. `tests/test_release_weights.py` pins the anneal ratio and the cadence
divisibility so a budget change cannot quietly leave them behind.

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
is equalized, not total parameter count.

## What gets logged

Every metric is namespaced by what produced it. `Trainer._store_epoch_metrics`
is the single place that assigns these names — agents and envs stay unaware of
the logging scheme, and both training loops (`_run_jax`, `_run_envpool`) funnel
through it so the two paths cannot drift apart.

| Prefix | Contents |
| --- | --- |
| `epoch`, `steps` | the run axes, ungrouped. `steps` is also the wandb x-axis. |
| `train/` | behaviour policy and learner: `score`, `length`, `episodes/`, `gradient_steps`, `loss/`, `reward/` (env components), `noise/`, and per-agent diagnostics (`train/td3/`, `train/ppo/`). |
| `test/` | held-out eval: `score`, `length`, `distinct_starts`, `score_per_step`. Fixed reset keys, so a change here is a change in the policy. |
| `sys/` | `sps`, `time/`, `mem/`, and `gpu/` — throughput and health, never a result. |

Two rules the logging itself enforces, both learned from the v1 grid:

- **An absent metric is logged as absent, not as zero.** `train/loss/*` is
  omitted for epochs with no gradient burst, so wandb shows a gap. A logged
  `0.0` is indistinguishable from a converged loss, and that ambiguity hid a
  bug in which all six off-policy arms ran 5M steps at zero gradient steps
  through a full overnight sweep.
- **`sys/gpu/*` is only logged by runs actually on the GPU.** `nvidia-smi`
  reports the whole card, so the GPU-free cells previously published another
  process's memory and utilisation as their own.

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
- **The sparse tasks are the ones to read carefully.** One noise schedule covers
  all 25, and `AcrobotSwingupSparse` / `CartpoleSwingupSparse` are where an arm
  that needed more exploration than that will simply sit at zero. That is a
  reportable result about exploration under matched settings, not a broken run —
  say which it is rather than quietly retuning one task.
- **`test/length` is only a metric where the env can terminate early.** Most
  dm_control tasks run to the 1000-step cap regardless, so `test/length` pins
  there and carries no information; on `HopperHop`/`HumanoidRun` and friends it
  is real. `test/score_per_step` divides it out.

## Released weights

The grid's real output is not only the figure — it is trained policies.
`scripts/export_release_weights.py` packages them:

```bash
uv run python scripts/export_release_weights.py --dry-run   # what would ship
uv run python scripts/export_release_weights.py --verify    # export + read back
uv run python scripts/export_release_weights.py --archive   # + .tar.gz + SHA256SUMS
uv run python scripts/export_release_weights.py --tasks CheetahRun,WalkerWalk
```

It publishes one bundle per (task, agent) from the headline `warp_gpu` cell, for
every task the manifest has a finished run for — so a partial grid exports what
it has. Two selection rules do the work:

- **Only `ok` runs at the longest completed budget**, resolved per task. A run
  that crashed at 40% leaves checkpoints that load perfectly and are not a
  release result — and `--smoke` records its 100k runs `ok` in the same ledger,
  so "the newest `ok` run" would publish a smoke policy the moment anyone
  validated the grid after training it. Pin it with `--steps 50000000`, or
  `--any-budget` to opt out.
- **The best checkpoint, not the last.** Highest `test/score` among the steps
  that actually have a checkpoint. On a saturating arm the final checkpoint is
  measurably worse than the run's own peak.

A bundle mirrors a run dir's shape, because `play.py` resolves its config as
`<checkpoint>/../../.hydra/config.yaml`:

```
weights/CheetahRun/td3.warp_gpu/
  .hydra/config.yaml     resolved run config (play.py reads this)
  .hydra/overrides.yaml  the CLI condition the run was launched with
  checkpoints/step_<N>/  the orbax checkpoint
  metadata.json          score, provenance, source run, git commit
```

It is self-contained — copy it anywhere and it still opens:

```bash
uv run python roxie/play.py \
    --checkpoint-path weights/CheetahRun/td3.warp_gpu/checkpoints/step_<N>
```

Playback forces CPU and MJX physics, so a warp-trained bundle needs neither a
GPU nor a warp install. The checkpoint carries target networks, optimizer slots
and the observation normalizer as well as the policy, so a bundle also works as
a training restart: `resume=<bundle dir>`.

`--verify` reads each exported checkpoint back off disk and checks the payload
against the directory it landed in. That is a copy check, not a behavioural one:
confirming the policy is the one that scored means rolling it out, which is what
`play.py` is for.

## Resuming

At 50M steps an arm is 1–2 h, and the grid is weeks, so a crash is expensive.
The trainer checkpoints every `save_steps` and the benchmark script prints the
exact `resume=` command when a run dies with a checkpoint on disk. Resume into a
*different* run dir than the dead leg — the CSV backend opens `log.csv` with
`"w"` on its first row, so resuming in place truncates the curve the first leg
wrote. The resumed leg logs total env steps (the trainer seeds its counters from
the checkpoint metadata), so the two halves concatenate into one curve.
