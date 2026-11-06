# Running trainings

Launching a run, resuming one, and watching a checkpoint.

*Back to the [README](../README.md).*

## Train

Every run is a self-contained experiment YAML under [`experiments/`](../experiments/), grouped by environment (`dmc/`). The folder is part of the config name:

```bash
uv run python roxie/train.py --config-name dmc/bench_sac                          # WalkerWalk
uv run python roxie/train.py --config-name dmc/bench_sac release.task=CheetahRun  # any of the 25
```

Move a run between devices by switching its `backend` group; the physics and the learner move together and nothing else changes:

```bash
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=batch_cpu  # GPU-free
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=mjx_cpu
```

The group lives in a subdirectory of the search path, so it needs the full `<dir>/backend@backend=` form; a bare `backend=` is rejected by Hydra's struct check.

Any key can be overridden from the command line:

```bash
uv run python roxie/train.py --config-name dmc/bench_sac \
    env.parallel_envs=400 trainer.save_steps=100_000
```

The env is a config group like the agent — [`roxie/configs/env/`](../roxie/configs/env/) holds one complete `env:` block per builder ([configuration.md](configuration.md)), pulled into a launchable's `defaults:` as `- /env: playground`. The benchmark launchables do **not** use it: their env block comes from the `backend` group. Change their task with `release.task=` and anything else with a plain key override (`env.impl=warp`, `env.max_episode_steps=500`).

`device=cpu` (or `device=gpu`) is a special override, parsed out of `sys.argv` before JAX is imported and hidden from Hydra, forcing the JAX platform for the whole process regardless of the config (and suppressing the banner's device checks, since it *is* the deliberate override). The configs express the same thing durably as `runtime.device` — see [Where the run runs](backends.md#where-the-run-runs) and [Ordering gotchas](backends.md#ordering-gotchas-env-vars-vs-jaxconfig).

Each run writes to its Hydra output dir: resolved config under `.hydra/`, epoch metrics to console + CSV, checkpoints under `checkpoints/`, and optionally Weights & Biases (`logging.wandb.enabled: true`).

## Several tasks

A multi-task launchable names its tasks in an env block ([environments.md](environments.md#several-tasks-at-once)). 

The env block's `weights` fix each task's share of the worlds for the run. A **curriculum** steers what the learner samples from them instead: at every epoch boundary it reads each task's training score and sets the replay's task weights for the passes that follow. [`roxie/configs/curriculum/`](../roxie/configs/curriculum/) holds `learning_progress`, which favours the tasks whose score is moving, and `weighted`, which pins fixed weights. A curriculum needs a multi-task env and an off-policy agent; `train.py` refuses anything else at launch.

Each task logs under its own name:

| Key | What |
|---|---|
| `train/task/<name>/score` | mean return of the task's episodes that finished this epoch; absent when none did |
| `train/task/<name>/reward` | mean per-step reward over the task's worlds |
| `train/task/<name>/weight` | the replay weight the curriculum chose for the next epoch |
| `test/task/<name>/score` | the task's eval return, over its own eval worlds |

A resume restarts the curriculum's averages.

`play.py` and `scripts/render_gif.py` replay one task of a multi-task run at a time, named with `--task-id`. The world runs that task's own env, and the policy gets its observation padded to the trained width with the task's one-hot appended; the checkpoint records each task's width, so a replay env that disagrees is refused rather than misread:

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/500000 --task-id walk
```

A replay always runs its physics in C MuJoCo, whatever the run trained on. A run keeps `mujoco_playground`'s task code for its reset, observations and reward, with only the physics calls leaving for C.

## Resume

```bash
uv run python roxie/train.py --config-name dmc/bench_td3 resume=outputs/<run>
```

`resume=` takes the run directory, its `checkpoints/` dir, or one `<N>` step dir; given a directory it picks the **highest** step. Like `device=`, it is parsed out of `sys.argv` before Hydra and is not a config key (a run that wants it recorded can set `resume.path` in its yaml).

The agent is built from the **config**, and only its numbers come from the checkpoint — so a resume may legitimately raise `trainer.steps` or retune a knob, unlike `play.py`, which rebuilds the agent from the checkpoint's own hyperparameters. What comes back: the networks and their targets, the optimizer moments, the observation-normalization statistics, each agent's own extra state (`Agent._checkpoint_modules`), and the trainer's progress. `trainer.steps` is a **total**, so the resumed leg runs until the whole budget is spent and the logged x-axis continues the same curve.

The replay buffer is the one thing not saved by default — it dominates a checkpoint's size and its host RAM while writing. Without it a resumed off-policy run replays its warmup to refill the buffer before learning again; with `trainer.save_buffer: true` the checkpoint carries it and the resume is exact:

```bash
uv run python roxie/train.py --config-name dmc/bench_td3 trainer.save_buffer=true
```

Resume into a *different* run dir than the dead leg: the CSV backend opens `log.csv` with `"w"` on its first row, so resuming in place truncates the curve the first leg wrote.

## Play

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/500000
```

Playback drives a single world into the interactive MuJoCo viewer, always on CPU/MJX even for a Warp-trained checkpoint — see [Determinism](backends.md#determinism-and-reproducibility). Trailing `key=value` args override the saved run config.

Both entry points restore the checkpoint through `roxie.utils.playback`, which reads the run's saved Hydra config and rewrites it onto the C physics above. C and the engine a run trained on are not numerically identical, so a replay's score can differ a little from the run's own eval.

### Rendering without a display

The viewer needs a screen. To get frames instead — for a README, or over ssh:

```bash
uv run python scripts/render_gif.py \
    --checkpoint-path outputs/<run>/checkpoints/500000 \
    --output images/humanoid-walk.gif --seconds 6 --skip 1
```

It rolls the policy, keeps `(qpos, qvel)` per step, and replays them through an offscreen MuJoCo renderer on EGL. `--camera` picks one of the model's own cameras; left empty it tracks the root body, which `--distance`, `--azimuth` and `--elevation` then frame. `--episodes` rolls several and renders the best-scoring one, which is only worth raising for a task whose reset randomizes — most of the dm_control humanoid tasks start from a fixed pose, so with a deterministic actor every episode retraces the first.
