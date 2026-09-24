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
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=envpool_cpu  # GPU-free
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

Both entry points restore the checkpoint through `roxie.utils.playback.load_policy`, which reads the run's saved Hydra config and rewrites the two backends that cannot replay themselves: Warp falls back to MJX on the same model, and an EnvPool run moves to the `mujoco_playground` twin of the same task, since a C++ pool exposes no `mjModel`/`mjData` to step or draw one world of.

### Rendering without a display

The viewer needs a screen. To get frames instead — for a README, or over ssh:

```bash
uv run python scripts/render_gif.py \
    --checkpoint-path outputs/<run>/checkpoints/500000 \
    --output images/humanoid-walk.gif --seconds 6 --skip 1
```

It rolls the policy, keeps `(qpos, qvel)` per step, and replays them through an offscreen MuJoCo renderer on EGL. `--camera` picks one of the model's own cameras; left empty it tracks the root body, which `--distance`, `--azimuth` and `--elevation` then frame. `--episodes` rolls several and renders the best-scoring one, which is only worth raising for a task whose reset randomizes — most of the dm_control humanoid tasks start from a fixed pose, so with a deterministic actor every episode retraces the first.
