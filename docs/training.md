# Running trainings

Launching a run, resuming one, and watching a checkpoint.

*Back to the [README](../README.md).*

## Train

Every run is a self-contained experiment YAML under [`experiments/`](../experiments/), grouped by environment (`dmc/`). The folder is part of the config name:

```bash
uv run python roxie/train.py --config-name dmc/bench_sac                          # WalkerWalk
uv run python roxie/train.py --config-name dmc/bench_sac release.task=CheetahRun  # any of the 25
```

Those configs are the [release benchmark](../experiments/README.md): one launchable per agent, with the dm_control task as an override. Move a run between devices — and, here, between two independent implementations of the same task — by switching its `backend` group; the physics and the learner move together and nothing else changes:

```bash
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=envpool_cpu  # GPU-free
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=mjx_cpu
```

A task can also live in its own repo: roxie's search-path plugin picks up `./experiments` as well as its own, so `--config-name mocap/bench_ppo` run from a [roxie-mocap](https://github.com/vittorione94/roxie-mocap) checkout composes against roxie's shared groups exactly like the ones here.

The group lives in a subdirectory of the search path, so it needs the full `<dir>/backend@backend=` form; a bare `backend=` is rejected by Hydra's struct check.

Any key can be overridden from the command line (Hydra):

```bash
uv run python roxie/train.py --config-name dmc/bench_sac \
    env.parallel_envs=400 trainer.save_steps=100_000
```

`device=cpu` (or `device=gpu`) is a special override, parsed out of `sys.argv` before JAX is imported and hidden from Hydra, which forces the JAX platform for the whole process regardless of what the config says. The configs express the same thing durably as `runtime.jax_platform` — see [Ordering gotchas](backends.md#ordering-gotchas-env-vars-vs-jaxconfig).

Each run writes to its Hydra output dir: resolved config under `.hydra/`, epoch metrics to console + CSV, checkpoints under `checkpoints/`, and optionally Weights & Biases (`logging.wandb.enabled: true`).

## Resume

```bash
uv run python roxie/train.py --config-name dmc/bench_td3 resume=outputs/<run>
```

`resume=` takes the run directory, its `checkpoints/` dir, or one `step_<N>` dir; given a directory it picks the **highest** step. Like `device=`, it is parsed out of `sys.argv` before Hydra and is not a config key (a run that wants it recorded can set `resume.path` in its yaml instead).

The agent is built from the **config**, and only its numbers come from the checkpoint — so a resume may legitimately raise `trainer.steps` or retune a knob, unlike `play.py`, which rebuilds the agent from the checkpoint's own hyperparameters. What comes back: the networks and their targets, the optimizer moments, the observation-normalization statistics, each agent's own extra state (the exploration schedule's step counter, SAC's temperature, MPO's Lagrange duals — see `Agent._checkpoint_modules`), and the trainer's progress. `trainer.steps` is a **total**, so the resumed leg runs until the whole budget is spent and the logged x-axis continues the same curve rather than starting a second one at 0.

The replay buffer is the one thing not saved by default — it dominates a checkpoint's size and its host RAM while writing. Without it a resumed off-policy run replays its warmup to refill the buffer before learning again; with `trainer.save_buffer: true` the checkpoint carries it and the resume is exact:

```bash
uv run python roxie/train.py --config-name dmc/bench_td3 trainer.save_buffer=true
```

## Play

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_500000
```

Playback drives a single world into the interactive MuJoCo viewer. It always forces CPU/MJX, even for a Warp-trained checkpoint — see [Determinism](backends.md#determinism-and-reproducibility). Trailing `key=value` args override the saved run config, e.g. `env.config.early_termination=false` to watch a clip run to its end instead of resetting on tracking collapse.
