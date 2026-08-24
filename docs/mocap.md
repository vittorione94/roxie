# Mocap tracking example

A humanoid motion-capture tracking task on dm_control's CMU Humanoid (V2020), living under [`examples/mocap/`](../examples/mocap/) rather than in the core package — the dependency direction is one-way, examples import from `roxie` and never the reverse.

*Back to the [README](../README.md).*

## Running it

Launchable configs live in [`experiments/mocap/`](../experiments/mocap/), one per agent:

```bash
uv run python roxie/train.py --config-name mocap/bench_ppo
uv run python roxie/train.py --config-name mocap/bench_ppo mocap/backend@backend=envpool_cpu
```

The task runs on `warp_gpu`, `envpool_cpu` or `envpool_gpu` (see [`experiments/mocap/backend/`](../experiments/mocap/backend/) and [backends.md](backends.md)); `envpool_cpu` is fully GPU-free. Watch a checkpoint with:

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_500000
```

Alongside the env there are four helper scripts: `check_envpool_parity.py`, `check_mocap_reward.py`, `check_openloop_tracking.py`, `check_cmu_mocap_data.py`.

## Clips and grounding

Clips are retargeted CMU data fetched from DeepMind's public HDF5 and cached in `~/.cache/roxie`; no manual conversion step is needed. Each clip is grounded by shifting it down until the lowest foot *collision surface* (not the geom origin — the feet are 25 mm-radius capsules, and grounding on origins buries them) rests on the floor.

## Reward and observations

The env computes a weighted reward from pose matching, joint velocities, end-effector positions, and split root position/orientation/velocity terms, with early termination on NaN, tracking collapse and root drift. Rotations are exchanged with the network in the **6D continuous representation** (Zhou et al.), never quaternions or Euler angles.

Reward weights and env settings are their own Hydra groups — [`experiments/mocap/reward/`](../experiments/mocap/reward/) and [`experiments/mocap/env_config/`](../experiments/mocap/env_config/) — so a launchable picks a block rather than restating it.

## Shaping the training distribution

Two features shape it:

- **Negative mining over start phases.** Uniform random starts spend most of their budget on clip regions already tracked well. The env keeps a per-bin failure *rate* over the clip (a rate, not a count — dying early means later phases are visited less, and a count would mistake that for competence), EMA-smoothed, and biases reset toward the failing bins as a mixture against uniform (`alpha`, kept well below 1: this is a re-weighting, not a curriculum). Watch `mining/effective_bins` — collapse toward 1 means coverage is being lost. This lives entirely in the env: on the GPU path the difficulty table *is* the env's `params`, updated through the generic `init_params` / `observe_params` / `epoch_refresh` hooks ([environments.md](environments.md)), and the CPU pool owns an identical table behind its own `epoch_refresh()`. Roxie itself has no notion of mining; the `mining/*` diagnostics reach the logger as ordinary per-step env metrics, and how often episodes actually fail is reported by `term/*`.
- **GPU clip residency.** `gpu_clip_budget` caps how many clips are resident on the GPU at once, reshuffled per epoch (`clip_swap`). A real swap invalidates in-progress episodes, whose stored clip indices reference the old chunk, so the trainer resets live envs only then. The CPU backend has no such budget — the full dataset always lives in host RAM.

## Eval protocol

The canonical eval protocol is fixed and deliberate: **start at frame 0, no reset noise, run the clip to its end.** That is the task as stated ("track this clip"), not a sample of it, and `play.py` starts at frame 0 too — so what you watch is what the metric measured.

See [`experiments/README.md`](../experiments/README.md) for the single-clip benchmark that compares every agent on identical settings.
