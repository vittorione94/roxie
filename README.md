<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_logo.png?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

A reinforcement learning framework built on JAX for continuous control in MuJoCo environments. Roxie provides vectorized training across hundreds of parallel environments using MJX, with support for locomotion tasks from [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground) and motion-capture tracking on custom humanoid models.

## Features

- **Massively parallel training** via JAX's `vmap`/`jit` over MJX environments (800+ parallel envs by default)
- **Multiple RL algorithms** &mdash; DDPG, TD3, SAC, MPO, PPO out of the box
- **Experiment-driven configuration** &mdash; each experiment is a self-contained YAML, managed by Hydra
- **Modular actor/critic networks** built with Flax NNX, configurable per-experiment
- **Exploration noise modules** &mdash; Ornstein-Uhlenbeck, Gaussian, composite, and adaptive
- **Observation normalization** with running statistics
- **Checkpointing and evaluation** with an interactive MuJoCo viewer

## Installation

Requires Python 3.10+.

```bash
pip install -e .
```

For GPU-accelerated training, install the appropriate [JAX CUDA build](https://jax.readthedocs.io/en/latest/installation.html) for your system.

## Quick start

### Training

Every training run is defined by an experiment config in the top-level `experiments/` directory, grouped by environment into `ant/`, `walker/`, and `mocap/` subfolders. Run one with:

```bash
python -m roxie.train --config-name walker/walker_ddpg
```

Override any parameter from the command line:

```bash
python -m roxie.train --config-name walker/walker_sac env.parallel_envs=400 trainer.save_steps=100_000
```

### Evaluation

Replay a trained checkpoint in an interactive MuJoCo viewer:

```bash
python roxie/play.py --checkpoint-path outputs/2025-01-15/14-30-00/checkpoints/step_500000
```

## Experiments

Each YAML under `experiments/` is a complete experiment definition &mdash; it selects an agent, noise strategy, environment, and trainer settings &mdash; and is grouped into a per-env subfolder (`ant/`, `walker/`, `mocap/`), so the `--config-name` includes that folder (e.g. `walker/walker_ddpg`). The ant and mocap tasks are self-contained examples: their envs and loaders live under [`examples/`](examples/), wired in via the `env.builder` dotted path in each config.

The `mocap/` folder additionally carries its own config groups: agent tunings (`agent/`), reward weights (`reward/`), env config (`env_config/`), and a **`backend/`** group (`warp` = mujoco_warp GPU, `envpool` = CPU native MuJoCo) that supplies the physics-backend mechanics (`env.builder`, `env.impl`, solver budgets). Each launchable picks a backend in its `defaults:` &mdash; e.g. `mocap/td3_warp` and `mocap/td3_envpool` are the same agent on the two backends, each keeping its own tuning.

To create a new experiment, add a YAML under the matching `experiments/<env>/` folder. It composes agent and noise configs from `roxie/configs/`:

```yaml
# @package _global_

defaults:
  - /agent: sac
  - /noise: ou
  - _self_

env:
  seed: 0
  parallel_envs: 800
  env_type: playground
  env_name: HumanoidStand

trainer:
  steps: 1_000_000_000
  epoch_steps: 100_000
  save_steps: 500_000
  test_episodes: 5
  show_progress: true
  replace_checkpoint: false

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

## Agents

| Agent | Type | Policy | Critic | Buffer |
|---|---|---|---|---|
| **DDPG** | Off-policy | Deterministic | Q(s,a) | Flat replay buffer |
| **TD3** | Off-policy | Deterministic | Twin Q(s,a) | Flat replay buffer |
| **SAC** | Off-policy | Stochastic (squashed Gaussian) | Twin Q(s,a) | Flat replay buffer |
| **MPO** | Off-policy | Stochastic (Gaussian) | Q(s,a) | Flat replay buffer |
| **PPO** | On-policy | Stochastic (Gaussian) | V(s) | Trajectory queue |

All agents share a common base class (`roxie.agents.agent.Agent`) that handles observation normalization, checkpointing, and the step/add/update interface consumed by the trainer.

Baseline agents (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are also available for sanity-checking environments.

## Mocap tracking

Roxie includes a motion-capture tracking environment for humanoid control. Reference motions are stored as `.npz` files containing joint positions, velocities, and body positions.

To convert clips from the CMU mocap dataset:

```bash
python -m roxie.data.convert_cmu --clip-id 03_01 --output-dir roxie/data
```

The `MocapTrackingEnv` computes a weighted reward from pose matching, joint velocity tracking, end-effector positions, and root alignment, with early termination on falls.

## Tests

```bash
pytest tests/
```

Tests cover models (actor/critic shapes and bounds), loss functions, exploration noise, and agent utilities.
