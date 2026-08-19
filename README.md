<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_logo.png?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

A reinforcement learning framework in JAX for continuous control in MuJoCo. Roxie trains across hundreds to thousands of parallel environments, and it runs the *same task* on three different physics backends — MJX, mujoco_warp, and native CPU MuJoCo — so that a result can be reproduced (and a bottleneck diagnosed) on either a GPU or a many-core CPU box.

The interesting part of this repo is not the algorithms, which are standard; it is that the CPU and GPU paths are deliberately kept semantically identical while being structurally very different. [docs/backends.md](docs/backends.md) is the document to read.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Backends: CPU vs GPU](docs/backends.md) — the design centrepiece
- [Configuration](#configuration)
- [Agents](#agents)
- [Mocap tracking example](#mocap-tracking-example)
- [Tests](#tests)

## Installation

Requires Python ≥ 3.11.

```bash
uv sync                 # CPU-only: MJX on CPU + the EnvPool/native-MuJoCo path
uv sync --group cuda    # Linux + NVIDIA: adds jax[cuda12] and warp-lang
```

or, with plain pip:

```bash
pip install -e .
```

The `cuda` group pins `warp-lang>=1.11,<1.13`. This is not conservatism: MuJoCo's vendored Warp bridge imports `warp._src.jax_experimental.ffi.GraphMode` and reads `warp.types.warp_type_to_np_dtype`. Warp 1.13 dropped the latter from the public API and 1.14 graduated `jax_experimental` into `jax`, moving both out from under the bridge. Verify a Warp install with:

```bash
uv run python scripts/check_warp.py     # exit 0 = warp usable
```

## Quick start

### Train

Every run is a self-contained experiment YAML under [`experiments/`](experiments/), grouped by environment (`walker/`, `mocap/`, `envpool/`). The folder is part of the config name:

```bash
uv run python roxie/train.py --config-name walker/walker_sac
uv run python roxie/train.py --config-name mocap/sweep_ppo           # GPU (Warp) physics
uv run python roxie/train.py --config-name mocap/sweep_ppo_envpool   # CPU physics
uv run python roxie/train.py --config-name envpool/halfcheetah_td3   # CPU, gym-style task
```

Any key can be overridden from the command line (Hydra):

```bash
uv run python roxie/train.py --config-name walker/walker_sac \
    env.parallel_envs=400 trainer.save_steps=100_000
```

`device=cpu` (or `device=gpu`) is a special override parsed before JAX is imported, and forces the JAX platform for the whole process regardless of what the config says — see [Ordering gotchas](docs/backends.md#ordering-gotchas-env-vars-vs-jaxconfig).

Each run writes to its Hydra output dir: resolved config under `.hydra/`, epoch metrics to console + CSV, checkpoints under `checkpoints/`, and optionally Weights & Biases (`logging.wandb.enabled: true`).

### Play

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_500000
```

Playback drives a single world into the interactive MuJoCo viewer. It always forces CPU/MJX, even for a Warp-trained checkpoint — see [Determinism](docs/backends.md#determinism-and-reproducibility). Trailing `key=value` args override the saved run config, e.g. `env.config.early_termination=false` to watch a clip run to its end instead of resetting on tracking collapse.

---

## Backends: CPU vs GPU

Roxie can put the *physics* on the GPU or the CPU, and — independently — the *agent* (networks, optimizers, replay buffer) on the GPU or the CPU. This is the design centrepiece of the repo, and it has its own document:

**→ [docs/backends.md](docs/backends.md)**

It covers the three physics backends (MJX, mujoco_warp, native CPU MuJoCo) and how they are selected; why there are two trainer loops and what they must agree on; auto-reset, truncation vs termination, and contact budgets; the memory trade that is the actual reason to run on CPU; Warp's CUDA graph-mode leak; where the agent runs relative to the physics, with measured throughput; CPU threading limits; determinism; the env-var vs `jax.config` ordering gotchas; the parity check that keeps the backends honest; and a short guide to [choosing a backend](docs/backends.md#choosing-a-backend).

---

## Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call.** `_target_` names the class and every sibling key is one of its keyword arguments — there is no `name:`/`args:` indirection and no registry. A knob that exists in Python but is missing from the YAML fails loudly at launch instead of silently taking its default. This is enforced by [`tests/test_agent_configs.py`](tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](tests/test_agent_construction.py).

Layout:

- [`roxie/configs/agent/`](roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus baselines).
- [`roxie/configs/noise/`](roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `composite`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one to them is a launch-time error.
- [`experiments/<env>/`](experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/mocap/`](experiments/mocap/) — additionally carries its own config groups: `agent/` (per-task tunings), `reward/`, `env_config/`, `backend/`, `sweep/` (shared blocks).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know.

The env factory itself is a dotted path (`env.builder`, resolved with `hydra.utils.get_method`), so the core training loop never branches on an env-type string and never names "mocap". Same for the viewer hook (`env.viewer`).

A minimal experiment:

```yaml
# @package _global_

defaults:
  - /agent: sac
  - _self_

env:
  seed: 0
  parallel_envs: 800
  builder: roxie.environment.loader.build_playground_env
  env_name: WalkerWalk
  impl: warp

runtime:
  matmul_precision: null

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

| Agent | Type | Policy | Critic | Returns | Exploration |
|---|---|---|---|---|---|
| **DDPG** | Off-policy | Deterministic | Q(s,a) | n-step | Noise module |
| **TD3** | Off-policy | Deterministic | Twin Q(s,a), delayed policy | n-step | Noise module |
| **D4PG** | Off-policy | Deterministic | Categorical Q(s,a) | n-step | Noise module, no target smoothing |
| **TD4** | Off-policy | Deterministic | Twin categorical Q(s,a), delayed policy | n-step | Noise module |
| **SAC** | Off-policy | Stochastic (squashed Gaussian) | Twin Q(s,a) | n-step | Entropy, auto α |
| **MPO** | Off-policy | Stochastic (Gaussian) | Q(s,a) | **1-step only** | Policy sampling + KL duals |
| **PPO** | On-policy | Stochastic (Gaussian) | V(s) | GAE | Policy entropy |

All share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the `step`/`add`/`update` interface the trainer consumes. Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are available for sanity-checking an environment.

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching automatically to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Two caveats when comparing them: **PPO is not replay-ratio comparable** (on-policy — judge it on score-vs-env-steps and score-vs-wall-clock, not gradient steps), and **MPO is ~20× more expensive per gradient step** (20 action samples per state at batch 512). MPO runs 1-step returns because `mpo.py` takes no `n_step`; that is an implementation gap, not a chosen handicap.

## Mocap tracking example

A humanoid motion-capture tracking task on dm_control's CMU Humanoid (V2020), living under [`examples/mocap/`](examples/mocap/) rather than in the core package — the dependency direction is one-way, examples import from `roxie` and never the reverse.

Clips are retargeted CMU data fetched from DeepMind's public HDF5 and cached in `~/.cache/roxie`; no manual conversion step is needed. Each clip is grounded by shifting it down until the lowest foot *collision surface* (not the geom origin — the feet are 25 mm-radius capsules, and grounding on origins buries them) rests on the floor.

The env computes a weighted reward from pose matching, joint velocities, end-effector positions, and split root position/orientation/velocity terms, with early termination on NaN, tracking collapse and root drift. Rotations are exchanged with the network in the **6D continuous representation** (Zhou et al.), never quaternions or Euler angles.

Two features shape the training distribution:

- **Negative mining over start phases.** Uniform random starts spend most of their budget on clip regions already tracked well. The env keeps a per-bin failure *rate* over the clip (a rate, not a count — dying early means later phases are visited less, and a count would mistake that for competence), EMA-smoothed, and biases reset toward the failing bins as a mixture against uniform (`alpha`, kept well below 1: this is a re-weighting, not a curriculum). Watch `mining/effective_bins` — collapse toward 1 means coverage is being lost.
- **GPU clip residency.** `gpu_clip_budget` caps how many clips are resident on the GPU at once, reshuffled per epoch (`clip_swap`). A real swap invalidates in-progress episodes, whose stored clip indices reference the old chunk, so the trainer resets live envs only then. The CPU backend has no such budget — the full dataset always lives in host RAM.

The canonical eval protocol is fixed and deliberate: **start at frame 0, no reset noise, run the clip to its end.** That is the task as stated ("track this clip"), not a sample of it, and `play.py` starts at frame 0 too — so what you watch is what the metric measured.

See [`experiments/mocap/sweep/README.md`](experiments/mocap/sweep/README.md) for the single-clip overfit sweep that compares every agent on identical settings, and the helper scripts alongside the env: `check_envpool_parity.py`, `check_mocap_reward.py`, `check_openloop_tracking.py`, `check_cmu_mocap_data.py`.

## Tests

```bash
uv run pytest tests/
```

Covering models (actor/critic shapes and bounds), loss functions, exploration noise, n-step returns, agent utilities, per-agent behaviour, and — importantly — that every agent config constructs and that no `__init__` keyword is missing from its YAML.
