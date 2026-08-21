<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_logo.png?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

[![tests](https://github.com/vittorione94/roxie/actions/workflows/tests.yml/badge.svg)](https://github.com/vittorione94/roxie/actions/workflows/tests.yml)

A reinforcement learning framework in JAX for continuous control in MuJoCo. Roxie trains across hundreds to thousands of parallel environments, and it runs the *same task* on three different physics backends — MJX, mujoco_warp, and native CPU MuJoCo — so that a result can be reproduced (and a bottleneck diagnosed) on either a GPU or a many-core CPU box.

The interesting part of this repo is not the algorithms, which are standard; it is that the CPU and GPU paths are deliberately kept semantically identical while being structurally very different. [docs/backends.md](docs/backends.md) is the document to read.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Backends: CPU vs GPU](docs/backends.md) — the design centrepiece
- [Release benchmark](#release-benchmark) — every agent, both tasks, CPU and GPU
- [Configuration](#configuration)
- [Agents](#agents)
- [Examples](#examples)
- [Tests](#tests)
- [Citation](#citation)

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

Every run is a self-contained experiment YAML under [`experiments/`](experiments/), grouped by environment (`walker/`, `mocap/`). The folder is part of the config name:

```bash
uv run python roxie/train.py --config-name walker/bench_sac    # simple task
uv run python roxie/train.py --config-name mocap/bench_ppo     # humanoid tracking
```

Those configs are the [release benchmark](experiments/README.md): one launchable per agent per task, at matched hyperparameters. Move a run between devices by switching its `backend` group — the physics and the learner move together, and nothing else changes:

```bash
uv run python roxie/train.py --config-name walker/bench_sac walker/backend@backend=mjx_cpu
uv run python roxie/train.py --config-name mocap/bench_ppo  mocap/backend@backend=envpool_cpu
```

(The group lives in a subdirectory of the search path, so it needs the full `<dir>/backend@backend=` form; a bare `backend=` is rejected by Hydra's struct check.)

Any key can be overridden from the command line (Hydra):

```bash
uv run python roxie/train.py --config-name walker/bench_sac \
    env.parallel_envs=400 trainer.save_steps=100_000
```

`device=cpu` (or `device=gpu`) is a special override, parsed out of `sys.argv` before JAX is imported and hidden from Hydra, which forces the JAX platform for the whole process regardless of what the config says. The configs express the same thing durably as `runtime.jax_platform` — see [Ordering gotchas](docs/backends.md#ordering-gotchas-env-vars-vs-jaxconfig).

Each run writes to its Hydra output dir: resolved config under `.hydra/`, epoch metrics to console + CSV, checkpoints under `checkpoints/`, and optionally Weights & Biases (`logging.wandb.enabled: true`).

### Resume

```bash
uv run python roxie/train.py --config-name walker/bench_td3 resume=outputs/<run>
```

`resume=` takes the run directory, its `checkpoints/` dir, or one `step_<N>` dir; given a directory it picks the **highest** step. Like `device=`, it is parsed out of `sys.argv` before Hydra and is not a config key (a run that wants it recorded can set `resume.path` in its yaml instead).

The agent is built from the **config**, and only its numbers come from the checkpoint — so a resume may legitimately raise `trainer.steps` or retune a knob, unlike `play.py`, which rebuilds the agent from the checkpoint's own hyperparameters. What comes back: the networks and their targets, the optimizer moments, the observation-normalization statistics, each agent's own extra state (the exploration schedule's step counter, SAC's temperature, MPO's Lagrange duals — see `Agent._checkpoint_modules`), and the trainer's progress. `trainer.steps` is a **total**, so the resumed leg runs until the whole budget is spent and the logged x-axis continues the same curve rather than starting a second one at 0.

The replay buffer is the one thing not saved by default — it dominates a checkpoint's size and its host RAM while writing. Without it a resumed off-policy run replays its warmup to refill the buffer before learning again; with `trainer.save_buffer: true` the checkpoint carries it and the resume is exact:

```bash
uv run python roxie/train.py --config-name walker/bench_td3 trainer.save_buffer=true
```

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

## Release benchmark

The launchable configs in [`experiments/`](experiments/) are one benchmark, and it is what the release figure is made of: **all seven agents on a simple task and on a hard one, across the CPU/GPU placements each task can express.**

```bash
scripts/run_release_benchmark.sh --dry-run   # the grid, with time estimates
scripts/run_release_benchmark.sh --smoke     # tiny budgets: does everything launch?
scripts/run_release_benchmark.sh             # the real thing, ~14 h, strictly sequential
uv run python roxie/report.py                # assemble the W&B report
uv run python roxie/plot.py --path outputs/release_v1 --output release.pdf
```

| Suite | Task | Cells |
|---|---|---|
| `walker_walk` | mujoco_playground WalkerWalk, 256 envs, 5M steps | `warp_gpu`, `mjx_gpu`, `mjx_cpu` |
| `mocap_cmu_006_13` | CMU humanoid tracking, one clip, 1000 envs, 50M steps | `warp_gpu`, `envpool_cpu`, `envpool_gpu` |

A *cell* is a (physics device, learner device) placement; between the two suites they cover all four combinations, including two that are **fully GPU-free**. Within a suite everything except the algorithm and the cell is held identical — env, budget, network shape, batch size, replay ratio, exploration schedule — so score-vs-steps ranks the algorithms and the wall-clock panels price the backends. Runs stream to one W&B project with `group` = suite and `job_type` = cell, which is the structure [`roxie/report.py`](roxie/report.py) rebuilds the report from.

The script is resumable (a manifest records each finished run), it refuses to start a GPU cell while another process holds the card, and it checks the W&B credential up front — because the benchmark configs set `relogin: false`, and an interactive login prompt would deadlock an overnight grid.

**→ [experiments/README.md](experiments/README.md)** — the grid, what is held fixed, what necessarily differs, and how to read the result.

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
  save_buffer: false   # checkpoint the replay buffer too (exact resume, big files)

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

## Examples

Everything under [`examples/`](examples/) sits outside the core package: examples import from `roxie`, never the reverse.

- **[Mocap tracking](docs/mocap.md)** ([`examples/mocap/`](examples/mocap/)) — humanoid motion-capture tracking on dm_control's CMU Humanoid, the hard task in the release benchmark. Clip fetching and grounding, the tracking reward, negative mining over start phases, GPU clip residency, and the fixed eval protocol.
- **Flashbax buffers** ([`examples/flashbax/`](examples/flashbax/)) — two standalone walkthroughs of the replay structures the agents use: the flat buffer and the trajectory queue.

## Tests

```bash
uv run pytest tests/
```

Covering models (actor/critic shapes and bounds), loss functions, exploration noise, n-step returns, agent utilities, per-agent behaviour, and — importantly — that every agent config constructs and that no `__init__` keyword is missing from its YAML.

CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the same suite on every push and pull request, on Python 3.11 and 3.12. The runners are CPU-only, so it exercises the default `uv sync` resolution — the MJX-on-CPU and EnvPool paths — and never the `cuda` group.

## Citation

If Roxie is useful in your research, please cite it:

```bibtex
@software{labarbera2026roxie,
  author  = {La Barbera, Vittorio},
  title   = {Roxie: reinforcement learning in {JAX} for continuous control in {MuJoCo}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/vittorione94/roxie},
  license = {MIT}
}
```

When citing a *result* rather than the code, please also say which backend cell produced it — `warp_gpu`, `mjx_gpu`, `mjx_cpu`, `envpool_cpu` or `envpool_gpu` — and the commit hash. The backends are kept semantically identical but not bitwise identical, so the cell is part of the experimental setup; see [Determinism](docs/backends.md#determinism-and-reproducibility).

### Acknowledgements

Roxie would not exist without [Tonic](https://github.com/fabiopardo/tonic) by Fabio Pardo, whose design is the direct inspiration for this codebase — one agent interface shared across algorithms, configuration as the thing you actually launch, and a benchmark that holds everything but the algorithm fixed. Please cite it too:

```bibtex
@article{pardo2020tonic,
  author  = {Pardo, Fabio},
  title   = {Tonic: A Deep Reinforcement Learning Library for Fast Prototyping and Benchmarking},
  journal = {arXiv preprint arXiv:2011.07537},
  year    = {2020}
}
```

Roxie is MIT-licensed ([LICENSE](LICENSE)). It builds on work that has its own citations — please cite those directly where relevant: [JAX](https://github.com/jax-ml/jax), [MuJoCo](https://github.com/google-deepmind/mujoco) and MJX, [mujoco_warp](https://github.com/google-deepmind/mujoco_warp), [MuJoCo Playground](https://github.com/google-deepmind/mujoco_playground), [EnvPool](https://github.com/sail-sg/envpool), [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax) and [flashbax](https://github.com/instadeepai/flashbax), and the [CMU Motion Capture Database](http://mocap.cs.cmu.edu/) as retargeted by [dm_control](https://github.com/google-deepmind/dm_control) for the [mocap example](docs/mocap.md).
