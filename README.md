<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_logo.png?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

[![tests](https://github.com/vittorione94/roxie/actions/workflows/tests.yml/badge.svg)](https://github.com/vittorione94/roxie/actions/workflows/tests.yml)

A reinforcement learning framework in JAX for continuous control in MuJoCo. Roxie trains across hundreds to thousands of parallel environments, and it runs the *same task* on three different physics backends — MJX, mujoco_warp, and native CPU MuJoCo — so that a result can be reproduced (and a bottleneck diagnosed) on either a GPU or a many-core CPU box.

The interesting part of this repo is not the algorithms, which are standard; it is that the CPU and GPU paths are deliberately kept semantically identical while being structurally very different. [docs/backends.md](docs/backends.md) is the document to read.

## Installation

Requires Python ≥ 3.11.

```bash
uv sync                 # CPU-only: MJX on CPU + the EnvPool/native-MuJoCo path
uv sync --group cuda    # Linux + NVIDIA: adds jax[cuda12] and warp-lang
uv run python scripts/check_warp.py     # exit 0 = warp usable
```

or, with plain pip: `pip install -e .`

The `cuda` group pins `warp-lang>=1.11,<1.13`, which is a hard constraint rather than caution — [why](docs/backends.md#the-warp-version-pin).

## Quick start

```bash
uv run python roxie/train.py --config-name walker/bench_sac    # simple task
uv run python roxie/train.py --config-name mocap/bench_ppo     # humanoid tracking
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_500000
```

Every run is a self-contained experiment YAML under [`experiments/`](experiments/), grouped by environment; the folder is part of the config name. Any key can be overridden on the command line, and switching the `backend` group moves the physics *and* the learner between devices without touching anything else.

**→ [docs/training.md](docs/training.md)** — overrides, `device=`, resuming a run, what a checkpoint carries, playback.

## Highlights

- **[Backends: CPU vs GPU](docs/backends.md)** — the design centrepiece. Three physics backends and two rollouts; the memory trade that is the actual reason to run on CPU; where the agent runs relative to the physics, with measured throughput; determinism; and the parity check that keeps the backends honest.
- **[Release benchmark](experiments/README.md)** — all seven agents on a simple task and a hard one, across the CPU/GPU placements each task can express: `walker_walk` (WalkerWalk, 256 envs, 5M steps) over `warp_gpu`/`mjx_gpu`/`mjx_cpu`, and `mocap_cmu_006_13` (CMU humanoid tracking, 1000 envs, **1B steps**) over `warp_gpu`/`envpool_gpu`. Between them the two suites cover all four (physics device, learner device) combinations, two of them **fully GPU-free**. Run it with `scripts/run_release_benchmark.sh`; package the resulting policies with `scripts/export_release_weights.py`.
- **[Environments](docs/environments.md)** — there are two ways to attach an environment, both shaped like Gymnasium's `functional_jax_env`: write a stateless `FuncEnv` and roxie batches it, or bring something already vectorized (EnvPool) that speaks the Gymnasium 5-tuple and it is driven as-is. Why the driver is roxie's own rather than upstream's, and what `terminal` versus `truncal` mean.
- **[Mocap tracking](docs/mocap.md)** — the hard task, under [`examples/mocap/`](examples/mocap/): humanoid motion-capture tracking on dm_control's CMU Humanoid, with negative mining over start phases and a fixed eval protocol.

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

All share one `Agent` base and pick their replay structure from `n_step`. Two caveats when comparing them: PPO is not replay-ratio comparable, and MPO is ~20× more expensive per gradient step.

**→ [docs/agents.md](docs/agents.md)** — the shared base, buffer selection, and what makes a fair comparison.

## Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call.** `_target_` names the class and every sibling key is one of its keyword arguments — there is no `name:`/`args:` indirection and no registry. A knob that exists in Python but is missing from the YAML fails loudly at launch instead of silently taking its default, which [`tests/test_agent_configs.py`](tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](tests/test_agent_construction.py) enforce.

**→ [docs/configuration.md](docs/configuration.md)** — the group layout, the four injected arguments, and an annotated experiment.

## Examples

Everything under [`examples/`](examples/) sits outside the core package: examples import from `roxie`, never the reverse. [`mocap/`](examples/mocap/) is the tracking task ([docs](docs/mocap.md)); [`flashbax/`](examples/flashbax/) holds two standalone walkthroughs of the replay structures the agents use.

## Tests

```bash
uv run pytest tests/
```

Covering models, loss functions, exploration noise, n-step returns, agent utilities, per-agent behaviour, and — importantly — that every agent config constructs and that no `__init__` keyword is missing from its YAML. CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the same suite on every push and pull request, on Python 3.11 and 3.12; the runners are CPU-only, so it exercises the MJX-on-CPU and EnvPool paths and never the `cuda` group.

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

When citing a *result*, please also name the backend cell and the commit — the backends are semantically but not bitwise identical, so the cell is part of the setup ([determinism](docs/backends.md#determinism-and-reproducibility)).

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

Roxie is MIT-licensed ([LICENSE](LICENSE)). It stands on JAX, MuJoCo/MJX, mujoco_warp, MuJoCo Playground, EnvPool, Flax, Optax and flashbax (see [`pyproject.toml`](pyproject.toml)), and on the CMU Motion Capture Database as retargeted by dm_control — please cite those directly where relevant.
