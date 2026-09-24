<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie-logo-transparent.svg?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

[![tests](https://github.com/vittorione94/roxie/actions/workflows/tests.yml/badge.svg)](https://github.com/vittorione94/roxie/actions/workflows/tests.yml)

A reinforcement learning framework in JAX for continuous control in MuJoCo. Seven agents, hundreds to thousands of parallel environments, and three physics backends — MJX, mujoco_warp and native CPU MuJoCo — driven through one interface.

<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/humanoid-walk.gif?raw=true" alt="A trained humanoid walking in MuJoCo" width="480">
</p>
<p align="center"><sub>TD4 on dm_control's <code>HumanoidWalk</code> after 455M environment steps — evaluation return 989 / 1000.<br>
Replayed from the released checkpoint with <a href="scripts/render_gif.py"><code>scripts/render_gif.py</code></a>.</sub></p>
          
## Installation

Requires Python ≥ 3.11.

```bash
uv sync                 # CPU-only: MJX on CPU + the EnvPool/native-MuJoCo path
uv sync --group cuda    # Linux + NVIDIA: adds jax[cuda12] and warp-lang
uv run python scripts/check_warp.py     # exit 0 = warp usable
```

or, with plain pip: `pip install -e .`. The `cuda` group pins `warp-lang>=1.11,<1.13` ([why](docs/backends.md#the-warp-version-pin)).

## Quick start

```bash
uv run python roxie/train.py --config-name dmc/bench_sac                          # WalkerWalk, GPU
uv run python roxie/train.py --config-name dmc/bench_sac release.task=CheetahRun  # any of the 25
uv run python roxie/train.py --config-name dmc/bench_sac dmc/backend@backend=envpool_cpu  # GPU-free
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_5000000
```

Every run is a self-contained experiment YAML under [`experiments/`](experiments/), grouped by environment; the folder is part of the config name. Any key can be overridden on the command line, and switching the `backend` group moves the physics *and* the learner between devices.

### Logging and plots

Every run writes console output and a `log.csv` into its Hydra output dir, always.
Weights & Biases is opt-in: the `logging.wandb` block in
[`experiments/dmc/bench/dmc.yaml`](experiments/dmc/bench/dmc.yaml) turns it on for the
benchmark configs, and any key in it is a normal override.

`roxie/plot.py` reads the CSVs, never the W&B API, so the figures work offline.

```bash
wandb login
uv run python roxie/train.py --config-name dmc/bench_sac release.task=HumanoidWalk \
    logging.wandb.project=my-roxie          # or logging.wandb.enabled=false

uv run python roxie/plot.py --path outputs/release_v1/HumanoidWalk/envpool_cpu/sac \
    --output sac.pdf                        # one run: score, losses, throughput, wall-clock
uv run python roxie/plot.py --path outputs/release_v1/HumanoidWalk/envpool_cpu \
    --output walk-cpu.pdf                   # a directory: every run under it, overlaid
```

Add `--grid` to get the release figures instead — see
[the release benchmark](docs/benchmark.md).

## Results

Seven agents on the three hardest dm_control humanoid tasks, at matched hyperparameters
and a matched 500M-step budget, 1024 parallel envs, on two independent implementations of
the physics — **3 tasks × 7 agents × 2 cells**.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/vittorione94/roxie/blob/main/images/release-dm_control-cpu-dark.png?raw=true">
  <img src="https://github.com/vittorione94/roxie/blob/main/images/release-dm_control-cpu.png?raw=true" alt="Episode return for seven agents on three dm_control humanoid tasks, against environment steps and against wall-clock time, on CPU">
</picture>

That is the **GPU-free** cell: EnvPool over dm_control's own C++ MuJoCo. The GPU figure,
the per-backend cost of an environment step, the final-score table and the method — what
is held identical, what necessarily differs, how to run the grid — are in
**[the release benchmark](docs/benchmark.md)**.

## Architecture

One config builds an env and an agent; `Trainer` then alternates a compiled
**acting chunk** with a **learning pass** until the step budget runs out.
Everything backend-specific is behind `Rollout`, so the loop above it never
learns which physics it is driving.

<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_workflow.png?raw=true" alt="Roxie architecture: experiments/*.yaml feeds train.py, which builds the env and agent and hands them to Trainer; Trainer alternates a compiled Rollout chunk with an Agent learning pass; Rollout descends through vector.py and FuncEnv to the three physics backends, the Agent through models, losses, exploration and the replay manager; runs write checkpoints and log.csv for plot.py and play.py">
</p>

The two boxes in the middle are where the compute goes, and both are *one*
XLA program each: the chunk fuses acting, physics, replay writes and episode
bookkeeping across a whole update window, and the learning pass fuses its
gradient steps with `lax.scan`. Nothing in either touches the host, which is
what keeps a thousand environments fed — see
[backends](docs/backends.md#why-there-are-two-rollouts).

## Documentation

- **[Backends](docs/backends.md)** — the three physics backends, the two rollouts, memory, contact budgets, device placement, determinism.
- **[Environments](docs/environments.md)** — the two ways to attach an env, the `FuncEnv` interface, builders, `terminal` vs `truncal`.
- **[Agents](docs/agents.md)** — the shared base, the hyperparameter dataclass, the fused window, the contract a new agent honours, buffers, checkpoints.
- **[Configuration](docs/configuration.md)** — the Hydra convention, the config groups, an annotated experiment.
- **[Training](docs/training.md)** — overrides, `device=`, resuming, playback.
- **[Release benchmark](docs/benchmark.md)** — the 3 × 7 × 2 grid, the figures, and the weights export.

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

All share one `Agent` base and pick their replay structure from `n_step`.

## Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call.** `_target_` names the class and every sibling key is one of its keyword arguments — no `name:`/`args:` indirection, no registry. A knob that exists in Python but is missing from the YAML fails at launch, which [`tests/test_agent_configs.py`](tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](tests/test_agent_construction.py) enforce.

## Examples

Everything under [`examples/`](examples/) sits outside the core package: examples import from `roxie`, never the reverse. [`flashbax/`](examples/flashbax/) holds two standalone walkthroughs of the replay structures the agents use.

A task can live outside this repo and still be launched through `roxie.train` — roxie's Hydra search-path plugin picks up `./experiments` as well as its own.

## Tests

```bash
uv run pytest tests/
```

Covers models, losses, exploration noise, n-step returns, agent utilities, per-agent behaviour, and that every agent config constructs with no `__init__` keyword missing from its YAML. CI ([`.github/workflows/tests.yml`](.github/workflows/tests.yml)) runs the same suite on Python 3.11 and 3.12; the runners are CPU-only, so it exercises the MJX-on-CPU and EnvPool paths and never the `cuda` group.

## Citation

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

When citing a result, name the backend cell and the commit — the backends are semantically but not bitwise identical ([determinism](docs/backends.md#determinism-and-reproducibility)).

### Acknowledgements

Roxie's design follows [Tonic](https://github.com/fabiopardo/tonic) by Fabio Pardo — one agent interface shared across algorithms, configuration as the thing you launch, and a benchmark that holds everything but the algorithm fixed. Please cite it too:

```bibtex
@article{pardo2020tonic,
  author  = {Pardo, Fabio},
  title   = {Tonic: A Deep Reinforcement Learning Library for Fast Prototyping and Benchmarking},
  journal = {arXiv preprint arXiv:2011.07537},
  year    = {2020}
}
```
