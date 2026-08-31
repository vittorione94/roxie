# Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call, and the env YAML *is* the builder call.**

*Back to the [README](../README.md).*

## The convention

`_target_` names the class and every sibling key is one of its keyword arguments — there is no `name:`/`args:` indirection and no registry. A knob that exists in Python but is missing from the YAML fails loudly at launch instead of silently taking its default. This is enforced by [`tests/test_agent_configs.py`](../tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](../tests/test_agent_construction.py).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know. Both are handled by [`agents.utils.build_agent`](../roxie/agents/utils.py), which `train.py` and the construction tests share.

One key in the agent block is *not* a constructor argument: `agent.device` says which hardware the agent runs on, and a JAX platform has to be chosen before the first `jax.*` call — long before an agent exists — so `build_agent` drops it after `train.py` has applied it. `env.device` is its counterpart in the env block, and together they are what an experiment states about its hardware; see [backends.md](backends.md#where-the-agent-runs-independent-of-where-the-physics-runs).

The env block follows the same convention: `env._target_` names a **builder** and every sibling key is one of its keyword arguments, instantiated by [`loader.build_env`](../roxie/environment/loader.py). So the core training loop never branches on an env-type string and never names a specific task — the builder can even live in another repo. Same for the viewer hook (`env.viewer`).

Five keys under `env:` are *not* builder arguments, because roxie consumes them itself — `device` is applied before anything is built, `parallel_envs` and `test_episodes` reach the builder as the injected `num_envs`/`test_episodes` instead, and `viewer`/`player` are `play.py`'s hooks. They are listed as `loader.TRAINER_ENV_KEYS` and stripped before instantiation, so a builder never declares a parameter it does not use. Everything else in the block is a builder keyword: a stale key is a `TypeError` at launch rather than a silently ignored setting, which is also what lets `build_envpool_env` collect per-task envpool options in a `**task_kwargs` tail.

## Layout

- [`roxie/configs/agent/`](../roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus baselines).
- [`roxie/configs/env/`](../roxie/configs/env/) — one file per env builder (`playground`, `envpool`): a complete `env:` block, pulled into a launchable's `defaults:` as `- /env: playground` and pointed at a task with `env.env_name=CheetahRun`. The leading `/` is required from a launchable in a subdirectory of `experiments/`, exactly as for `/agent:`.
- [`roxie/configs/noise/`](../roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one to them is a launch-time error.
- [`experiments/<env>/`](../experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/dmc/`](../experiments/dmc/) — the release benchmark, carrying its own config groups: `agent/` (matched hyperparameters), `backend/` (the device cells), `bench/` and `noise/` (shared blocks).

## A launchable

The benchmark launchables are the worked examples — [`experiments/dmc/bench_sac.yaml`](../experiments/dmc/bench_sac.yaml) is one file of `defaults:`, because everything else is a shared group. A standalone experiment pulls in the env group and overrides what it needs:

```yaml
# @package _global_

defaults:
  - /agent: sac
  - /env: playground
  - _self_

agent:
  device: gpu          # where the networks, optimizers and replay buffer run

env:
  device: gpu          # where the physics runs
  env_name: WalkerWalk
  impl: warp
  parallel_envs: 800

runtime:
  matmul_precision: null

trainer:
  steps: 1_000_000_000
  epoch_steps: 100_000
  save_steps: 500_000
  test_episodes: 5
  show_progress: true
  replace_checkpoint: false   # true keeps only the latest checkpoint, not the series
  save_buffer: false   # checkpoint the replay buffer too (exact resume, big files)

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

See [running trainings](training.md) for command-line overrides and the two argv-parsed specials (`device=`, `resume=`).
