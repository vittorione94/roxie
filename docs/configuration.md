# Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call, and the env YAML *is* the builder call.**

*Back to the [README](../README.md).*

## The convention

`_target_` names the class and every sibling key is one of its keyword arguments — no `name:`/`args:` indirection, no registry. A knob that exists in Python but is missing from the YAML fails at launch instead of silently taking its default, enforced by [`tests/test_agent_configs.py`](../tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](../tests/test_agent_construction.py).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know. Both are handled by [`agents.utils.build_agent`](../roxie/agents/utils.py), shared by `train.py` and the construction tests.

Hardware placement is not an agent key: `runtime.device` (`cpu`/`gpu`/null) places the whole run, and a JAX platform has to be chosen before the first `jax.*` call — long before an agent exists. See [backends.md](backends.md#where-the-run-runs).

The env block follows the same convention: `env._target_` names a **builder**, instantiated by [`loader.build_env`](../roxie/environment/loader.py). So the core training loop never branches on an env-type string and never names a specific task — the builder can live in another repo. Same for the viewer hook (`env.viewer`).

A few keys under `env:` are *not* builder arguments, because roxie consumes them itself: `parallel_envs` and `test_episodes` reach the builder as the injected `num_envs`/`test_episodes`, `viewer`/`player` are `play.py`'s hooks, and `obs_size`/`action_size`/`obs_action_size` are written *back* into the block by `publish_env_shapes` once the env exists, so an agent yaml can interpolate `${env.obs_size}`. They are listed as `loader.TRAINER_ENV_KEYS` and stripped before instantiation. Everything else in the block is a builder keyword: a stale key is a `TypeError` at launch, which is also what lets `build_envpool_env` collect per-task options in a `**task_kwargs` tail.

## Layout

- [`roxie/configs/agent/`](../roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus the play-only baselines in [`roxie/agents/basic.py`](../roxie/agents/basic.py), which `Trainer.run` refuses to train).
- [`roxie/configs/env/`](../roxie/configs/env/) — one file per env builder (`playground`, `envpool`): a complete `env:` block, pulled into a launchable's `defaults:` as `- /env: playground`. The leading `/` is required from a launchable in a subdirectory of `experiments/`, exactly as for `/agent:`.
- [`roxie/configs/noise/`](../roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one is a launch-time error.
- [`experiments/<env>/`](../experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/dmc/`](../experiments/dmc/) — the release benchmark, carrying its own groups: `agent/`, `backend/`, `bench/` and `noise/`.

## A launchable

A standalone experiment pulls in the env group and overrides what it needs:

```yaml
# @package _global_

defaults:
  - /agent: sac
  - /env: playground
  - _self_

env:
  env_name: WalkerWalk
  impl: warp
  parallel_envs: 800

runtime:
  device: gpu          # networks, optimizers, replay buffer AND physics
  matmul_precision: tensorfloat32

trainer:
  steps: 1_000_000_000
  epoch_steps: 100_000
  save_steps: 500_000
  test_episodes: 5
  show_progress: true
  replace_checkpoint: false   # true keeps only the latest checkpoint, not the series
  save_buffer: false          # checkpoint the replay buffer too (exact resume, big files)

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

See [running trainings](training.md) for command-line overrides and the two argv-parsed specials (`device=`, `resume=`).
