# Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call.**

*Back to the [README](../README.md).*

## The convention

`_target_` names the class and every sibling key is one of its keyword arguments — there is no `name:`/`args:` indirection and no registry. A knob that exists in Python but is missing from the YAML fails loudly at launch instead of silently taking its default. This is enforced by [`tests/test_agent_configs.py`](../tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](../tests/test_agent_construction.py).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know.

The env factory itself is a dotted path (`env.builder`, resolved with `hydra.utils.get_method`), so the core training loop never branches on an env-type string and never names a specific task — the builder can even live in another repo. Same for the viewer hook (`env.viewer`).

## Layout

- [`roxie/configs/agent/`](../roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus baselines).
- [`roxie/configs/noise/`](../roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `composite`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one to them is a launch-time error.
- [`experiments/<env>/`](../experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/dmc/`](../experiments/dmc/) — the release benchmark, carrying its own config groups: `agent/` (matched hyperparameters), `backend/` (the device cells), `bench/` and `noise/` (shared blocks).

## A launchable

The benchmark launchables are the worked examples — [`experiments/dmc/bench_sac.yaml`](../experiments/dmc/bench_sac.yaml) is one file of `defaults:`, because everything else is a shared group. A standalone experiment spells its own settings out instead:

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

See [running trainings](training.md) for command-line overrides and the two argv-parsed specials (`device=`, `resume=`).
