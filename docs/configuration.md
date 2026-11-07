# Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call, and the env YAML *is* the builder call.**

*Back to the [README](../README.md).*

## The convention

`_target_` names the class and every sibling key is one of its keyword arguments — no `name:`/`args:` indirection, no registry. A knob that exists in Python but is missing from the YAML fails at launch instead of silently taking its default, enforced by [`tests/test_agent_configs.py`](../tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](../tests/test_agent_construction.py).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know. Both are handled by [`agents.utils.build_agent`](../roxie/agents/utils.py), shared by `train.py` and the construction tests.

Hardware placement is not an agent key: `runtime.device` (`cpu`/`gpu`/null) places the whole run, and a JAX platform has to be chosen before the first `jax.*` call — long before an agent exists. See [backends.md](backends.md#where-the-run-runs).

The env block follows the same convention: `env._target_` names a **builder**, which [`loader.build_env`](../roxie/environment/loader.py) calls with the block's other keys. So the core training loop never branches on an env-type string and never names a specific task — the builder can live in another repo. Same for the viewer hook (`env.viewer`).

A few keys under `env:` are *not* builder arguments, because roxie consumes them itself: `parallel_envs` and `test_episodes` reach the builder as the injected `num_envs`/`test_episodes`, `viewer`/`player` are `play.py`'s hooks, and `obs_size`/`action_size`/`obs_action_size`/`pixel_shape` are written *back* into the block by `publish_env_shapes` once the env exists, so an agent yaml can interpolate `${env.obs_size}`. They are listed as `loader.TRAINER_ENV_KEYS` and stripped before the call. Everything else in the block is a builder keyword: a stale key is a `TypeError` at launch.

## Layout

- [`roxie/configs/agent/`](../roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus the play-only baselines in [`roxie/agents/basic.py`](../roxie/agents/basic.py), which `Trainer.run` refuses to train).
- [`roxie/configs/env/`](../roxie/configs/env/) — one file per env builder (`playground`, and `multitask`, which composes several tasks): a complete `env:` block, pulled into a launchable's `defaults:` as `- /env: playground`. The leading `/` is required from a launchable in a subdirectory of `experiments/`, exactly as for `/agent:`.
- [`roxie/configs/curriculum/`](../roxie/configs/curriculum/) — task curricula for a multi-task env (`learning_progress`, `weighted`), pulled in as `curriculum=` ([training.md](training.md#several-tasks)).
- [`roxie/configs/noise/`](../roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one is a launch-time error.
- [`roxie/configs/vision/`](../roxie/configs/vision/) — the pixel encoder every network puts in front of its MLP (`cnn`). See [Vision](#vision).
- [`experiments/<env>/`](../experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/release/`](../experiments/release/) — the release benchmark, single-task and multi-task, carrying its own groups: `agent/`, `backend/`, `bench/` and `noise/`.
- [`experiments/vision/`](../experiments/vision/) — training from pixels.

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

## Vision

Any agent can learn from pixels. The encoder is a network knob like `features`: every actor and critic block in every agent yaml carries

```yaml
vision: ${oc.select:vision,null}
```

which is `null` — a plain MLP — unless a top-level `vision` group is selected, and then every network in the agent builds a [`PixelEncoder`](../roxie/models/vision.py) from it:

```yaml
defaults:
  - /agent: sac
  - /env: playground
  - /vision: cnn
  - _self_

env:
  env_name: CartpoleBalance
  impl: warp
  vision: true

vision:
  channels: [32, 32, 32, 32]   # one entry per conv layer
  kernel_sizes: [3, 3, 3, 3]
  strides: [2, 1, 1, 1]
  latent_dim: 50
```

The architecture comes from [`vision/cnn.yaml`](../roxie/configs/vision/cnn.yaml) and is overridden like any other block (`vision.latent_dim=100`); one network's own block can diverge from the rest (`agent.critic_config.critic1.vision.latent_dim=100`). [`experiments/vision/cartpole_sac.yaml`](../experiments/vision/cartpole_sac.yaml) is a complete launchable. List `- /vision: null` instead to leave it to the command line — `+vision=cnn` cannot reach the group from a launchable in a subdirectory.

**Observations stay flat.** The env flattens its image row-major from `(H, W, C)` and puts it at the *front* of the observation vector, any state after it, and advertises the shape as `pixel_shape`, which `publish_env_shapes` writes to `env.pixel_shape` for the encoder block to read. The replay buffer, the running observation statistics, the rollout and the losses never know the difference; only a network with a `vision` block reshapes that leading slice back into an image, encodes it (conv stack → linear → LayerNorm → tanh), and hands its MLP the latent followed by the rest. An env without pixels publishes `pixel_shape: null`, which the encoder refuses at construction.

**Each network owns its encoder.** The actor, the critic and each twin head train their own from their own loss, and target networks copy them like any other layer. That is a conv stack per network rather than one shared trunk, in exchange for working unchanged under all seven agents' targets, optimizers and checkpoints.

**Where pixels come from.** On mujoco_playground, `env.vision: true` renders through mujoco_warp — so `impl: warp`, a CUDA GPU, and a task that implements a vision mode (today the dense Cartpole tasks). A `FuncEnv` of your own opts in by setting `pixel_shape` and returning the flat layout above. Warp needs the `cuda` dependency group (`uv run --group cuda ...`); without it JAX falls back to the CPU and the renderer cannot find its context.

[`examples/vision/humanoid_rgbd.py`](../examples/vision/humanoid_rgbd.py) renders a task playground has no vision mode for: the humanoid's head-mounted `egocentric` camera, colour and depth stacked into one 64×64×4 image ahead of its proprioception, from a builder outside `roxie/`. [`experiments/vision/humanoid_rgbd_sac.yaml`](../experiments/vision/humanoid_rgbd_sac.yaml) trains it. A rendering env has to start each world from its own `rng`: a warp render context renders all its worlds at once, so a reset that ignores its key leaves the vmapped data unbatched and every world receives every world's image.

**Replay memory.** Pixels are stored as float32: a 64×64×3 frame is 48 KB, so size `agent.memory_config.max_length` for it.

