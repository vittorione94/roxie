# CMU_006_13 overfit sweep

One launchable per agent, all pointed at the same single-clip overfit task, to
answer: **which algorithm gets furthest on CMU_006_13 (the hard 40.8s clip),
per env step and per wall-clock hour?**

```bash
uv run python roxie/train.py --config-name mocap/sweep_td3    # reference arm
uv run python roxie/train.py --config-name mocap/sweep_ddpg
uv run python roxie/train.py --config-name mocap/sweep_td4
uv run python roxie/train.py --config-name mocap/sweep_d4pg
uv run python roxie/train.py --config-name mocap/sweep_sac
uv run python roxie/train.py --config-name mocap/sweep_mpo
uv run python roxie/train.py --config-name mocap/sweep_ppo
```

Each writes to `outputs/sweep_cmu_006_13/<agent>/<timestamp>/`. Run them one at
a time — they all want ~1000 Warp envs plus a GPU learner.

The sweep is Warp-only by design (same physics for every arm). To run an arm on
CPU physics instead, both the backend and the env count have to move together —
`backend=envpool env.parallel_envs=48 +trainer.async_learner=true` — which
takes it out of the sweep comparison, so do that as a separate experiment.

## Held identical across every arm

Verified by composing all seven configs and diffing the resolved blocks.

| Held fixed | Where |
| --- | --- |
| env: 1 clip `CMU_006_13`, 1000 envs, position actuation, self-collisions, no clip swap | `sweep/overfit_cmu_006_13.yaml` |
| env physics/obs: ctrl_dt 0.025, look_ahead 5, terminations | `env_config/cmu.yaml` |
| reward: all weights + kernel bandwidths | `reward/cmu_tracking.yaml` |
| backend: Warp, `WARP_STAGED_EX` | `backend/warp.yaml` |
| trainer: 1e9 steps, 1M-step epochs, 100 test episodes | `sweep/overfit_cmu_006_13.yaml` |
| actor + critic MLPs: `[1024, 512, 256]`, layer norm on | every `agent/*_sweep.yaml` |
| batch 512, 128 grad steps / 8k env steps (replay ratio ~8), buffer 500k, warmup 30k | every off-policy `agent/*_sweep.yaml` |
| gamma 0.99, tau 0.005, lr 3e-4, grad-norm clip 1 | every `agent/*_sweep.yaml` |
| exploration noise (deterministic arms only): gaussian 0.4 -> 0.1 over 50M | `noise/sweep_gaussian.yaml` |

## What necessarily differs (algorithmic, not tuning)

| Arm | Critic | Returns | Exploration |
| --- | --- | --- | --- |
| ddpg | 1x Q | n-step 5 | gaussian noise |
| td3 | 2x Q, policy delay 2 | n-step 5 | gaussian noise |
| td4 | 2x categorical (101 atoms, [-10, 100]) | n-step 5 | gaussian noise |
| d4pg | 1x categorical (101 atoms, [-10, 100]) | n-step 5 | gaussian noise, no target smoothing |
| sac | 2x Q | **1-step** (no `n_step` in sac.py) | entropy, auto alpha |
| mpo | 1x Q | **1-step** (no `n_step` in mpo.py) | policy sampling + KL duals |
| ppo | V critic | GAE(0.95), on-policy | policy entropy |

Caveats worth remembering when ranking:

- **PPO is not replay-ratio comparable.** On-policy: 32 x 1000 = 32k transitions
  per update, then the queue drains. Judge it on score-vs-env-steps and
  score-vs-wall-clock, not on gradient steps.
- **MPO is ~20x more expensive per gradient step** (20 action samples per state
  at batch 512). Same replay ratio, far fewer steps/s. Lower its
  `learning_steps` if it can't keep up — and note the change.
- **SAC/MPO run 1-step returns** because their agents don't take `n_step`; that
  is an implementation gap, not a chosen handicap.
- Twin vs single critic is part of the algorithm, so per-network width is what
  is equalized, not total parameter count.

## Reading the results

Prior campaign finding: `test/score / test/length ~ 0.60` regardless of config —
score is essentially survival time, so track `test/length` alongside score.
Epoch-to-epoch eval std was +-2.7 at 50 test episodes (this sweep uses 100), so
**rank on the back-half mean +- std, never on a peak epoch**. Health checks per
arm: `loss/critic` roughly 1-3, `|loss/actor|` (~mean Q) under ~90.
