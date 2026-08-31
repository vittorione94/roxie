# Agents

What the seven agents share, how they choose a replay buffer, and what makes a comparison between them fair.

*Back to the [README](../README.md).*

## The shared base

All share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the `step`/`add`/`update` interface the trainer consumes. Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are available for sanity-checking an environment.

`Agent` also assembles what every actor-critic agent builds identically — target copies, the two optimizers, observation statistics, the `TrainState` itself (`_init_train_state`) — and exports the hyperparameter block a checkpoint carries (`_export_hyperparams`, plus `_replay_hyperparams` off-policy), so a subclass writes only its own knobs. The pieces of an update that are the same everywhere live in `roxie.agents.utils`: `fused_grad_steps` (the `lax.scan` that fuses a burst of gradient steps into one compiled program), `soft_update` (Polyak target averaging), `transition_prototype` and `build_replay` (the buffer's schema and kind). An agent module is then just its losses, its step function, and the knobs it adds.

Each agent's YAML in [`roxie/configs/agent/`](../roxie/configs/agent/) *is* its constructor call — see [configuration](configuration.md).

## Buffers

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching automatically to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Both structures have standalone walkthroughs under [`examples/flashbax/`](../examples/flashbax/).

## Comparing them

Two caveats: **PPO is not replay-ratio comparable** (on-policy — judge it on score-vs-env-steps and score-vs-wall-clock, not gradient steps), and **MPO is ~20× more expensive per gradient step** (20 action samples per state at batch 512). MPO runs 1-step returns because `mpo.py` takes no `n_step`; that is an implementation gap, not a chosen handicap.

The [release benchmark](../experiments/README.md) is the setting where the comparison is controlled: everything except the algorithm and the backend cell is held identical.
