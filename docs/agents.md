# Agents

What the seven agents share, how they choose a replay buffer, and what makes a comparison between them fair.

*Back to the [README](../README.md).*

## The shared base

All share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the `step`/`add`/`update` interface the trainer consumes. Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are available for sanity-checking an environment.

Each agent's YAML in [`roxie/configs/agent/`](../roxie/configs/agent/) *is* its constructor call — see [configuration](configuration.md).

## Buffers

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching automatically to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Both structures have standalone walkthroughs under [`examples/flashbax/`](../examples/flashbax/).

## Comparing them

Two caveats: **PPO is not replay-ratio comparable** (on-policy — judge it on score-vs-env-steps and score-vs-wall-clock, not gradient steps), and **MPO is ~20× more expensive per gradient step** (20 action samples per state at batch 512). MPO runs 1-step returns because `mpo.py` takes no `n_step`; that is an implementation gap, not a chosen handicap.

The [release benchmark](../experiments/README.md) is the setting where the comparison is controlled: everything except the algorithm and the backend cell is held identical.
