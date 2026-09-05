# Agents

What the seven agents share, how an update is dispatched to the device, how they choose a replay buffer, and what makes a comparison between them fair.

*Back to the [README](../README.md).*

## The shared base

All share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the `step`/`add`/`update` interface the trainer consumes. Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are available for sanity-checking an environment.

`Agent` also assembles what every actor-critic agent builds identically — target copies, the two optimizers, observation statistics, the `TrainState` itself (`_init_train_state`) — and exports the hyperparameter block a checkpoint carries (`_export_hyperparams`, plus `_replay_hyperparams` off-policy), so a subclass writes only its own knobs. The pieces of an update that are the same everywhere live in `roxie.agents.utils`: `fused_grad_steps` (the `lax.scan` that fuses a burst of gradient steps into one compiled program), `soft_update` (Polyak target averaging), `transition_prototype` and `build_replay` (the buffer's schema and kind). An agent module is then just its losses, its step function, and the knobs it adds.

Each agent's YAML in [`roxie/configs/agent/`](../roxie/configs/agent/) *is* its constructor call — see [configuration](configuration.md).

## The fused burst: `burst_nodes` and `graph_jit`

The trainer's unit of work is not a step, it is a **window**: `steps_between_updates` env steps, then a burst of `learning_steps` gradient steps. Both halves are single compiled programs — `fused_grad_steps` scans the gradient steps, and on the JAX-physics path [`JaxRollout.collect`](../roxie/utils/rollout.py) scans the whole acting chunk too, so the window costs **two host dispatches instead of one per step**. (The acting half fuses only where it can be traced: PPO and the EnvPool pool step through a Python loop. The gradient burst is fused for every learning agent.)

That makes the *boundary* the cost. `nnx.jit` re-walks the module graph in Python on **every** call, and both dispatches cross one: measured on this repo's train states, 2.2 ms for DDPG's (98 leaves) and 3.5 ms for TD3's (142), twice per ~16 ms window — 28% of DDPG's loop and 36% of TD3's, spent on the host with the GPU at ~50% utilization. Handing a plain `jax.jit` the graphdef as a *static* argument and the state as a pytree costs one pytree traversal instead: 0.4 / 0.6 ms. On AcrobotSwingup / `warp_gpu` / 256 envs that took DDPG from 130k to 245k sps and TD3 from 97k to 199k, utilization 53% → 99%/88%, with **bit-for-bit identical curves** on a matched seed — it moves no computation, no ordering and no RNG stream, only where the graph is walked. (`nnx.cached_partial` is the flax-native answer and does not apply here: it requires every attribute of a cached node to be an `nnx.Variable`, and `TrainState` holds `buffer_state` and `obs_stats` as raw arrays.)

Three pieces in [`roxie/agents/utils.py`](../roxie/agents/utils.py) implement that:

**`SplitNodes`** holds a tuple of nnx nodes in split form — `(graphdef, pytree)` — across calls, so a window pays at most one `nnx.split`.

**`burst_nodes`** is an agent's `SplitNodes` handle: *the nodes a burst mutates*. Anything the fused steps write to has to ride in the same split, or it is silently reset to its entry value on every step of the scan — which is why the side modules are in there and not held as ordinary attributes:

| Agent | `_num_burst_nodes` | Nodes |
|---|---|---|
| DDPG, TD3, D4PG, TD4, PPO | 1 | `state` (the `TrainState`: actor, critic, targets, both optimizers, replay buffer, obs stats) |
| SAC | 3 | `state`, `log_alpha_module`, `alpha_optimizer` |
| MPO | 3 | `state`, `dual_params`, `dual_optimizer` |

Each entry is declared as a `BurstNode(index)` descriptor on the class, an attribute view that lets `self.state` or `self.log_alpha_module` read and write like the plain attributes they replaced while the hot loop exchanges only pytrees. (A `None` node raises `AttributeError`, which is what keeps `hasattr(agent, "state")` False for the stateless baselines in `roxie.agents.basic` — `checkpoint_payload` branches on it.)

The deterministic agents' exploration noise sits in a **second** handle, `_noise_nodes`, because only the *acting* burst touches it: the acting burst takes `_burst_nodes` **plus** `_noise_nodes`, the gradient burst takes `_burst_nodes` alone. Keeping it out of the gradient split is what spares the four deterministic agents (DDPG, TD3, D4PG, TD4) a `noise_module` parameter their `_grad_steps` would only thread through unchanged.

**`graph_jit`** is the decorator that hoists the walk out. The wrapped function is written exactly as it was under `nnx.jit` — takes live nodes, returns them first — but the caller passes the `SplitNodes` in their place and the graphdef rides as a static argument; the merge happens at trace time only. `num_nodes` says how many leading arguments collapse into that one split (3 for SAC and MPO), `static_argnames` names the loop constants, and `donate` (default on) donates the pytree, because the replay buffer threads unchanged through the scan and without donation XLA allocates a second copy of it per burst. PPO's `_prepare_rollout` is the one `donate=False` site — read-only on the buffer, called once per rollout.

Two contracts follow from this, and both are easy to break when adding an agent:

- **`SplitNodes` is updated in place**, so the node vanishes from both sides of a call:
  ```python
  self.state, actor_loss, critic_loss = _grad_steps(self.state, key, ...)   # under nnx.jit
  actor_loss, critic_loss = _grad_steps(self._burst_nodes, key, ...)        # under graph_jit
  ```
- **A burst donates its input**, so any reference held across one is a deleted buffer. The caller must adopt what came back — `graph_jit` does it for the agent, `JaxRollout.collect` does it explicitly for both handles.

Host code that materializes the live nodes (checkpoint save detaching `buffer_state`, `restore` merging into the live modules, eval reading the actor) still wins: **the live nodes stay the authority**, and reading `.live` marks the split stale so it is re-derived on the next burst. That is one `nnx.split` per epoch boundary instead of two per window, which is the whole point. None of it is thread-safe, and it does not need to be — `AsyncLearner` makes its learner thread the sole owner of `agent.state`, and `pause()` quiesces it before eval or checkpointing looks.

Where the remaining host time goes, and which of the two candidates for removing it is worth trying, is documented at the top of [`roxie/utils/rollout.py`](../roxie/utils/rollout.py).

## Buffers

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching automatically to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Both structures have standalone walkthroughs under [`examples/flashbax/`](../examples/flashbax/).

## Comparing them

Two caveats: **PPO is not replay-ratio comparable** (on-policy — judge it on score-vs-env-steps and score-vs-wall-clock, not gradient steps), and **MPO is ~20× more expensive per gradient step** (20 action samples per state at batch 512). MPO runs 1-step returns because `mpo.py` takes no `n_step`; that is an implementation gap, not a chosen handicap.

The [release benchmark](../experiments/README.md) is the setting where the comparison is controlled: everything except the algorithm and the backend cell is held identical.
