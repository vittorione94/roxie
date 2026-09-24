# Agents

What the seven agents share, how an update is dispatched, and the contract a new agent has to honour.

*Back to the [README](../README.md).*

## The shared base

All agents share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the interface the loop consumes: `select_action` and `buffer_transitions` inside the fused chunk, `eval_action` inside the eval loop, and `learn` at the window boundary. `Agent` also assembles what every actor-critic agent builds identically — target copies, the two optimizers, observation statistics, the `TrainState` (`_init_train_state`) — and exports the hyperparameter block a checkpoint carries (`_export_hyperparams`, plus `_replay_hyperparams` off-policy).

The shared parts of an update live in `roxie.agents.utils`: `fused_grad_steps` (the `lax.scan` fusing a pass's gradient steps into one program), `soft_update` (Polyak averaging), `reduce_diagnostics`, `transition_prototype` and `build_replay`. An agent module is then its losses, its step function, and the knobs it adds.

Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`, in [`roxie/agents/basic.py`](../roxie/agents/basic.py)) are play-only: their policy *is* their own Python state, so there is nothing for the compiled acting to be handed, and their `learn` raises.

Each agent's YAML in [`roxie/configs/agent/`](../roxie/configs/agent/) *is* its constructor call — see [configuration](configuration.md).

### Hyperparameters

Every knob a learning agent has lives in one frozen dataclass on `agent.hp`, declared in [`roxie/agents/hyperparams.py`](../roxie/agents/hyperparams.py) and named by the class's `hyperparams_cls`. Framework code reads `agent.hp.gamma`, never `agent.gamma`; a baseline carries `hp = None`.

```
AgentHyperparams           seed, gamma, both learning rates, max_grad_norm,
                           learning_steps, steps_between_updates, obs normalization
├─ OffPolicyHyperparams    tau, memory_warmup
│  ├─ DDPGHyperparams      n_step, target smoothing, pre_activation_coef   (DDPG, D4PG)
│  │  └─ TD3Hyperparams    policy_delay, twin_q_weight                     (TD3, TD4)
│  ├─ SACHyperparams       alpha_learning_rate, the temperature dual, policy_delay, twin_q_weight
│  └─ MPOHyperparams       n_step, dual_learning_rate, the KL bounds, the dual inits
└─ PPOHyperparams          gae_lambda, clip_eps, entropy_coef, target_kl, num_minibatches
```

Three things follow from it being a dataclass:

* **The checkpoint round-trips by construction.** `_export_hyperparams` is `dataclasses.asdict(self.hp)` and `Agent.load` rebuilds `hyperparams_cls` by field name. The on-disk layout is flat, one key per knob.
* **A learning pass takes the whole set as ONE static argument.** `frozen=True` makes it hashable, so each `_<agent>_grad_steps` declares `static_argnames=("hp", ...)` instead of listing a dozen scalars.
* **The type answers capability questions.** `isinstance(agent.hp, OffPolicyHyperparams)` is how the trainer asks whether there is a replay to warm up.

Derived quantities stay attributes instead: SAC's resolved `target_entropy`, PPO's `minibatch_size`, the buffer geometry. The exception is PPO's `steps_between_updates`, which the trainer reads for every agent the same way — PPO computes it from its queue width and writes it back onto `hp`, declaring it in `derived_hyperparams` so the config test knows the yaml must not set it.

Two knobs whose spelling is not self-evident:

* `twin_q_weight` blends the twin bootstrap continuously between `min` (1.0), the mean of the two heads (0.5) and `max` (0.0).
* `adv_norm_decay` divides PPO's advantage by a debiased running standard deviation carried across rollouts; `0.0` falls back to the current rollout's own spread. The accumulator squares the *centered* advantage, so the scale is a standard deviation and not an RMS.

## The fused window

The trainer's unit of work is a **window**: `steps_between_updates` env steps, then a pass of `learning_steps` gradient steps. Both halves are single compiled programs — [`Rollout.collect`](../roxie/utils/rollout.py) scans the whole acting chunk, `fused_grad_steps` scans the gradient steps — so a window costs **two host dispatches rather than a handful per step**, on every agent and both backends. A C++ pool cannot be traced but can be *called* from inside a trace, so `EnvPoolRollout` steps its pool through an ordered `io_callback` and compiles the same chunk ([backends.md](backends.md#why-there-are-two-rollouts)).

That makes the *boundary* the cost, which is why `nnx.jit` is nowhere near the hot loop: it re-walks the module graph in Python on every call. So `TrainState` is declared a **pytree** rather than an opaque nnx graph node, and a learning pass hands it straight to `jax.jit` and `lax.scan` with no `nnx.split`/`merge`. (`buffer_state` and `obs_stats` are annotated `nnx.data()` because they are raw arrays rather than `nnx.Variable`s, and the two target slots because a slot born `None` — SAC's and PPO's — would otherwise be classified static and refuse a later assignment.)

Each agent's `_<agent>_grad_steps` is a module-level `jax.jit` function, with the same argument split everywhere:

- **Static** — `hp`, `replay_sample_fn`, `n_steps`, plus SAC's resolved `target_entropy`.
- **Donated** — the train state. The replay buffer threads unchanged through the scan, and without donation XLA allocates a full second copy of it per pass.
- **Traced** — the key and the action bounds. Observation-normalization mean/std are loop constants *within* a pass but move between them, so `_<agent>_grad_steps` derives them once at the top and hoists them into the scanned step.

Anything the fused steps write to must travel in the donated set, or it is silently reset to its entry value on every step of the scan:

| Agent | Donated |
|---|---|
| DDPG, TD3, D4PG, TD4, PPO | `state` (the `TrainState`: actor, critic, targets, both optimizers, replay buffer, obs stats) |
| SAC | `state`, `log_alpha_module`, `alpha_optimizer` |
| MPO | `state`, `dual_params`, `dual_optimizer` |

One contract follows: **a learning pass donates its input**, so any reference held across one is a deleted buffer. The caller must assign back what came back — `Agent.learn` does it off the `LearningOutput`, and `Trainer._run` does it for `collect`, which donates the train state and the noise module both. PPO's `_prepare_rollout` is the one jitted entry point that donates nothing: it is read-only on the queue and runs once per rollout.

Exploration noise stays *out* of the learning pass and travels as its own argument to `collect`: only the acting half touches it, and its decay counter has to advance on the carry rather than on `self`. That also spares the four deterministic agents a `noise_module` parameter their grad steps would thread through unchanged.

`hasattr(agent, "state")` is False for the stateless baselines, which never call `_init_train_state`; `checkpoint_payload` branches on it to return `None`.

## The agent contract

Each of these is a constraint the fused window imposes on the base class.

### Acting: the agent keeps no acting state

`select_action` has one signature for all seven agents, in two halves. Called bare — `select_action(obs, key)` — it acts off `self.state`, which is what `play.py` and the per-step rollouts use. The keyword-only half (`actor`, `obs_stats`, `noise_module`, `critic`) overrides each piece of that state instead.

That second half is what lets a fused acting chunk run against a `lax.scan` carry: inside a trace the actor, the statistics and the noise counter are *carried values*, and reading them off `self` would bake the first chunk's weights into every chunk. Every keyword is part of the shared signature whether or not a given agent has the thing — DDPG ignores `critic`, PPO ignores `noise_module` — so one caller drives all of them. The play-only baselines take the positional half only.

The same rule explains `extras`, the third return value: behaviour an agent computed while acting and cannot recover later — empty for everything but PPO, which puts its log-prob, value and pre-tanh action there. It rides out on the transition handed to `buffer_transitions` rather than being stashed as `self.last_*`, because acting and buffering happen inside one `lax.scan`, where such an attribute would be a traced value escaping its trace.

### Evaluation is a separate entry point

`eval_action` is deliberately not `select_action(evaluate=True)`: that path carries the noise module, whose counter cannot be mutated across trace levels inside the eval loop's `while_loop`, and an eval must not anneal exploration it is not using. The two agree by construction — `deterministic_action` takes the mode of a stochastic actor and passes a deterministic one through.

Its actor and statistics are arguments rather than `self.state` for the same reason acting's are: the eval is one compiled program in which the agent is a constant.

### Observation normalization

Acting is normalized against the statistics as they stood when the chunk opened, for every agent. The fused chunk stacks the observations it steps through and folds the whole `(T * B, D)` block into `obs_stats` once, after its `lax.scan` (`Agent.absorb_obs_stats`), rather than T times on the carry. `ObsStats` is running sums, so the statistics land exactly where per-step folds would have left them; what the deferral moves is only *when acting sees them*. In exchange, `obs_mean_std`'s sqrt is hoisted out of the loop and six-odd accumulators come off the scan carry.

PPO's `freeze_obs_norm_per_chunk` is a stronger pin: its snapshot is the same `ObsStats` object acting and the *update* both run under, and it survives the chunk until `learn` clears it, because its ratio is only meaningful against the observations the behaviour policy saw. `_prepare_rollout` re-normalizes the stored observations to recompute that ratio, so drift between acting and updating would make the clip and the KL early stop fire on normalization noise.

`freeze_acting_norm` is the host-side half: the rollout calls it at every chunk boundary and gets back the `ObsStats` acting must use, or `None` from agents with no update-time pin — which every caller reads as "use the ones off the train state". `EnvPoolRollout` needs it returned, its acting being one dispatch per step; `JaxRollout` hoists the same value off its traced pytree.

`obs_norm_clip` bounds the z-score; `None` leaves it unbounded. It is a *static* argument to `normalize_obs`, and travels through a checkpoint as a number, non-positive meaning `None`.

### `learn` is unconditional; `due_for_update` consumes its boundary

`learn` runs one learning pass in place and returns `(gradient_steps, actor_loss, critic_loss)`. Unconditional is the contract: the caller asks `due_for_update` first, so `learn` never has to answer "is it time" and a test can drive a pass directly. The step count comes back rather than being assumed, because PPO decides it on device — its trust region can stop early. A play-only baseline raises here rather than returning a zero that would read as a converged loss.

`learn` is written once, on the base: it sizes the pass off `hp.learning_steps`, hands it to the agent's `_compile_and_run`, binds back what came back, and records diagnostics. That work returns a `LearningOutput` — a `struct.dataclass` of `state`, the two averaged losses, per-step `diagnostics`, and an `extra_state` for donated modules living outside `TrainState` (SAC's temperature, MPO's duals) plus PPO's last-pass scalars, which `_apply_extra_state` unpacks. The step count is deliberately *not* in that pytree: it is a host `int` for every agent but PPO, and returning it through a jitted boundary would cast it to a device scalar. It rides beside the pytree as `(steps_taken, LearningOutput)`, and PPO — whose count is genuinely on device — syncs for it with `.item()`.

`due_for_update(steps)` is true at most once per `steps_between_updates` env steps past warmup, and three of its properties are easy to break:

* **It is not idempotent.** A true answer *consumes* the boundary it fired on. Ask it exactly where the pass would run.
* **`memory_warmup` is the only gate, and it is not optional.** flashbax allocates with `jnp.empty_like`, so a batch drawn below the buffer's `min_length` is uninitialized memory rather than an error.
* **It is not `(steps - memory_warmup) % between == 0`.** The trainer advances `steps` in strides of `parallel_envs`, so that fires only when the offset is itself a multiple of the stride; otherwise the residue cycles without ever reaching 0. The boundary is floored instead.

No backlog is queued — the boundary jumps to wherever `steps` now is, so a restored checkpoint resumes on schedule. PPO overrides the whole gate: its work is due when a rollout has finished, which its queue answers directly.

### Diagnostics are banked on device, drained per epoch

An agent reports the health of its update through `record_diagnostics` at the end of a pass and `pop_diagnostics`, which the trainer drains once per epoch under `train/`. Only the keys differ between agents — the plumbing is a `DiagnosticsTracker` ([`roxie/utils/diagnostics.py`](../roxie/utils/diagnostics.py)) that knows nothing about agents.

Recorded values are **device scalars**, kept unread until the drain, so a pass never syncs the host on the critical path. The epoch reduction applies the rule each key deserves — `max` for `DIAGNOSTIC_MAX_KEYS`, `sum` for counts, otherwise a gradient-step-weighted mean — the same rule `reduce_diagnostics` applies across the steps of one pass. Two rates ride along as levels rather than per-epoch spikes:

* `updates_per_env_step`, the realized replay ratio. Off the configured `learning_steps / steps_between_updates` means the update gate is not firing as intended.
* `buffer_frac`. Below 1.0 the sampler draws from a narrower window than configured. The agent adds this one itself, being the only party that knows how big its buffer is.

The env-step count is passed *in* by the trainer, so both rates are against the loop's own count.

## Buffers

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Both structures have standalone walkthroughs under [`examples/flashbax/`](../examples/flashbax/).

### Writing into one

`buffer_transitions` writes one env-step batch **in place**, handing the transition to the store exactly as the caller assembled it: the time axis, the pruning to the fields this buffer allocated, and the NaN scrub all belong to `ReplayManager` ([`roxie/utils/memory.py`](../roxie/utils/memory.py)). Its optional `state` argument is the traced train state a fused chunk carries as a `lax.scan` carry; omit it and the agent's own state is used with the jitted, donating add. The two always go together, since a caller that has its own state is inside a trace, where the jitted wrapper would only nest a `pjit`.

`fill_buffer` is the bulk sibling for the warmup fill: the adds scan inside one dispatch and the statistics take a single pass over the block. The fused chunk splits the same idea across the two — it passes `update_stats=False` and folds the block in afterwards through `absorb_obs_stats`, which keeps acting's weighting (the landing observation counted twice) while paying for it once.

`terminal` means **true termination only**. Marking a time-limit truncation terminal zeroes its bootstrap and collapses Q at the cutoff — for every env at once, since they hit the limit in lockstep — so `truncation` is stored separately, which also lets n-step windows stop there ([environments.md](environments.md)).

## What a checkpoint carries

Three blocks, plus the modules that live outside the train state:

* **`_export_hyperparams`** — flat, one key per knob; `Agent.load` rebuilds `hyperparams_cls` from it by field name. Abstract, because a non-learning baseline has no `hp`.
* **`_replay_hyperparams`** — the *derived* block every replay-driven agent carries. None of these is a constructor keyword: the buffer's geometry comes from `memory_config`, and the env sizes are read back off the buffer, because the shape it was **allocated** with is what a checkpoint is rebuilt against.
* **`_checkpoint_modules`** — agent-owned `nnx` modules kept next to `self.state`: DDPG's exploration noise, SAC's temperature, MPO's duals, plus their optimizers. `save`/`restore` serialize `self.state` wholesale, so dropping these would restart exploration and the duals at their init values on resume. Keys are attribute names, restored with `setattr`.

The on-disk format lives in [`roxie/utils/checkpoint.py`](../roxie/utils/checkpoint.py); `checkpoint_payload` / `load` / `restore` on the agent are its agent-facing spelling.
