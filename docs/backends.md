# Backends: CPU vs GPU

Roxie runs on the GPU or on the CPU — the *physics* and the *agent* (networks, optimizers, replay buffer) together, never split.

*Back to the [README](../README.md).*

## The three physics backends

| | `impl: jax` (MJX) | `impl: warp` (mujoco_warp) | `impl: batch` (`mujoco.batch`) |
|---|---|---|---|
| Device | GPU (or CPU) | GPU only | CPU only |
| Vectorization | `jax.vmap` over a traced step | `jax.vmap` → Warp kernels | C++ thread pool over one row per world |
| Batch dim lives in | the XLA program | the XLA program | native code; the XLA program sees `(N, ...)` arrays |
| Contact allocation | **static**, sized to *all* potential geom pairs | **budgeted** — `naconmax` / `njmax` / `naccdmax` | **dynamic**, MuJoCo allocates as needed |
| Reached through | a traced step | a traced step | an ordered `io_callback` |
| Bounded by | VRAM | VRAM | system RAM + core count |
| Built by | `loader.build_playground_env` | `loader.build_playground_env` (`impl: warp`) | `loader.build_playground_env` (`impl: batch`) |

Selection is a Hydra config group: [`experiments/release/backend/`](../experiments/release/backend/) holds `warp_gpu.yaml`, `batch_cpu.yaml`, `mjx_gpu.yaml` and `mjx_cpu.yaml`, and a launchable picks one in its `defaults:`. Everything backend-mechanical (`impl`, solver budgets, graph mode, thread count, `runtime.device`) lives in that group; per-experiment tuning stays in the launchable.

`impl` picks which physics steps the *same* task: the XML, the reset, the observation and the reward are upstream `mujoco_playground` code on all three. `loader.uses_batch` asks the config wherever roxie needs to know (pinning the run to the CPU in `train.py`).

`BatchVectorEnv` copies back only the `mjData` fields in `vector.py`'s `_BATCH_CARRIED` and `_BATCH_DERIVED`; every other field the task sees keeps its reset value. So at build time it traces the task's step and refuses one that reads any other field, naming it; adding a field `mj_step` always computes to `_BATCH_DERIVED` is how such a task is let in. Resets still run on MJX; only the step moves to native MuJoCo.

`train.py` prints a banner at startup reporting the backend that *actually* loaded (read off the constructed env, not echoed from config) and flags a mismatch against what was requested.

## One rollout

Every env presents the same interface ([environments.md](environments.md)), so `Trainer._run` is a single loop and `Rollout` a single class: it holds the chunk, the warmup, the eval loop and the epoch boundary for all three kinds of physics.

**The loop must never touch the host.** Every quantity a Python `if` would branch on — done flags, buffer occupancy, episode returns — lives in a device array, and reading one forces a blocking device→host sync. So:

- **`collect` is the unit of work, not `step`.** A whole chunk — acting, physics, buffering, running statistics, episode bookkeeping — fuses into one `jax.jit` program. The chunk is never widened past one `steps_between_updates` window, which keeps the actor fixed for its duration.
- **Auto-reset is branchless.** The loop pre-builds a *reset pool* of `parallel_envs` fresh states and, each step, gathers a random pool entry per env and `jnp.where`s it in against the done mask. No control flow, no sync, one fused kernel. The pool is regenerated once per epoch.
- **Statistics accumulate on-device.** Per-episode return/length means *and* standard deviations come from running sums and sums-of-squares, reduced only at the epoch boundary.
- **Evaluation is one compiled rollout.** `Rollout.evaluate` puts the episode loop inside `jax.lax.while_loop` so the `~all(done)` check is evaluated on-device, with a static `max_steps` cap. It is deliberately not the acting program: `evaluate` is static and flips a stochastic actor to its mode, nothing is donated (eval must not consume the weights it is scoring), and it reuses one fixed key rather than advancing a stream.
- **Agents own their update schedule.** The trainer asks `agent.due_for_update(steps)` — a host-side comparison of step counts — and is explicitly forbidden from reading buffer device state (a flashbax `can_sample`) to decide itself.
- **Everything is precompiled up front**, with timings printed: reset, reset pool, a probe train step, the warmup rollout, the replay fill and the gradient step.

**Native physics cannot be traced, but it can be *called* from inside a trace.** `BatchVectorEnv` steps `mujoco.batch` through an **ordered** `io_callback`, so a chunk is still one dispatch, with acting, buffering and bookkeeping on device between steps. The task code then runs in the same XLA program with `mjx_env.step` swapped for the results the batch already computed, and raises if a task would have handed that step a different action. Ordered is what makes it sound: the rows are stateful, so steps must run in sequence with each other and with the loop feeding them.

The rows hold float64 state and the carry a float32 copy of it. A world is reloaded from the carry only when the carry differs from what the batch last returned — an auto-reset, a fresh reset, a task editing `data` — so an unchanged world keeps its low bits, and everything above the driver can treat the state as a plain pytree.

A JAX env cannot own mutable state inside a trace, so an env that adapts itself has the rollout thread its **`params`** through every call and through its `observe_params`/`epoch_refresh` hooks; `params` stays traced, so refreshing it each epoch does not retrigger compilation. Roxie never knows what the state *is*.

## Auto-reset

Auto-reset is same-step, and deliberately *not* Gymnasium's `AutoresetMode.NEXT_STEP`: a real reset of "however many envs happen to be done" has a data-dependent shape and cannot be jitted, whereas a gather from a fixed-size pool can. Precomputing the pool also turns a reset from a full `mj_forward` (paid purely to build the observation) into a memcpy.

The driver returns two things from a step for the same reason — a `Timestep` holding the *pre*-reset observation (the true next state, which is what the replay buffer must store) and a `VecState` holding the *post*-reset observation (what the next action is selected from).

## Truncation is not termination

An env declares three things and `JaxVectorEnv.step` combines them:

- **`terminal`** — a genuine failure (a fall, a NaN, drift past a limit). Zeroes the bootstrap in the Bellman target.
- **`truncal`** — the env's *own* non-failure cutoff (a reference trajectory ran out; a goal was reached). Must *still* bootstrap the next-state value; marking it terminal collapses Q at the cutoff.
- **the step limit** — the driver's `max_episode_steps`.

The reported flags are `terminated = terminal & ~truncal` and `truncated = truncal | step_limit`; an episode ends on either. Note the asymmetry: an env-internal truncation *clears* termination, but the step limit does not — a genuine fall on the very last step is still a fall. `observe_params` is handed `terminated`, never the union.

[`tests/test_env_protocol.py`](../tests/test_env_protocol.py) pins all of this against a toy env.

## Memory

The CPU backend exists to make one trade: **the GPU path is bounded by VRAM; the CPU path is bounded by system RAM.** On the GPU, the replay buffer, the contact arena, per-env reference data and the activations all compete for the same memory. On CPU, `parallel_envs` and replay capacity stop competing with the physics.

Two GPU-side allocation rules:

- **Buffer *capacity* is not update size.** PPO's `max_length_time_axis` is queue capacity, and `update()` drains every step, so occupancy oscillates between 0 and `sample_sequence_length`. Capacity far above the sample length reserves VRAM for a queue that never uses it.
- **Donate, or pay double.** An eager `replay.add` copies the whole queue every env step; the write is jitted with `donate_argnums=(0,)`. The same applies to `_<agent>_grad_steps`, where without donation XLA allocates a second copy of the read-only replay buffer per update, and to the warmup fill (`_batch_add`), where the transient 2× of the observation store can OOM the pool.

`nvidia-smi` reports XLA's **preallocated arena** (~75% of the card by default), not resident data. Real usage is `jax.local_devices()[0].memory_stats()["bytes_in_use"]` and `["peak_bytes_in_use"]`.

RAM is not a throughput lever: steady-state throughput is essentially flat in `parallel_envs`. Raising it on CPU is a batch-size decision. Where the extra RAM pays is an off-policy replay buffer living in system RAM instead of VRAM — which PPO, being on-policy, cannot use at all.

## Contact budgets: three backends, three models

- **Native MuJoCo (CPU)** allocates contacts dynamically. There is nothing to size, so `backend/batch_cpu.yaml` leaves the Warp budget keys null.
- **Warp (GPU)** uses a **single global contact arena shared across all vmapped worlds** (rows tagged by `contact__worldid`, so these Data leaves have no per-env batch dimension — which is why the auto-reset gather skips leaves lacking a leading per-env dim). `naconmax` therefore scales with `parallel_envs`, and must cover *broadphase AABB-overlap candidate pairs*, not just the contacts that resolve. `njmax` is per-world. `naccdmax` (GJK/EPA convex narrowphase) defaults to the full `naconmax` and reserves EPA scratch per buffer; a builder for a body where few geoms take the convex path should cap it below that. The humanoid is small enough that `naconmax: null` — mujoco_playground's own per-env tuning — covers it, so `backend/warp_gpu.yaml` sets none of these. Watch for "broadphase/narrowphase/nefc overflow" warnings on anything heavier. Roxie leaves those on but silences Warp's solver and linesearch iteration warnings, because `mujoco_playground` caps those iterations low on purpose and hitting the cap would otherwise print for every world on most steps.
- **MJX (`impl: jax`)** ignores `naconmax` and statically sizes contact arrays to *all potential geom pairs* — ~980 on a full humanoid, against Warp's budgeted handful. That is why `self_collisions=false` is the right default there and not elsewhere.

The collision setting is not purely a performance knob: **it changes the task.** A retargeted motion-capture reference that interpenetrates its own limbs has to run without self-collision, or the solver pushes the body out of the pose the tracking reward asks it to hold. On the dm_control suite this does not arise; the bodies come with their own contact exclusions.

## The Warp version pin

The `cuda` group pins `warp-lang==1.18.0`, the version `mujoco-mjx` 3.17 names in its own `warp` extra. The pin has to move with MuJoCo: mjx imports its Warp bridge at import time, and on an older warp that raises a `TypeError` rather than an `ImportError`, so `import mujoco.mjx` fails outright. `scripts/check_warp.py` answers whether a bump works.

## Warp's CUDA graph mode

`mjx.put_model(impl='warp')` defaults to `GraphMode.WARP`, whose graph-capture cache is keyed on per-step buffer *addresses*. Under JAX those addresses change every step, so a new CUDA graph is captured per step; the cache is bounded but eviction only drops the Python reference — the native host descriptors are never reclaimed. The result is host-RAM growth invisible to `jax.live_arrays()` and to GPU memory counters, which OOM-kills long runs.

Roxie therefore defaults Warp to **`graph_mode: WARP_STAGED_EX`**: capture once onto fixed staging buffers, then replay every step plus a cheap device→staging memcpy. `JAX`/`NONE` also avoid the leak but launch this step's many small kernels eagerly instead of replaying a graph.

## Where the run runs

One knob places the whole run:

```yaml
runtime:
  device: gpu    # networks, optimizers, replay buffer AND physics
```

`cpu`, `gpu`, or null for "whatever JAX picks". It is read by `train.py` before the first `jax.*` call (`loader.resolve_placement`) and then checked against reality by the startup banner:

```
  AGENT (networks, optimizers, replay):  GPU   [cuda]
  ENV   (physics):                       GPU   [warp kernels on the JAX device (cuda)]
  OK - matches runtime.device (agent) = 'gpu'
  OK - matches runtime.device (env) = 'gpu'
```

Both halves are reported and both are checked, because a cell declaring `gpu` can land on a machine with no card.

The agent is pure JAX, so the device *is* the JAX platform, and MJX and Warp physics are traced into that same XLA program. **`impl: batch` is the one that cannot follow**: `mujoco.batch` steps on CPU threads whatever JAX does, so `resolve_placement` pins such a run to the CPU — and `runtime.device: gpu` alongside it is a launch-time error rather than a split. There is no CPU-physics / GPU-learner split: it is all-CPU or all-GPU.

The CPU pin is not just a default. The batch never touches the GPU, but the agent is plain JAX and would otherwise still claim the card — and preallocate most of it — so "running on CPU" would leave the GPU fully occupied. **The CPU backend frees the card entirely.**

## Determinism and reproducibility

Warp's atomic contact reductions are **not bit-reproducible**: two identical rollouts diverge by ~1e-6 per step, which chaotic contact dynamics amplify. C MuJoCo playback is exact. Hence:

- `play.py` unconditionally forces `JAX_PLATFORMS=cpu` and runs the physics in C MuJoCo, coercing `impl: warp` → `impl: jax` only so the task can be built without Warp. Checkpoints are agent-side, so the same weights load on any backend, but C is not numerically identical to the engine a run trained on, so a replay's score can differ a little from its eval. Warp-only knobs are ignored.
- Evaluation uses `test_episodes: 10` rather than 1, so Warp's non-reproducibility shows up as visible spread rather than a silently shifting single number.

Evaluation itself is pinned: fixed reset keys held constant across epochs *and* across runs, so a change in `test/score` is a change in the policy. `test/distinct_starts` is logged because `test/length/std == 0` is ambiguous — it means either a degenerate eval or a policy saturating the episode cap.

## Ordering gotchas (env vars vs `jax.config`)

The GPU/CPU settings have to be applied at different moments, and getting this wrong fails *silently*. `train.py` handles each in the right place:

| Setting | Mechanism | Why |
|---|---|---|
| `JAX_PLATFORMS` (from `device=` on the CLI) | `os.environ`, **before `import jax`** | Parsed once at `import jax`. Setting it afterwards is ignored. |
| `runtime.device` (from the composed config) | `jax.config.update("jax_platforms", ...)` | JAX is already imported by the time the config is composed. `gpu` forces nothing — it lets JAX pick the accelerator, so a missing card is a banner line rather than an import error. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_ALLOCATOR` | `os.environ`, after import, before the first `jax.*` call | The PJRT C++ client reads these when it lazily initializes. |
| `XLA_FLAGS=--xla_gpu_autotune_level=0` | `os.environ` | Required on Blackwell (RTX 5080): XLA's autotuning phase hangs indefinitely compiling MJX kernels. Skipped when a `vision` group is selected: untuned cuDNN runs the pixel encoder's convolutions ~70× slower, and a vision run's physics is warp, which compiled fine with autotuning on. |
| `runtime.matmul_precision` | `jax.config.update` | Global — reaches agent networks *and* MJX physics, so set it for a whole sweep, never a single arm. |

**Backend env vars are decided from the composed config, not from `sys.argv`.** `env.impl: warp` set inside an experiment YAML never appears in argv, so an argv sniff misses it — and JAX then preallocates its default 75% of VRAM and the Warp reset constants OOM. The `impl == "warp"` branch also switches XLA to CUDA's async allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`): the BFC allocator caps out on *fragmentation* rather than on a leak, and PPO's rollout-length growth produces enough alloc/free churn to trigger it. Both are `setdefault`, so the environment still wins.

## Keeping the backends honest

The cells run the same `mujoco_playground` tasks — same XML, reset, observation and reward code — on a shared 0–1000 return scale, so a score gap between them is a finding about the physics or the hardware. What keeps that true:

- **The task code is literally shared.** [`tests/test_batch_physics.py`](../tests/test_batch_physics.py) steps every task once on MJX and on `mujoco.batch` from the same reset and fails if the observation or reward disagree.
- **A task may not transform its action.** The batch steps before the task runs, so `BatchVectorEnv` raises rather than applying the wrong controls.
- **Termination/truncation is one implementation** (see [above](#truncation-is-not-termination)).
- **Derived fields lag the state by one substep**, because `mjx.step` leaves them that way and the tasks were written against it; `BatchVectorEnv` steps with `mjSTAGE_NONE` to match.

The agreement does not extend past that first step: on the humanoid, MJX and native MuJoCo trajectories have visibly parted by step 20.
