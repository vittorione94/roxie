# Backends: CPU vs GPU

Roxie runs on the GPU or on the CPU — the *physics* and the *agent* (networks, optimizers, replay buffer) together, never split.

*Back to the [README](../README.md).*

## The three physics backends

| | `impl: jax` (MJX) | `impl: warp` (mujoco_warp) | EnvPool (native MuJoCo) |
|---|---|---|---|
| Device | GPU (or CPU) | GPU only | CPU only |
| Vectorization | `jax.vmap` over a traced step | `jax.vmap` → Warp kernels | C++/Python thread pool, one `MjData` per env |
| Batch dim lives in | the XLA program | the XLA program | native code; Python sees `(N, ...)` numpy |
| Contact allocation | **static**, sized to *all* potential geom pairs | **budgeted** — `naconmax` / `njmax` / `naccdmax` | **dynamic**, MuJoCo allocates as needed |
| Rollout | `JaxRollout` | `JaxRollout` | `EnvPoolRollout` |
| Bounded by | VRAM | VRAM | system RAM + core count |
| Built by | `loader.build_playground_env` | `loader.build_playground_env` (`impl: warp`) | `loader.build_envpool_env` |

Selection is a Hydra config group: [`experiments/dmc/backend/`](../experiments/dmc/backend/) holds `warp_gpu.yaml`, `envpool_cpu.yaml`, `mjx_gpu.yaml` and `mjx_cpu.yaml`, and a launchable picks one in its `defaults:`. Everything backend-mechanical (the builder in `env._target_`, `impl`, solver budgets, graph mode, `runtime.device`) lives in that group; per-experiment tuning stays in the launchable.

`impl` is a key of the playground builder only — it picks which kernels step the *same* JAX env. EnvPool is not a third value of it but the other builder, named in `env._target_`, and carries no `impl:` key; `loader.uses_envpool` asks the builder wherever roxie needs to know (pinning the run to the CPU in `train.py`, "there is no viewer for a pool" in `play.py`).

The `envpool` column is **not** the same program on a different device: playground reimplements the dm_control tasks as JAX/MJX programs, while EnvPool wraps dm_control's own C++ physics. Two independent implementations of one task specification — see [Keeping the backends honest](#keeping-the-backends-honest).

`train.py` prints a banner at startup reporting the backend that *actually* loaded (read off the constructed env, not echoed from config) and flags a mismatch against what was requested.

## Why there are two rollouts

Every env presents the same interface ([environments.md](environments.md)). `Trainer._run` is therefore a single loop over both, and so is the rollout above it: `Rollout` holds the chunk, the warmup and the eval loop, and `build_rollout` picks the subclass — `EnvPoolRollout` for an `EnvPoolVectorEnv`, `JaxRollout` for everything else. A subclass supplies only what genuinely differs: how a batch of envs is reset, how one step is taken, and what the epoch boundary does to the env.

**The loop must never touch the host.** Every quantity a Python `if` would branch on — done flags, buffer occupancy, episode returns — lives in a device array, and reading one forces a blocking device→host sync. So:

- **`collect` is the unit of work, not `step`.** A whole chunk — acting, physics, buffering, running statistics, episode bookkeeping — fuses into one `jax.jit` program. The chunk is never widened past one `steps_between_updates` window, which keeps the actor fixed for its duration.
- **Auto-reset is branchless** (`JaxRollout`; a C++ pool resets itself internally). The loop pre-builds a *reset pool* of `parallel_envs` fresh states and, each step, gathers a random pool entry per env and `jnp.where`s it in against the done mask. No control flow, no sync, one fused kernel. The pool is regenerated once per epoch.
- **Statistics accumulate on-device.** Per-episode return/length means *and* standard deviations come from running sums and sums-of-squares, reduced only at the epoch boundary.
- **Evaluation is one compiled rollout.** `Rollout.evaluate` puts the episode loop inside `jax.lax.while_loop` so the `~all(done)` check is evaluated on-device, with a static `max_steps` cap. It is deliberately not the acting program: `evaluate` is static and flips a stochastic actor to its mode, nothing is donated (eval must not consume the weights it is scoring), and it reuses one fixed key rather than advancing a stream.
- **Agents own their update schedule.** The trainer asks `agent.due_for_update(steps)` — a host-side comparison of step counts — and is explicitly forbidden from reading buffer device state (a flashbax `can_sample`) to decide itself.
- **Everything is precompiled up front**, with timings printed: the warmup rollout, the replay fill and the gradient step for both backends, plus reset, reset pool and a probe train step on the JAX path. A pool step cannot be taken back, so `EnvPoolRollout` traces its callback against the driver's declared `step_spec` instead of probing.

**A C++ pool cannot be traced, but it can be *called* from inside a trace.** `EnvPoolRollout.advance` (`pool_advance`) runs the pool's step as an **ordered** `io_callback`, so a pool chunk is one dispatch too, with acting, buffering and bookkeeping staying on device between pool steps. Ordered is what makes it sound: the pool is stateful, so its steps must run in sequence with each other and with the loop feeding them. The consequence is that episode bookkeeping is on device for *both* backends, and both return the same per-chunk summary.

What the two must agree on is semantics. The termination rule is shared code (`JaxVectorEnv.step`, mirrored in the CPU pool); the rest is maintained by hand:

- both update observation-normalization statistics from the live policy's state distribution over every step it takes, folded in once per chunk — see [agents.md](agents.md#observation-normalization);
- both log the same keys, including `test/distinct_starts`, `test/score_per_step`, the env's own `metrics` dict, per-agent diagnostics, GPU telemetry and the host-memory watch;
- both let the env refresh whatever it adapts about itself and *then* rebuild the reset pool, in that order, at the epoch boundary.

An env that adapts its own state shows the structural difference. A JAX env cannot own mutable state inside a trace, so the rollout threads the env's **`params`** through every call and through its `observe_params`/`epoch_refresh` hooks; `params` stays traced, so refreshing it each epoch does not retrigger compilation. A CPU pool resets in plain Python and owns its state, so the rollout only tells it *when* to refresh — a bare `epoch_refresh()`. In neither case does roxie know what the state *is*.

## Auto-reset

Auto-reset is same-step on both backends (EnvPool convention), and deliberately *not* Gymnasium's `AutoresetMode.NEXT_STEP`: a real reset of "however many envs happen to be done" has a data-dependent shape and cannot be jitted, whereas a gather from a fixed-size pool can. Precomputing the pool also turns a reset from a full `mj_forward` (paid purely to build the observation) into a memcpy, which matters on a hand-written CPU pool; EnvPool's own pools reset in C++ and need none of this.

The JAX driver returns two things from a step for the same reason — a `Timestep` holding the *pre*-reset observation (the true next state, which is what the replay buffer must store) and a `VecState` holding the *post*-reset observation (what the next action is selected from). A C++ pool resets internally and hands back only the post-reset observation, so there the two are the same array. That is the one place the backends genuinely differ: on the JAX path the true final obs feeds the obs-normalization statistics, on the CPU path the reset obs does.

## Truncation is not termination

Shared by both paths. An env declares three things and `JaxVectorEnv.step` combines them:

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

RAM is not a throughput lever: the CPU pool costs ~1.6 MB/env, but steady-state throughput is essentially flat in `parallel_envs`. Raising it on CPU is a batch-size decision. Where the extra RAM pays is an off-policy replay buffer living in system RAM instead of VRAM — which PPO, being on-policy, cannot use at all.

## Contact budgets: three backends, three models

- **Native MuJoCo (CPU)** allocates contacts dynamically. There is nothing to size, which is why `backend/envpool_cpu.yaml` omits the Warp budget keys.
- **Warp (GPU)** uses a **single global contact arena shared across all vmapped worlds** (rows tagged by `contact__worldid`, so these Data leaves have no per-env batch dimension — which is why the auto-reset gather skips leaves lacking a leading per-env dim). `naconmax` therefore scales with `parallel_envs`, and must cover *broadphase AABB-overlap candidate pairs*, not just the contacts that resolve. `njmax` is per-world. `naccdmax` (GJK/EPA convex narrowphase) defaults to the full `naconmax` and reserves EPA scratch per buffer; a builder for a body where few geoms take the convex path should cap it below that. The dm_control bodies are small enough that `naconmax: null` — mujoco_playground's own per-env tuning — covers all 25, so `backend/warp_gpu.yaml` sets none of these. Watch for "broadphase/narrowphase/nefc overflow" warnings on anything heavier.
- **MJX (`impl: jax`)** ignores `naconmax` and statically sizes contact arrays to *all potential geom pairs* — ~980 on a full humanoid, against Warp's budgeted handful. That is why `self_collisions=false` is the right default there and not elsewhere.

The collision setting is not purely a performance knob: **it changes the task.** A retargeted motion-capture reference that interpenetrates its own limbs has to run without self-collision, or the solver pushes the body out of the pose the tracking reward asks it to hold. On the dm_control suite this does not arise; the bodies come with their own contact exclusions.

## The Warp version pin

The `cuda` group pins `warp-lang>=1.11,<1.13`. MuJoCo's vendored Warp bridge imports `warp._src.jax_experimental.ffi.GraphMode` and reads `warp.types.warp_type_to_np_dtype`. Warp 1.13 dropped the latter from the public API and 1.14 graduated `jax_experimental` into `jax`, moving both out from under the bridge. `uv run python scripts/check_warp.py` exits 0 when the installed Warp is usable.

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

The agent is pure JAX, so the device *is* the JAX platform, and a `build_playground_env` env is traced into that same XLA program. **EnvPool is the one builder that cannot follow**: its pools step native MuJoCo on C++ threads whatever JAX does, so `resolve_placement` pins an EnvPool run to the CPU — and `runtime.device: gpu` alongside it is a launch-time error rather than a split. There is no CPU-physics / GPU-learner split: it is all-CPU or all-GPU.

The CPU pin is not just a default. The pool's physics never touches the GPU, but the agent is plain JAX and would otherwise still claim the card — and preallocate most of it — so "running on CPU" would leave the GPU fully occupied. **The CPU backend frees the card entirely.**

## Determinism and reproducibility

Warp's atomic contact reductions are **not bit-reproducible**: two identical rollouts diverge by ~1e-6 per step, which chaotic contact dynamics amplify. CPU/MJX playback is exact. Hence:

- `play.py` unconditionally forces `JAX_PLATFORMS=cpu` and coerces `impl: warp` → `impl: jax`. Checkpoints are agent-side, so the policy replays identically on either backend, and Warp-only knobs are ignored by the MJX builder.
- Evaluation uses `test_episodes: 10` rather than 1, so Warp's non-reproducibility shows up as visible spread rather than a silently shifting single number.

Evaluation itself is pinned: fixed reset keys held constant across epochs *and* across runs, so a change in `test/score` is a change in the policy. `test/distinct_starts` is logged because `test/length/std == 0` is ambiguous — it means either a degenerate eval or a policy saturating the episode cap.

## Ordering gotchas (env vars vs `jax.config`)

The GPU/CPU settings have to be applied at different moments, and getting this wrong fails *silently*. `train.py` handles each in the right place:

| Setting | Mechanism | Why |
|---|---|---|
| `JAX_PLATFORMS` (from `device=` on the CLI) | `os.environ`, **before `import jax`** | Parsed once at `import jax`. Setting it afterwards is ignored. |
| `runtime.device` (from the composed config) | `jax.config.update("jax_platforms", ...)` | JAX is already imported by the time the config is composed. `gpu` forces nothing — it lets JAX pick the accelerator, so a missing card is a banner line rather than an import error. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_ALLOCATOR` | `os.environ`, after import, before the first `jax.*` call | The PJRT C++ client reads these when it lazily initializes. |
| `XLA_FLAGS=--xla_gpu_autotune_level=0` | `os.environ` | Required on Blackwell (RTX 5080): XLA's autotuning phase hangs indefinitely compiling MJX kernels. |
| `runtime.matmul_precision` | `jax.config.update` | Global — reaches agent networks *and* MJX physics, so set it for a whole sweep, never a single arm. |

**Backend env vars are decided from the composed config, not from `sys.argv`.** `env.impl: warp` set inside an experiment YAML never appears in argv, so an argv sniff misses it — and JAX then preallocates its default 75% of VRAM and the Warp reset constants OOM. The `impl == "warp"` branch also switches XLA to CUDA's async allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`): the BFC allocator caps out on *fragmentation* rather than on a leak, and PPO's rollout-length growth produces enough alloc/free churn to trigger it. Both are `setdefault`, so the environment still wins.

## Keeping the backends honest

The two cells run the same 25 dm_control tasks through two independent implementations on a shared 0–1000 return scale, so a score gap between them is a finding about one of the implementations rather than a hardware artefact. What "same task" requires:

- **The reward has to be the same function.** dm_control rewards are normalised tolerances in [0, 1], so both sides landing in the same range is necessary but not sufficient.
- **The actuation mode has to match.** Torque motors versus PD position servos is a different task.
- **Termination/truncation semantics have to match**, or one side bootstraps where the other does not (see [above](#truncation-is-not-termination)).
- **Observations need not be byte-identical, and here they are not.** mujoco_playground omits some of dm_control's observation groups on a few tasks (`HumanoidRun` is 67-dim against EnvPool's 95, `FingerSpin` 9 against 12; most match). Both sides are self-consistent, so each cell trains and evaluates against its own spec. The consequence is that **a policy trained on one cell cannot be loaded against the other**, which is why the weights export publishes one cell and names it.

`roxie.environment.suites` is the single place the two naming schemes are reconciled (`BallInCup` ↔ `BallInCupCatch-v1`), and [`tests/test_benchmark_suite.py`](../tests/test_benchmark_suite.py) fails if a task exists on one side and not the other.
