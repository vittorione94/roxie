# Backends: CPU vs GPU

Roxie can put the *physics* on the GPU or the CPU, and — independently — the *agent* (networks, optimizers, replay buffer) on the GPU or the CPU. Those two choices are what the rest of this document is about.

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
| Built by | `roxie.environment.loader.build_playground_env` | `roxie.environment.loader.build_playground_env` (`impl: warp`) | `roxie.environment.loader.build_envpool_env` |

Selection is a Hydra config group: [`experiments/dmc/backend/`](../experiments/dmc/backend/) holds `warp_gpu.yaml`, `envpool_cpu.yaml`, `mjx_gpu.yaml` and `mjx_cpu.yaml`, and a launchable picks one in its `defaults:`. Everything backend-mechanical (the builder in `env._target_`, `impl`, solver budgets, graph mode, and the `agent.device`/`env.device` pair) lives in that group; per-experiment tuning stays in the launchable. That is what makes the cells of the [release benchmark](../experiments/README.md) a genuine A/B: they differ in exactly one block.

`impl` is a key of the playground builder only — it picks which kernels step the *same* JAX env. EnvPool is not a third value of it but the other builder, named in `env._target_`, and its column carries no `impl:` key at all; `loader.uses_envpool` asks the builder wherever roxie needs to know (learner placement in `train.py`, "there is no viewer for a pool" in `play.py`).

Note what the `envpool` column is and is not. It is **not** the same program on a different device — playground reimplements the dm_control tasks as JAX/MJX programs, while EnvPool wraps dm_control's own C++ physics. Two independent implementations of one task specification. The release benchmark runs all 25 dm_control tasks through both for exactly that reason; see [Keeping the backends honest](#keeping-the-backends-honest).

`train.py` prints a loud banner at startup reporting the backend that *actually* loaded (read off the constructed env, not echoed from config) and flags a mismatch against what was requested.

## Why there are two rollouts

Every env in roxie — playground, EnvPool, a task living in its own repo — presents the same interface, described in [environments.md](environments.md). `Trainer._run` is therefore a single loop, and the only thing that varies is the *rollout*: `build_rollout` picks `EnvPoolRollout` for an `EnvPoolVectorEnv` and `JaxRollout` for everything else. They are not gratuitously duplicated; the two obey opposite constraints.

**`JaxRollout` must never touch the host.** Every quantity that a Python `if` would want to branch on — episode-done flags, buffer occupancy, episode returns — lives in a device array, and reading one forces a blocking device→host sync that serializes the whole asynchronous GPU pipeline. So the JAX loop is written to keep the host out of the way:

- **Auto-reset is branchless.** Instead of resetting done envs on demand, the loop pre-builds a *reset pool* of `parallel_envs` fresh states and, each step, gathers a random pool entry per env and `jnp.where`s it in against the done mask. No control flow, no sync, one fused kernel. The pool is regenerated once per epoch.
- **Statistics accumulate on-device.** Per-episode return/length means *and standard deviations* are computed from running sums and sums-of-squares reduced only at the epoch boundary, rather than keeping a host-side list of episodes (which would sync every step).
- **Evaluation is one compiled rollout.** `_make_eval_fn` puts the entire episode loop inside `jax.lax.while_loop` so the `~all(done)` termination check is evaluated on-device, with a static `max_steps` cap to bound compute. A Python `while` here would drag a device array back per step.
- **Agents gate their own updates.** The trainer calls `agent.update(steps=...)` unconditionally and the agent decides internally whether to run gradient steps. The trainer is explicitly forbidden from reading buffer device state (e.g. a flashbax `can_sample`) to make that decision.
- **Everything is precompiled up front**, with timings printed: reset, reset pool, train step, agent step, replay add, warmup rollout, replay fill, gradient step. A cold compile mid-run is a stall.

**`EnvPoolRollout` has none of those constraints and should not pretend to.** The pool steps in C++, hands back numpy, and Python branching is free. So it is a plain loop with numpy accumulators and a Python `for` eval loop. Trying to force the JAX idioms here would only add dispatch overhead.

What the two *must* agree on is semantics. The termination rule is now shared code (`JaxVectorEnv.step`, mirrored explicitly in the CPU pool), but the rest is still maintained by hand:

- both update observation-normalization statistics from the live policy's state distribution every step (not just during warmup);
- both log the same keys, including `test/distinct_starts`, `test/score_per_step`, the env's own `metrics` dict, per-agent diagnostics (`pop_diagnostics`), GPU telemetry and the host-memory watch;
- both let the env refresh whatever it adapts about itself and *then* rebuild the reset pool, in that order, at the epoch boundary.

An env that adapts its own state is the clearest illustration of the structural difference. A JAX env cannot own mutable state inside a trace, so the rollout threads the env's **`params`** through every call and through the env's own `observe_params`/`epoch_refresh` hooks; `params` stays traced, precisely so that refreshing it each epoch does not retrigger compilation. A CPU pool resets in plain Python and simply owns its state, so the rollout only has to tell it *when* to refresh — a bare `epoch_refresh()`. Same feature, opposite ownership, and in neither case does roxie know what the state *is*: the negative mining over start phases in [roxie-mocap](https://github.com/vittorione94/roxie-mocap) is written entirely against those hooks — from a different repository — and its diagnostics ride out with the ordinary per-step metrics rather than through a channel of their own.

## Auto-reset: the same trick, for two different reasons

The GPU path uses a reset pool to avoid host synchronization. A hand-written CPU pool ended up wanting one too, for an unrelated reason: an on-demand reset pays a full `mj_forward` purely to build the observation, and an untrained policy on a task that terminates early resets ~20% of envs *per step*. Precomputing a pool turns a reset into a memcpy. That was the single biggest CPU throughput win measured on the mocap humanoid (now [roxie-mocap](https://github.com/vittorione94/roxie-mocap)) — and, incidentally, a parity fix, since it is what the JAX path had always done. EnvPool's own pools reset in C++ and need none of this; the lesson applies to anyone writing a pool by hand.

Auto-reset is same-step on both backends (EnvPool convention), and deliberately *not* Gymnasium's `AutoresetMode.NEXT_STEP`: the reason is mechanical, in that a real reset of "however many envs happen to be done" has a data-dependent shape and cannot be jitted, whereas a gather from a fixed-size pool can.

The JAX driver returns two things from a step for exactly this reason — a `Timestep` holding the *pre*-reset observation (the true next state, which is what the replay buffer must store) and a `VecState` holding the *post*-reset observation (what the next action is selected from). A C++ pool resets internally and hands back only the post-reset observation, so there the two are the same array. That is the one place the backends genuinely differ: on the JAX path the true final obs feeds the obs-normalization statistics, on the CPU path the reset obs does.

## Truncation is not termination

Shared by both paths, and worth stating because it is easy to get wrong. An env declares three things and `JaxVectorEnv.step` combines them:

- **`terminal`** — a genuine failure (a fall, a NaN, drift past a limit). Zeroes the bootstrap in the Bellman target.
- **`truncal`** — the env's *own* non-failure cutoff (a reference trajectory ran out; a goal was reached). Must *still* bootstrap the next-state value; marking it terminal collapses Q at the cutoff.
- **the step limit** — the driver's `max_episode_steps`.

The reported flags are `terminated = terminal & ~truncal` and `truncated = truncal | step_limit`; an episode ends on either. Note the asymmetry: an env-internal truncation *clears* termination, but the step limit does not — a genuine fall on the very last step is still a fall. `observe_params` is handed `terminated`, never the union, so an env adapting to failure never mistakes an env-internal or step-limit cutoff for one.

[`tests/test_env_protocol.py`](../tests/test_env_protocol.py) pins all of this against a toy env.

## Memory: the actual reason to run on CPU

This is the trade the CPU backend exists to make: **the GPU path is bounded by a 16 GB card; the CPU path is bounded by 64 GB of system RAM.** On the GPU, the replay buffer, the contact arena, any per-env reference data and the activations all compete for the same VRAM. On CPU, `parallel_envs` and replay capacity stop competing with the physics for memory.

Two GPU-side allocation traps are worth knowing, because both were expensive to find:

- **Buffer *capacity* is not update size.** PPO's `max_length_time_axis` is queue capacity, and `update()` drains every step, so occupancy oscillates between 0 and `sample_sequence_length`. A capacity of 512 against a sample length of 32 reserved 2.31 GB of VRAM for a queue that never exceeds 32 entries.
- **Donate, or pay double.** `replay.add` called eagerly copies the whole queue every env step; jitting it with `donate_argnums=(0,)` took `agent.add` from 43 ms to 1.8 ms. The same applies to the fused `_grad_steps`: without donation, XLA allocates a full second copy of the read-only replay buffer on every update (~1.4 GB), and the same trap applies to the warmup replay fill, where the transient 2× of the buffer's observation store OOMs the pool outright.

Also: what `nvidia-smi` reports is XLA's **preallocated arena** (~75% of the card by default), not resident data. To see real usage, read `jax.local_devices()[0].memory_stats()["bytes_in_use"]` and `["peak_bytes_in_use"]`.

The important negative result: **RAM is not a throughput lever.** The CPU pool costs ~1.6 MB/env, so 5000 envs is ~8.7 GB of a 64 GB box — but steady-state throughput is essentially flat in `parallel_envs` (measured: 25.1k sps @ 1000 → 27.8k @ 5000). Raising `parallel_envs` on CPU is a batch-size / gradient-quality decision, not a speed one. The one place the extra RAM genuinely pays is an **off-policy replay buffer living in system RAM instead of VRAM** — which PPO, being on-policy with no replay, cannot use at all.

## Contact budgets: three backends, three models

Collision handling is where the backends diverge most sharply, and it drives configuration on both the memory *and* the task-definition axis.

- **Native MuJoCo (CPU)** allocates contacts dynamically. There is nothing to size. This is why `backend/envpool_cpu.yaml` simply omits the Warp budget keys.
- **Warp (GPU)** uses a **single global contact arena shared across all vmapped worlds** (rows tagged by `contact__worldid`, so these Data leaves have no per-env batch dimension — which is exactly why the trainer's auto-reset gather skips leaves lacking a leading per-env dim). `naconmax` therefore scales with `parallel_envs`, and it must cover *broadphase AABB-overlap candidate pairs* (~48/world peak here), not just the ~15 contacts that actually resolve. `njmax` is per-world. A third budget, `naccdmax` (GJK/EPA convex narrowphase), defaults to the full `naconmax` and reserves ~90 MB of EPA scratch per buffer; a bespoke builder for a body where few geoms take the convex path should cap it well below that. The dm_control bodies are small enough that `naconmax: null` — mujoco_playground's own per-env tuning — is right for all 25, so `backend/warp_gpu.yaml` sets none of these. Watch for "broadphase/narrowphase/nefc overflow" warnings on anything heavier.
- **MJX (`impl: jax`)** ignores `naconmax` entirely and statically sizes contact arrays to *all potential geom pairs* — ~980 on a full humanoid. Self-collisions are therefore roughly **75× heavier on MJX than on Warp**, which is why `self_collisions=false` is the right default there and not elsewhere, and why a humanoid-scale body can be affordable on Warp and not on MJX at the same env count.

Note that the collision setting is not purely a performance knob: **it changes the task.** A retargeted motion-capture reference that interpenetrates its own limbs has to run without self-collision, or the solver pushes the body out of the very pose the tracking reward is asking it to hold — see [roxie-mocap](https://github.com/vittorione94/roxie-mocap). On the dm_control suite this does not arise; the bodies come with their own contact exclusions.

## The Warp version pin

The `cuda` dependency group pins `warp-lang>=1.11,<1.13`. This is not conservatism: MuJoCo's vendored Warp bridge imports `warp._src.jax_experimental.ffi.GraphMode` and reads `warp.types.warp_type_to_np_dtype`. Warp 1.13 dropped the latter from the public API and 1.14 graduated `jax_experimental` into `jax`, moving both out from under the bridge. `uv run python scripts/check_warp.py` exits 0 when the installed Warp is usable.

## Warp's CUDA graph mode

`mjx.put_model(impl='warp')` defaults to `GraphMode.WARP`, whose graph-capture cache is keyed on per-step buffer *addresses*. Under JAX those addresses change every step, so a new CUDA graph is captured per step; the cache is bounded but eviction only drops the Python reference — the native host descriptors are never reclaimed. The result is ~0.25 GB of host-RAM growth per 1M steps, invisible to `jax.live_arrays()` and to GPU memory counters, which OOM-kills long runs (typically during a checkpoint save, since that adds a transient spike).

Roxie therefore defaults Warp to **`graph_mode: WARP_STAGED_EX`**: capture once onto fixed staging buffers, then replay every step plus a cheap device→staging memcpy. The obvious-looking `JAX`/`NONE` modes also avoid the leak but are much slower, because they launch this step's many small kernels eagerly instead of replaying a graph. The recapture was the problem, not graph replay.

## Where the agent runs (independent of where the physics runs)

Each half declares its own hardware, in its own config block:

```yaml
agent:
  device: gpu    # networks, optimizers, replay buffer
env:
  device: cpu    # physics
```

`cpu`, `gpu`, or null for "whatever JAX picks". Both are read by `train.py` before the first `jax.*` call (`loader.resolve_placement`) and both are then **checked against reality** by the startup banner, which prints where each half actually ended up:

```
  AGENT (networks, optimizers, replay):  GPU   [cuda]
  ENV   (physics):                       CPU   [native MuJoCo, C++ thread pool]
  OK - matches agent.device = 'gpu'
  OK - matches env.device = 'cpu'
```

Only one of the two is a control. `agent.device` decides, because the agent is pure JAX and therefore *is* the JAX platform. `env.device` decides nothing by itself — what it gets depends on the builder, which is why declaring it is worth doing:

| Builder | `env.device` | Notes |
|---|---|---|
| `build_envpool_env` | genuinely independent | Native MuJoCo on C++ threads, on the CPU whatever JAX does. This is the real hybrid. |
| `build_playground_env` | follows `agent.device` | Traced into the same XLA program as the agent; declaring the other device gets a loud MISMATCH line, not a split. |

With CPU physics there are then two viable configurations:

| | Physics | Agent | Notes |
|---|---|---|---|
| `agent.device: cpu` | CPU pool | CPU (JAX) | **Default when `agent.device` is unset on an EnvPool run.** The card is genuinely free. |
| `agent.device: gpu` | CPU pool | GPU (JAX) | Faster, but puts ~1.4 GB back on the card and re-couples the run to it. |

The default matters more than it looks. The EnvPool pool's physics never touches the GPU, but the agent is plain JAX and would otherwise still claim the card — and preallocate most of it — so "running on CPU" would leave the GPU fully occupied, which is the opposite of the point. **The CPU backend is meant to free the card entirely**, so `backend/envpool_cpu.yaml` pins both halves to `cpu`.

Measured on a 12-core/24-thread 7900X with a bespoke CPU pool, PPO, `parallel_envs=1000`, obs 1069, nets `[1024, 512, 256]` — i.e. on a humanoid-scale task ([roxie-mocap](https://github.com/vittorione94/roxie-mocap)), which is where the learner is heavy enough for the split to matter:

| Configuration | Throughput |
|---|---|
| CPU physics + GPU learner | ~17.2k sps |
| Fully GPU-free | ~13.6k sps |
| Physics-only ceiling (no learner) | ~38–40k sps |

Going GPU-free cost ~20% there, and that gap is entirely the dense actor/critic GEMMs — which is what the GPU is for. On the dm_control suite the nets are `[256, 256]` and the effect is much smaller, which is why the release grid's GPU-free cell is simply `agent.device: cpu` and there is no hybrid cell in the default grid.

## Async learner (CPU physics + GPU learner only)

In the hybrid configuration the loop is otherwise fully serial: step the CPU physics, then run the GPU gradient burst, each device idle while the other works. For agents where the learner dominates (D4PG's distributional critic, say), that caps throughput well below the CPU physics ceiling.

`trainer.async_learner: true` moves gradient updates onto a background thread that is the **sole owner** of `agent.state` (networks, optimizers, replay buffer, obs stats). The acting thread never touches it: it selects actions from a behaviour-actor snapshot the learner publishes, steps the envs, and pushes transitions through a queue. That single-owner rule is not stylistic — the fused `_grad_steps` donates the whole train state including the replay buffer, and donation is only sound because nothing else references it concurrently. Eval and checkpointing bracket themselves with `pause()`/`resume()` so they observe a quiescent state.

Replay ratio is preserved (the learner runs one burst per `steps_between_updates` boundary crossed, so it can never run ahead of collected data); the behaviour policy lags by up to one burst, which is standard for async off-policy RL. `learner_chunk` trades interleaving granularity against per-dispatch overhead.

This is an `envpool`-loop-only option, and it is pointless when the agent is also on CPU — there are no two devices to overlap.

## CPU parallelism: Amdahl, not cores

EnvPool owns its own C++ thread pool, so on the `envpool_cpu` cell there is nothing here to tune — which is most of the argument for using it rather than writing a pool.

If you *do* write one (roxie drives any pool that speaks the Gymnasium 5-tuple), the lessons from the hand-written one that used to live here, now in [roxie-mocap](https://github.com/vittorione94/roxie-mocap), are worth having. It ran one `MjData` per env behind a persistent `ThreadPoolExecutor`; MuJoCo's Python bindings release the GIL inside `mj_step`, so threads scale across cores without pickling, and on the 7900X 24 threads beat 16 once env count was well above core count because the SMT siblings absorb memory stalls. The physics saturated the machine; what capped average utilization was the **single-threaded numpy between physics passes**, and the rule that emerged was entirely about *operation size*:

- **Large numpy ops thread; thousands of tiny per-env assignments do not.** Splitting observation assembly across 6 workers by env range cut it from 22.8 ms to 10.6 ms (bit-identical output) because it is large ufuncs and concatenates that release the GIL. Conversely, moving the per-env `MjData` → buffer field copies into worker threads made things *worse* — 24 threads just queue on the GIL, and physics-only throughput fell from 40k to 29k sps.
- **Defer GIL-held Python into the worker.** Pooled reset went from 23.1 ms to 2.1 ms by deferring the per-env `mj_resetData` + qpos/qvel restore into the worker thread behind a flag. The numpy gathers were only 0.84 ms of that; the other 22 ms was ~13 µs of Python per env.

Serial fraction went from 33.6% to 18.6% of wall-clock at 5000 envs, and the remainder — ~30k small assignments capturing `MjData` into buffers — is provably not threadable in Python. Eliminating it needs `mujoco.rollout`, the C++ batched threaded API. Which is, in the end, the same observation as "use EnvPool".

## Determinism and reproducibility

Warp's atomic contact reductions are **not bit-reproducible**: two identical rollouts diverge by ~1e-6 per step, which the chaotic contact dynamics then amplify. CPU/MJX playback is exact. This is why:

- `play.py` unconditionally forces `JAX_PLATFORMS=cpu` and coerces `impl: warp` → `impl: jax` for playback. Checkpoints are agent-side, so the policy replays identically on either backend, and Warp-only knobs are simply ignored by the MJX builder.
- The release grid uses `test_episodes: 10` rather than 1. dm_control resets are stochastic, so most of that is a genuine sample size — but on a task with a deterministic reset the same setting is still worth keeping, because it is what makes Warp's non-reproducibility show up as visible spread rather than as a silently shifting single number.

Evaluation itself is pinned: fixed reset keys held constant across epochs *and* across runs, so a change in `test/score` is a change in the policy rather than a different draw. `test/distinct_starts` is logged precisely because `test/length/std == 0` is ambiguous — it means either a degenerate eval or a policy saturating the episode cap, which are opposite news.

## Ordering gotchas (env vars vs `jax.config`)

The GPU/CPU settings have to be applied at different moments, and getting this wrong fails *silently*. `train.py` handles each in the right place:

| Setting | Mechanism | Why |
|---|---|---|
| `JAX_PLATFORMS` (from `device=` on the CLI) | `os.environ`, **before `import jax`** | Parsed once at `import jax`. Setting the env var afterwards is ignored. |
| `agent.device` (from the composed config) | `jax.config.update("jax_platforms", ...)` | JAX is already imported by the time the config is composed, so this *must* go through `jax.config`. `gpu` forces nothing — it lets JAX pick the accelerator, so a missing card is a banner line rather than an import error. |
| `XLA_PYTHON_CLIENT_MEM_FRACTION`, `XLA_PYTHON_CLIENT_ALLOCATOR` | `os.environ`, after import, before the first `jax.*` call | The PJRT C++ client reads these when it lazily initializes. |
| `XLA_FLAGS=--xla_gpu_autotune_level=0` | `os.environ` | Required on Blackwell (RTX 5080): XLA's autotuning phase hangs indefinitely compiling MJX kernels. |
| `runtime.matmul_precision` | `jax.config.update` | Global — reaches agent networks *and* MJX physics. |

Two of these deserve elaboration.

**Backend env vars are decided from the composed config, not from `sys.argv`.** `env.impl: warp` set inside an experiment YAML never appears in argv, so an argv sniff misses it — and JAX then preallocates its default 75% of VRAM and the Warp reset constants OOM. The `impl == "warp"` branch also switches XLA to CUDA's async allocator (`XLA_PYTHON_CLIENT_ALLOCATOR=cuda_async`): the BFC allocator caps out on *fragmentation*, not on a leak, and PPO's rollout-length growth produces enough alloc/free churn to trigger it (observed: death at epoch 131 requesting a contiguous 2.02 GiB with GPU memory flat for the whole run). Both are `setdefault`, so the environment still wins.

**`matmul_precision` is worth setting, and the default is the wrong guess.** On this RTX 5080, JAX's default f32 matmul resolves to the **bf16** path — which is both slower *and* less accurate than TF32:

| `jax_default_matmul_precision` | ms/grad step | rel. err vs f32 |
|---|---|---|
| `float32` / `highest` | 1.87 | — (reference) |
| `tensorfloat32` | **1.26** | 4.1e-5 |
| unset (JAX default) | 2.14 | 6.5e-5 |
| `bfloat16` | 2.10 | 6.5e-5 |

So `tensorfloat32` is a strict improvement: ~1.7× faster *and* closer to the f32 reference. Neither "default = fastest" nor "lower precision = faster" holds here. Because it is global, flip it for a whole sweep at once — never for a single arm, or the comparison stops being an A/B on the algorithm.

## Keeping the backends honest

Two implementations of the same task will drift. The release benchmark is built so that the drift is *measured* rather than assumed: all 25 dm_control tasks run on both `warp_gpu` (mujoco_playground's JAX reimplementation) and `envpool_cpu` (dm_control's own C++ physics), at a matched budget, on a shared 0–1000 return scale. **A score gap between the two cells on a task is a finding about one of the implementations.** That is the check, and it is the benchmark itself rather than a script.

What "same task" requires, and what to look at when a pair disagrees:

- **The reward has to be the same function.** dm_control rewards are normalised tolerances in [0, 1], so both sides landing in the same range is necessary but nowhere near sufficient — compare curves, not ceilings.
- **The actuation mode has to match.** Torque motors versus PD position servos is a *different task*, not a detail.
- **Termination/truncation semantics have to match**, or one side bootstraps where the other does not (see [Truncation is not termination](#truncation-is-not-termination)).
- **Observations need not be byte-identical, and here they are not.** mujoco_playground omits some of dm_control's observation groups on a few tasks: `HumanoidRun` is 67-dim against EnvPool's 95, `FingerSpin` 9 against 12; most of the 25 match exactly. Both sides are self-consistent, so each cell trains and evaluates against its own spec and the score comparison stays meaningful. The consequence to remember is that **a policy trained on one cell cannot be loaded against the other** — which is why the weights export publishes one cell and names it.

`roxie.environment.suites` is the single place the two naming schemes are reconciled (`BallInCup` ↔ `BallInCupCatch-v1`), and [`tests/test_benchmark_suite.py`](../tests/test_benchmark_suite.py) fails if a task exists on one side and not the other.

## Choosing a backend

- **Have a GPU, want maximum throughput on a task that fits in VRAM** → `backend: warp_gpu`, `parallel_envs` in the low thousands. Note the "low thousands": on small bodies at a few hundred envs the card is not saturated and the CPU pool is *faster* (measured on the dm_control suite at 256 envs — 13.7k sps on `envpool_cpu` against 6.6k on `warp_gpu`). The GPU wins on env count, not env size.
- **Have a GPU but need a large off-policy replay buffer, or the contact arena won't fit** → CPU physics; the buffer moves to system RAM.
- **Want the card free** (shared machine, another job, or the display is on the same GPU) → `backend: envpool_cpu`, which declares `agent.device: cpu` alongside `env.device: cpu`.
- **No GPU at all** → `backend: envpool_cpu`, or `backend: mjx_cpu` to keep the very same JAX program and only move the device.
- **Debugging, or you need bit-reproducibility** → CPU/MJX.
- **Learner-dominated agent on CPU physics with a spare GPU** → `backend: envpool_cpu` with `agent.device: gpu` (leave `env.device: cpu`), plus `trainer.async_learner: true`.
