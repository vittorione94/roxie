# Backends: CPU vs GPU

Roxie can put the *physics* on the GPU or the CPU, and — independently — the *agent* (networks, optimizers, replay buffer) on the GPU or the CPU. Those two choices are what the rest of this document is about.

*Back to the [README](../README.md).*

## The three physics backends

| | `impl: jax` (MJX) | `impl: warp` (mujoco_warp) | `impl: envpool` (native MuJoCo) |
|---|---|---|---|
| Device | GPU (or CPU) | GPU only | CPU only |
| Vectorization | `jax.vmap` over a traced step | `jax.vmap` → Warp kernels | C++/Python thread pool, one `MjData` per env |
| Batch dim lives in | the XLA program | the XLA program | native code; Python sees `(N, ...)` numpy |
| Contact allocation | **static**, sized to *all* potential geom pairs | **budgeted** — `naconmax` / `njmax` / `naccdmax` | **dynamic**, MuJoCo allocates as needed |
| Trainer loop | `Trainer._run_jax` | `Trainer._run_jax` | `Trainer._run_envpool` |
| Bounded by | VRAM | VRAM | system RAM + core count |
| Built by | `roxie.environment.loader.build_playground_env` | `examples.mocap.loader.build_mocap_env` | `roxie.environment.envpool_adapter.build_envpool_env` / `examples.mocap.mocap_envpool.build_mocap_envpool_env` |

Selection is a Hydra config group. For the mocap task, [`experiments/mocap/backend/`](../experiments/mocap/backend/) holds `warp.yaml` and `envpool.yaml`, and a launchable picks one in its `defaults:`. Everything backend-mechanical (builder, `impl`, solver budgets, graph mode, thread count) lives in that group; per-experiment tuning stays in the launchable. That is what makes [`mocap/sweep_ppo`](../experiments/mocap/sweep_ppo.yaml) and [`mocap/sweep_ppo_envpool`](../experiments/mocap/sweep_ppo_envpool.yaml) a genuine A/B: they differ in exactly one line.

`train.py` prints a loud banner at startup reporting the backend that *actually* loaded (read off the constructed env, not echoed from config) and flags a mismatch against what was requested.

## Why there are two trainer loops

`Trainer.run()` dispatches on the env type: `EnvPoolWrapper` → `_run_envpool`, everything else → `_run_jax`. They are not gratuitously duplicated; the two loops obey opposite constraints.

**`_run_jax` must never touch the host.** Every quantity that a Python `if` would want to branch on — episode-done flags, buffer occupancy, episode returns — lives in a device array, and reading one forces a blocking device→host sync that serializes the whole asynchronous GPU pipeline. So the JAX loop is written to keep the host out of the way:

- **Auto-reset is branchless.** Instead of resetting done envs on demand, the loop pre-builds a *reset pool* of `parallel_envs` fresh states and, each step, gathers a random pool entry per env and `jnp.where`s it in against the done mask. No control flow, no sync, one fused kernel. The pool is regenerated once per epoch.
- **Statistics accumulate on-device.** Per-episode return/length means *and standard deviations* are computed from running sums and sums-of-squares reduced only at the epoch boundary, rather than keeping a host-side list of episodes (which would sync every step).
- **Evaluation is one compiled rollout.** `_make_eval_fn` puts the entire episode loop inside `jax.lax.while_loop` so the `~all(done)` termination check is evaluated on-device, with a static `max_steps` cap to bound compute. A Python `while` here would drag a device array back per step.
- **Agents gate their own updates.** The trainer calls `agent.update(steps=...)` unconditionally and the agent decides internally whether to run gradient steps. The trainer is explicitly forbidden from reading buffer device state (e.g. a flashbax `can_sample`) to make that decision.
- **Everything is precompiled up front**, with timings printed: reset, reset pool, train step, agent step, replay add, warmup rollout, replay fill, gradient step. A cold compile mid-run is a stall.

**`_run_envpool` has none of those constraints and should not pretend to.** The pool steps in C++, hands back numpy, and Python branching is free. So it is a plain loop with numpy accumulators and a Python `for` eval loop. Trying to force the JAX idioms here would only add dispatch overhead.

What the two loops *must* agree on is semantics, and that agreement is maintained by hand:

- both update observation-normalization statistics from the live policy's state distribution every step (not just during warmup);
- both log the same keys, including `test/distinct_starts`, `test/score_per_step`, the env's own `metrics` dict, per-agent diagnostics (`pop_diagnostics`), negative-mining health, GPU telemetry and the host-memory watch;
- both refresh the negative-mining table and *then* rebuild the reset pool, in that order, at the epoch boundary.

The split of labour for negative mining is the clearest illustration of the structural difference. A JAX env cannot own mutable state inside a trace, so the trainer threads `mining_weights` through `reset` as a **traced argument** — precisely so that refreshing it each epoch does not retrigger compilation. A CPU pool resets in plain Python and simply owns its own table, so the trainer only has to tell it *when* to refresh. Same algorithm, opposite ownership.

## Auto-reset: the same trick, for two different reasons

The GPU path uses a reset pool to avoid host synchronization. The CPU mirror ended up using one too, for an unrelated reason: an on-demand reset pays a full `mj_forward` purely to build the observation, and an untrained policy on the mocap task resets ~20% of envs *per step* (mean episode length ≈ 4). Precomputing a pool turns a reset into a memcpy. This was the single biggest CPU throughput win — and, incidentally, a parity fix, since it is what `_run_jax` had always done.

Auto-reset is same-step on both backends (EnvPool convention): the observation returned on a done step is the *reset* observation. The JAX trainer acts from the auto-reset state, so the entry following a `done` in the trajectory buffer is the reset obs on both sides, and nothing bootstraps off the true final observation. It differs in exactly one place — on the JAX path the true final obs feeds the obs-normalization statistics, on the CPU path the reset obs does.

## Truncation is not termination

Shared by both paths, and worth stating because it is easy to get wrong. `TerminationWrapper` separates:

- **termination** — a genuine failure (NaN, tracking collapse, root drift). Zeroes the bootstrap in the Bellman target.
- **truncation** — a time-out (episode step limit, or clip end). Must *still* bootstrap the next-state value; marking it terminal collapses Q at the cutoff.

Both are surfaced in `info` and `done` is their union so the trainer still auto-resets. The env's own internal truncation is pulled back *out* of `done` before the termination flag is computed. Negative mining observes `info["termination"]`, never `done`, so clip-end and step-limit cutoffs are not mined for as if they were failures.

## Memory: the actual reason to run on CPU

This is the trade the CPU backend exists to make: **the GPU path is bounded by a 16 GB card; the CPU path is bounded by 64 GB of system RAM.** On the GPU, the replay buffer, the contact arena, the reference clips and the activations all compete for the same VRAM. On CPU, `parallel_envs` and replay capacity stop competing with the physics for memory.

Two GPU-side allocation traps are worth knowing, because both were expensive to find:

- **Buffer *capacity* is not update size.** PPO's `max_length_time_axis` is queue capacity, and `update()` drains every step, so occupancy oscillates between 0 and `sample_sequence_length`. A capacity of 512 against a sample length of 32 reserved 2.31 GB of VRAM for a queue that never exceeds 32 entries.
- **Donate, or pay double.** `replay.add` called eagerly copies the whole queue every env step; jitting it with `donate_argnums=(0,)` took `agent.add` from 43 ms to 1.8 ms. The same applies to the fused `_grad_steps`: without donation, XLA allocates a full second copy of the read-only replay buffer on every update (~1.4 GB), and the same trap applies to the warmup replay fill, where the transient 2× of the buffer's observation store OOMs the pool outright.

Also: what `nvidia-smi` reports is XLA's **preallocated arena** (~75% of the card by default), not resident data. To see real usage, read `jax.local_devices()[0].memory_stats()["bytes_in_use"]` and `["peak_bytes_in_use"]`.

The important negative result: **RAM is not a throughput lever.** The CPU pool costs ~1.6 MB/env, so 5000 envs is ~8.7 GB of a 64 GB box — but steady-state throughput is essentially flat in `parallel_envs` (measured: 25.1k sps @ 1000 → 27.8k @ 5000). Raising `parallel_envs` on CPU is a batch-size / gradient-quality decision, not a speed one. The one place the extra RAM genuinely pays is an **off-policy replay buffer living in system RAM instead of VRAM** — which PPO, being on-policy with no replay, cannot use at all.

## Contact budgets: three backends, three models

Collision handling is where the backends diverge most sharply, and it drives configuration on both the memory *and* the task-definition axis.

- **Native MuJoCo (CPU)** allocates contacts dynamically. There is nothing to size. This is why `backend/envpool.yaml` simply omits the Warp budget keys.
- **Warp (GPU)** uses a **single global contact arena shared across all vmapped worlds** (rows tagged by `contact__worldid`, so these Data leaves have no per-env batch dimension — which is exactly why the trainer's auto-reset gather skips leaves lacking a leading per-env dim). `naconmax` therefore scales with `parallel_envs`, and it must cover *broadphase AABB-overlap candidate pairs* (~48/world peak here), not just the ~15 contacts that actually resolve. `njmax` is per-world. A third budget, `naccdmax` (GJK/EPA convex narrowphase), defaults to the full `naconmax` and reserves ~90 MB of EPA scratch per buffer; `build_mocap_env` caps it at `parallel_envs*4` because only two geoms on this humanoid take the convex path at all. Watch for "broadphase/narrowphase/nefc overflow" warnings.
- **MJX (`impl: jax`)** ignores `naconmax` entirely and statically sizes contact arrays to *all potential geom pairs* — ~980 for this humanoid. Self-collisions are therefore roughly **75× heavier on MJX than on Warp**, which is why `self_collisions=false` is the right default there and not elsewhere.

Note that the collision setting is not purely a performance knob: it changes the task. The mocap sweep runs `collisions: ground` because the retargeted CMU reference interpenetrates its own limbs, so under full self-collision the solver pushes the body out of the very pose the tracking reward is asking it to hold.

## Warp's CUDA graph mode

`mjx.put_model(impl='warp')` defaults to `GraphMode.WARP`, whose graph-capture cache is keyed on per-step buffer *addresses*. Under JAX those addresses change every step, so a new CUDA graph is captured per step; the cache is bounded but eviction only drops the Python reference — the native host descriptors are never reclaimed. The result is ~0.25 GB of host-RAM growth per 1M steps, invisible to `jax.live_arrays()` and to GPU memory counters, which OOM-kills long runs (typically during a checkpoint save, since that adds a transient spike).

Roxie therefore defaults Warp to **`graph_mode: WARP_STAGED_EX`**: capture once onto fixed staging buffers, then replay every step plus a cheap device→staging memcpy. The obvious-looking `JAX`/`NONE` modes also avoid the leak but are much slower, because they launch this step's many small kernels eagerly instead of replaying a graph. The recapture was the problem, not graph replay.

## Where the agent runs (independent of where the physics runs)

With CPU physics there are two viable configurations, and the choice is exposed as `runtime.jax_platform`:

| | Physics | Agent | Notes |
|---|---|---|---|
| `runtime.jax_platform: cpu` | CPU pool | CPU (JAX) | **Default for `backend: envpool`.** The card is genuinely free. |
| `runtime.jax_platform: null` | CPU pool | GPU (JAX) | Faster, but puts ~1.4 GB back on the card and re-couples the run to it. |

The default matters more than it looks. The EnvPool pool's physics never touches the GPU, but the agent is plain JAX and would otherwise still claim the card — and preallocate most of it — so "running on CPU" would leave the GPU fully occupied, which is the opposite of the point. **The CPU backend is meant to free the card entirely**, so `backend/envpool.yaml` pins `jax_platform: cpu`.

Measured on a 12-core/24-thread 7900X, PPO, `parallel_envs=1000`, obs 1069, nets `[1024, 512, 256]`:

| Configuration | Throughput |
|---|---|
| CPU physics + GPU learner | ~17.2k sps |
| Fully GPU-free | ~13.6k sps |
| Physics-only ceiling (no learner) | ~38–40k sps |

Going GPU-free costs ~20%, and that gap is entirely the dense actor/critic GEMMs — which is what the GPU is for. If you want that 20% back and can spare the VRAM, set `runtime.jax_platform: null`.

## Async learner (CPU physics + GPU learner only)

In the hybrid configuration the loop is otherwise fully serial: step the CPU physics, then run the GPU gradient burst, each device idle while the other works. For agents where the learner dominates (D4PG's distributional critic, say), that caps throughput well below the CPU physics ceiling.

`trainer.async_learner: true` moves gradient updates onto a background thread that is the **sole owner** of `agent.state` (networks, optimizers, replay buffer, obs stats). The acting thread never touches it: it selects actions from a behaviour-actor snapshot the learner publishes, steps the envs, and pushes transitions through a queue. That single-owner rule is not stylistic — the fused `_grad_steps` donates the whole train state including the replay buffer, and donation is only sound because nothing else references it concurrently. Eval and checkpointing bracket themselves with `pause()`/`resume()` so they observe a quiescent state.

Replay ratio is preserved (the learner runs one burst per `steps_between_updates` boundary crossed, so it can never run ahead of collected data); the behaviour policy lags by up to one burst, which is standard for async off-policy RL. `learner_chunk` trades interleaving granularity against per-dispatch overhead.

This is an `envpool`-loop-only option, and it is pointless when the agent is also on CPU — there are no two devices to overlap.

## CPU parallelism: Amdahl, not cores

The CPU mocap pool runs one `MjData` per env stepped by a persistent `ThreadPoolExecutor`; MuJoCo's Python bindings release the GIL inside `mj_step`, so threads scale across cores without pickling. `num_threads: null` means `min(parallel_envs, cpu_cores)`, and on the 7900X 24 threads beats 16 once env count is well above core count, because `mj_step` releases the GIL and the SMT siblings absorb memory stalls. Below ~100 envs the extra threads only add dispatch overhead.

The physics itself saturates the machine. What caps average utilization is the **single-threaded numpy between physics passes**, and the rule that emerged is entirely about *operation size*:

- **Large numpy ops thread; thousands of tiny per-env assignments do not.** Splitting observation assembly across 6 workers by env range cut it from 22.8 ms to 10.6 ms (bit-identical output) because it is large ufuncs and concatenates that release the GIL. Conversely, moving the per-env `MjData` → buffer field copies into worker threads made things *worse* — 24 threads just queue on the GIL, and physics-only throughput fell from 40k to 29k sps. Those stay on the main thread, env-major, into preallocated buffers, gathered once per step.
- **Defer GIL-held Python into the worker.** Pooled reset went from 23.1 ms to 2.1 ms by deferring the per-env `mj_resetData` + qpos/qvel restore into the worker thread behind a `_needs_reset` flag. The numpy gathers were only 0.84 ms of that; the other 22 ms was ~13 µs of Python per env. (Consequence: the single-env `_get_obs`/`_get_reward` adapters are no longer valid against a live pool, since they read `_datas` directly.)

Serial fraction went from 33.6% to 18.6% of wall-clock at 5000 envs. The remainder is the `MjData`→buffer capture, ~30k small assignments, which is provably not threadable in Python; eliminating it needs `mujoco.rollout` (the C++ batched threaded API), which returns only state + sensordata — so body positions and frames would have to come from added `framepos`/`framequat` sensors. That is the next real step, and a redesign.

## Determinism and reproducibility

Warp's atomic contact reductions are **not bit-reproducible**: two identical rollouts diverge by ~1e-6 per step, which the chaotic contact dynamics then amplify. CPU/MJX playback is exact. This is why:

- `play.py` unconditionally forces `JAX_PLATFORMS=cpu` and coerces `impl: warp` → `impl: jax` for playback. Checkpoints are agent-side, so the policy replays identically on either backend, and Warp-only knobs are simply ignored by the MJX builder.
- The mocap sweep uses `test_episodes: 4` rather than 1 on a single-clip task — not for start-state diversity (the eval protocol pins frame 0 with no noise, so all four are the same rollout) but so that Warp's non-reproducibility shows up as visible spread.

Evaluation itself is pinned: fixed reset keys held constant across epochs *and* across runs, so a change in `test/score` is a change in the policy rather than a different draw. `test/distinct_starts` is logged precisely because `test/length/std == 0` is ambiguous — it means either a degenerate eval or a policy saturating the episode cap, which are opposite news.

## Ordering gotchas (env vars vs `jax.config`)

The GPU/CPU settings have to be applied at different moments, and getting this wrong fails *silently*. `train.py` handles each in the right place:

| Setting | Mechanism | Why |
|---|---|---|
| `JAX_PLATFORMS` (from `device=` on the CLI) | `os.environ`, **before `import jax`** | Parsed once at `import jax`. Setting the env var afterwards is ignored. |
| `runtime.jax_platform` (from the composed config) | `jax.config.update("jax_platforms", ...)` | JAX is already imported by the time the config is composed, so this *must* go through `jax.config`. |
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

Two backends implementing the same task will drift, and this one did: the CPU mirror silently fell ~6 features behind until the observation vectors were literally different lengths (1057 vs 1069). [`examples/mocap/check_envpool_parity.py`](../examples/mocap/check_envpool_parity.py) is the executable statement of the contract — **extend it whenever either side gains an observation or reward term.** Current agreement is 7.8e-07 on observations and <1e-5 on reward components.

The drifts it now guards against are a good checklist of what "same task" actually requires: identical actuation mode (torque motors vs PD position servos is a *different task*, not a detail), identical observation field order, identical reward decomposition and weights, identical reset filter state, identical negative-mining start distribution, and identical termination/truncation semantics with matching metric keys.

## Choosing a backend

- **Have a GPU, want maximum throughput on a task that fits in VRAM** → `backend: warp`, `parallel_envs` in the low thousands.
- **Have a GPU but need a large off-policy replay buffer, or the contact arena won't fit** → CPU physics; the buffer moves to system RAM.
- **Want the card free** (shared machine, another job, or the display is on the same GPU) → `backend: envpool` with its default `runtime.jax_platform: cpu`.
- **No GPU at all** → `backend: envpool`, or `impl: jax` with `device=cpu` for smaller MJX tasks.
- **Debugging, or you need bit-reproducibility** → CPU/MJX.
- **Learner-dominated agent on CPU physics with a spare GPU** → `runtime.jax_platform: null` plus `trainer.async_learner: true`.
