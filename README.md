<p align="center">
<img src="https://github.com/vittorione94/roxie/blob/main/images/roxie_logo.png?raw=true" alt="Roxie" style="width:50%; height:auto;">
</p>

# Roxie

A reinforcement learning framework in JAX for continuous control in MuJoCo. Roxie trains across hundreds to thousands of parallel environments, and it runs the *same task* on three different physics backends — MJX, mujoco_warp, and native CPU MuJoCo — so that a result can be reproduced (and a bottleneck diagnosed) on either a GPU or a many-core CPU box.

The interesting part of this repo is not the algorithms, which are standard; it is that the CPU and GPU paths are deliberately kept semantically identical while being structurally very different. [Backends: CPU vs GPU](#backends-cpu-vs-gpu) is the section to read.

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Backends: CPU vs GPU](#backends-cpu-vs-gpu) — the design centrepiece
- [Configuration](#configuration)
- [Agents](#agents)
- [Mocap tracking example](#mocap-tracking-example)
- [Tests](#tests)

## Installation

Requires Python ≥ 3.11.

```bash
uv sync                 # CPU-only: MJX on CPU + the EnvPool/native-MuJoCo path
uv sync --group cuda    # Linux + NVIDIA: adds jax[cuda12] and warp-lang
```

or, with plain pip:

```bash
pip install -e .
```

The `cuda` group pins `warp-lang>=1.11,<1.13`. This is not conservatism: MuJoCo's vendored Warp bridge imports `warp._src.jax_experimental.ffi.GraphMode` and reads `warp.types.warp_type_to_np_dtype`. Warp 1.13 dropped the latter from the public API and 1.14 graduated `jax_experimental` into `jax`, moving both out from under the bridge. Verify a Warp install with:

```bash
uv run python scripts/check_warp.py     # exit 0 = warp usable
```

## Quick start

### Train

Every run is a self-contained experiment YAML under [`experiments/`](experiments/), grouped by environment (`walker/`, `mocap/`, `envpool/`). The folder is part of the config name:

```bash
uv run python roxie/train.py --config-name walker/walker_sac
uv run python roxie/train.py --config-name mocap/sweep_ppo           # GPU (Warp) physics
uv run python roxie/train.py --config-name mocap/sweep_ppo_envpool   # CPU physics
uv run python roxie/train.py --config-name envpool/halfcheetah_td3   # CPU, gym-style task
```

Any key can be overridden from the command line (Hydra):

```bash
uv run python roxie/train.py --config-name walker/walker_sac \
    env.parallel_envs=400 trainer.save_steps=100_000
```

`device=cpu` (or `device=gpu`) is a special override parsed before JAX is imported, and forces the JAX platform for the whole process regardless of what the config says — see [Ordering gotchas](#ordering-gotchas-env-vars-vs-jaxconfig).

Each run writes to its Hydra output dir: resolved config under `.hydra/`, epoch metrics to console + CSV, checkpoints under `checkpoints/`, and optionally Weights & Biases (`logging.wandb.enabled: true`).

### Play

```bash
uv run python roxie/play.py --checkpoint-path outputs/<run>/checkpoints/step_500000
```

Playback drives a single world into the interactive MuJoCo viewer. It always forces CPU/MJX, even for a Warp-trained checkpoint — see [Determinism](#determinism-and-reproducibility). Trailing `key=value` args override the saved run config, e.g. `env.config.early_termination=false` to watch a clip run to its end instead of resetting on tracking collapse.

---

## Backends: CPU vs GPU

Roxie can put the *physics* on the GPU or the CPU, and — independently — the *agent* (networks, optimizers, replay buffer) on the GPU or the CPU. Those two choices are what the rest of this section is about.

### The three physics backends

| | `impl: jax` (MJX) | `impl: warp` (mujoco_warp) | `impl: envpool` (native MuJoCo) |
|---|---|---|---|
| Device | GPU (or CPU) | GPU only | CPU only |
| Vectorization | `jax.vmap` over a traced step | `jax.vmap` → Warp kernels | C++/Python thread pool, one `MjData` per env |
| Batch dim lives in | the XLA program | the XLA program | native code; Python sees `(N, ...)` numpy |
| Contact allocation | **static**, sized to *all* potential geom pairs | **budgeted** — `naconmax` / `njmax` / `naccdmax` | **dynamic**, MuJoCo allocates as needed |
| Trainer loop | `Trainer._run_jax` | `Trainer._run_jax` | `Trainer._run_envpool` |
| Bounded by | VRAM | VRAM | system RAM + core count |
| Built by | `roxie.environment.loader.build_playground_env` | `examples.mocap.loader.build_mocap_env` | `roxie.environment.envpool_adapter.build_envpool_env` / `examples.mocap.mocap_envpool.build_mocap_envpool_env` |

Selection is a Hydra config group. For the mocap task, [`experiments/mocap/backend/`](experiments/mocap/backend/) holds `warp.yaml` and `envpool.yaml`, and a launchable picks one in its `defaults:`. Everything backend-mechanical (builder, `impl`, solver budgets, graph mode, thread count) lives in that group; per-experiment tuning stays in the launchable. That is what makes [`mocap/sweep_ppo`](experiments/mocap/sweep_ppo.yaml) and [`mocap/sweep_ppo_envpool`](experiments/mocap/sweep_ppo_envpool.yaml) a genuine A/B: they differ in exactly one line.

`train.py` prints a loud banner at startup reporting the backend that *actually* loaded (read off the constructed env, not echoed from config) and flags a mismatch against what was requested.

### Why there are two trainer loops

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

### Auto-reset: the same trick, for two different reasons

The GPU path uses a reset pool to avoid host synchronization. The CPU mirror ended up using one too, for an unrelated reason: an on-demand reset pays a full `mj_forward` purely to build the observation, and an untrained policy on the mocap task resets ~20% of envs *per step* (mean episode length ≈ 4). Precomputing a pool turns a reset into a memcpy. This was the single biggest CPU throughput win — and, incidentally, a parity fix, since it is what `_run_jax` had always done.

Auto-reset is same-step on both backends (EnvPool convention): the observation returned on a done step is the *reset* observation. The JAX trainer acts from the auto-reset state, so the entry following a `done` in the trajectory buffer is the reset obs on both sides, and nothing bootstraps off the true final observation. It differs in exactly one place — on the JAX path the true final obs feeds the obs-normalization statistics, on the CPU path the reset obs does.

### Truncation is not termination

Shared by both paths, and worth stating because it is easy to get wrong. `TerminationWrapper` separates:

- **termination** — a genuine failure (NaN, tracking collapse, root drift). Zeroes the bootstrap in the Bellman target.
- **truncation** — a time-out (episode step limit, or clip end). Must *still* bootstrap the next-state value; marking it terminal collapses Q at the cutoff.

Both are surfaced in `info` and `done` is their union so the trainer still auto-resets. The env's own internal truncation is pulled back *out* of `done` before the termination flag is computed. Negative mining observes `info["termination"]`, never `done`, so clip-end and step-limit cutoffs are not mined for as if they were failures.

### Memory: the actual reason to run on CPU

This is the trade the CPU backend exists to make: **the GPU path is bounded by a 16 GB card; the CPU path is bounded by 64 GB of system RAM.** On the GPU, the replay buffer, the contact arena, the reference clips and the activations all compete for the same VRAM. On CPU, `parallel_envs` and replay capacity stop competing with the physics for memory.

Two GPU-side allocation traps are worth knowing, because both were expensive to find:

- **Buffer *capacity* is not update size.** PPO's `max_length_time_axis` is queue capacity, and `update()` drains every step, so occupancy oscillates between 0 and `sample_sequence_length`. A capacity of 512 against a sample length of 32 reserved 2.31 GB of VRAM for a queue that never exceeds 32 entries.
- **Donate, or pay double.** `replay.add` called eagerly copies the whole queue every env step; jitting it with `donate_argnums=(0,)` took `agent.add` from 43 ms to 1.8 ms. The same applies to the fused `_grad_steps`: without donation, XLA allocates a full second copy of the read-only replay buffer on every update (~1.4 GB), and the same trap applies to the warmup replay fill, where the transient 2× of the buffer's observation store OOMs the pool outright.

Also: what `nvidia-smi` reports is XLA's **preallocated arena** (~75% of the card by default), not resident data. To see real usage, read `jax.local_devices()[0].memory_stats()["bytes_in_use"]` and `["peak_bytes_in_use"]`.

The important negative result: **RAM is not a throughput lever.** The CPU pool costs ~1.6 MB/env, so 5000 envs is ~8.7 GB of a 64 GB box — but steady-state throughput is essentially flat in `parallel_envs` (measured: 25.1k sps @ 1000 → 27.8k @ 5000). Raising `parallel_envs` on CPU is a batch-size / gradient-quality decision, not a speed one. The one place the extra RAM genuinely pays is an **off-policy replay buffer living in system RAM instead of VRAM** — which PPO, being on-policy with no replay, cannot use at all.

### Contact budgets: three backends, three models

Collision handling is where the backends diverge most sharply, and it drives configuration on both the memory *and* the task-definition axis.

- **Native MuJoCo (CPU)** allocates contacts dynamically. There is nothing to size. This is why `backend/envpool.yaml` simply omits the Warp budget keys.
- **Warp (GPU)** uses a **single global contact arena shared across all vmapped worlds** (rows tagged by `contact__worldid`, so these Data leaves have no per-env batch dimension — which is exactly why the trainer's auto-reset gather skips leaves lacking a leading per-env dim). `naconmax` therefore scales with `parallel_envs`, and it must cover *broadphase AABB-overlap candidate pairs* (~48/world peak here), not just the ~15 contacts that actually resolve. `njmax` is per-world. A third budget, `naccdmax` (GJK/EPA convex narrowphase), defaults to the full `naconmax` and reserves ~90 MB of EPA scratch per buffer; `build_mocap_env` caps it at `parallel_envs*4` because only two geoms on this humanoid take the convex path at all. Watch for "broadphase/narrowphase/nefc overflow" warnings.
- **MJX (`impl: jax`)** ignores `naconmax` entirely and statically sizes contact arrays to *all potential geom pairs* — ~980 for this humanoid. Self-collisions are therefore roughly **75× heavier on MJX than on Warp**, which is why `self_collisions=false` is the right default there and not elsewhere.

Note that the collision setting is not purely a performance knob: it changes the task. The mocap sweep runs `collisions: ground` because the retargeted CMU reference interpenetrates its own limbs, so under full self-collision the solver pushes the body out of the very pose the tracking reward is asking it to hold.

### Warp's CUDA graph mode

`mjx.put_model(impl='warp')` defaults to `GraphMode.WARP`, whose graph-capture cache is keyed on per-step buffer *addresses*. Under JAX those addresses change every step, so a new CUDA graph is captured per step; the cache is bounded but eviction only drops the Python reference — the native host descriptors are never reclaimed. The result is ~0.25 GB of host-RAM growth per 1M steps, invisible to `jax.live_arrays()` and to GPU memory counters, which OOM-kills long runs (typically during a checkpoint save, since that adds a transient spike).

Roxie therefore defaults Warp to **`graph_mode: WARP_STAGED_EX`**: capture once onto fixed staging buffers, then replay every step plus a cheap device→staging memcpy. The obvious-looking `JAX`/`NONE` modes also avoid the leak but are much slower, because they launch this step's many small kernels eagerly instead of replaying a graph. The recapture was the problem, not graph replay.

### Where the agent runs (independent of where the physics runs)

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

### Async learner (CPU physics + GPU learner only)

In the hybrid configuration the loop is otherwise fully serial: step the CPU physics, then run the GPU gradient burst, each device idle while the other works. For agents where the learner dominates (D4PG's distributional critic, say), that caps throughput well below the CPU physics ceiling.

`trainer.async_learner: true` moves gradient updates onto a background thread that is the **sole owner** of `agent.state` (networks, optimizers, replay buffer, obs stats). The acting thread never touches it: it selects actions from a behaviour-actor snapshot the learner publishes, steps the envs, and pushes transitions through a queue. That single-owner rule is not stylistic — the fused `_grad_steps` donates the whole train state including the replay buffer, and donation is only sound because nothing else references it concurrently. Eval and checkpointing bracket themselves with `pause()`/`resume()` so they observe a quiescent state.

Replay ratio is preserved (the learner runs one burst per `steps_between_updates` boundary crossed, so it can never run ahead of collected data); the behaviour policy lags by up to one burst, which is standard for async off-policy RL. `learner_chunk` trades interleaving granularity against per-dispatch overhead.

This is an `envpool`-loop-only option, and it is pointless when the agent is also on CPU — there are no two devices to overlap.

### CPU parallelism: Amdahl, not cores

The CPU mocap pool runs one `MjData` per env stepped by a persistent `ThreadPoolExecutor`; MuJoCo's Python bindings release the GIL inside `mj_step`, so threads scale across cores without pickling. `num_threads: null` means `min(parallel_envs, cpu_cores)`, and on the 7900X 24 threads beats 16 once env count is well above core count, because `mj_step` releases the GIL and the SMT siblings absorb memory stalls. Below ~100 envs the extra threads only add dispatch overhead.

The physics itself saturates the machine. What caps average utilization is the **single-threaded numpy between physics passes**, and the rule that emerged is entirely about *operation size*:

- **Large numpy ops thread; thousands of tiny per-env assignments do not.** Splitting observation assembly across 6 workers by env range cut it from 22.8 ms to 10.6 ms (bit-identical output) because it is large ufuncs and concatenates that release the GIL. Conversely, moving the per-env `MjData` → buffer field copies into worker threads made things *worse* — 24 threads just queue on the GIL, and physics-only throughput fell from 40k to 29k sps. Those stay on the main thread, env-major, into preallocated buffers, gathered once per step.
- **Defer GIL-held Python into the worker.** Pooled reset went from 23.1 ms to 2.1 ms by deferring the per-env `mj_resetData` + qpos/qvel restore into the worker thread behind a `_needs_reset` flag. The numpy gathers were only 0.84 ms of that; the other 22 ms was ~13 µs of Python per env. (Consequence: the single-env `_get_obs`/`_get_reward` adapters are no longer valid against a live pool, since they read `_datas` directly.)

Serial fraction went from 33.6% to 18.6% of wall-clock at 5000 envs. The remainder is the `MjData`→buffer capture, ~30k small assignments, which is provably not threadable in Python; eliminating it needs `mujoco.rollout` (the C++ batched threaded API), which returns only state + sensordata — so body positions and frames would have to come from added `framepos`/`framequat` sensors. That is the next real step, and a redesign.

### Determinism and reproducibility

Warp's atomic contact reductions are **not bit-reproducible**: two identical rollouts diverge by ~1e-6 per step, which the chaotic contact dynamics then amplify. CPU/MJX playback is exact. This is why:

- `play.py` unconditionally forces `JAX_PLATFORMS=cpu` and coerces `impl: warp` → `impl: jax` for playback. Checkpoints are agent-side, so the policy replays identically on either backend, and Warp-only knobs are simply ignored by the MJX builder.
- The mocap sweep uses `test_episodes: 4` rather than 1 on a single-clip task — not for start-state diversity (the eval protocol pins frame 0 with no noise, so all four are the same rollout) but so that Warp's non-reproducibility shows up as visible spread.

Evaluation itself is pinned: fixed reset keys held constant across epochs *and* across runs, so a change in `test/score` is a change in the policy rather than a different draw. `test/distinct_starts` is logged precisely because `test/length/std == 0` is ambiguous — it means either a degenerate eval or a policy saturating the episode cap, which are opposite news.

### Ordering gotchas (env vars vs `jax.config`)

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

### Keeping the backends honest

Two backends implementing the same task will drift, and this one did: the CPU mirror silently fell ~6 features behind until the observation vectors were literally different lengths (1057 vs 1069). [`examples/mocap/check_envpool_parity.py`](examples/mocap/check_envpool_parity.py) is the executable statement of the contract — **extend it whenever either side gains an observation or reward term.** Current agreement is 7.8e-07 on observations and <1e-5 on reward components.

The drifts it now guards against are a good checklist of what "same task" actually requires: identical actuation mode (torque motors vs PD position servos is a *different task*, not a detail), identical observation field order, identical reward decomposition and weights, identical reset filter state, identical negative-mining start distribution, and identical termination/truncation semantics with matching metric keys.

### Choosing a backend

- **Have a GPU, want maximum throughput on a task that fits in VRAM** → `backend: warp`, `parallel_envs` in the low thousands.
- **Have a GPU but need a large off-policy replay buffer, or the contact arena won't fit** → CPU physics; the buffer moves to system RAM.
- **Want the card free** (shared machine, another job, or the display is on the same GPU) → `backend: envpool` with its default `runtime.jax_platform: cpu`.
- **No GPU at all** → `backend: envpool`, or `impl: jax` with `device=cpu` for smaller MJX tasks.
- **Debugging, or you need bit-reproducibility** → CPU/MJX.
- **Learner-dominated agent on CPU physics with a spare GPU** → `runtime.jax_platform: null` plus `trainer.async_learner: true`.

---

## Configuration

Hydra, with a strict convention: **the agent YAML *is* the constructor call.** `_target_` names the class and every sibling key is one of its keyword arguments — there is no `name:`/`args:` indirection and no registry. A knob that exists in Python but is missing from the YAML fails loudly at launch instead of silently taking its default. This is enforced by [`tests/test_agent_configs.py`](tests/test_agent_configs.py) and [`tests/test_agent_construction.py`](tests/test_agent_construction.py).

Layout:

- [`roxie/configs/agent/`](roxie/configs/agent/) — one file per algorithm (`ddpg`, `td3`, `td4`, `d4pg`, `sac`, `mpo`, `ppo`, plus baselines).
- [`roxie/configs/noise/`](roxie/configs/noise/) — exploration noise modules (`ou`, `gaussian`, `composite`, `adaptive`). Agents that explore from their own policy (SAC, MPO, PPO) take **no** noise group; adding one to them is a launch-time error.
- [`experiments/<env>/`](experiments/) — launchable experiments. `--config-name` includes the folder.
- [`experiments/mocap/`](experiments/mocap/) — additionally carries its own config groups: `agent/` (per-task tunings), `reward/`, `env_config/`, `backend/`, `sweep/` (shared blocks).

Only four arguments are injected by `train.py` rather than coming from YAML — `env_obs_size`, `env_action_size`, `action_low`, `action_high` — because only the env knows them. Nested `*_config` blocks stay unresolved (`_recursive_=False`) so each agent instantiates its own actor/critic/memory/optimizers, injecting shapes the trainer cannot know.

The env factory itself is a dotted path (`env.builder`, resolved with `hydra.utils.get_method`), so the core training loop never branches on an env-type string and never names "mocap". Same for the viewer hook (`env.viewer`).

A minimal experiment:

```yaml
# @package _global_

defaults:
  - /agent: sac
  - _self_

env:
  seed: 0
  parallel_envs: 800
  builder: roxie.environment.loader.build_playground_env
  env_name: WalkerWalk
  impl: warp

runtime:
  matmul_precision: null

trainer:
  steps: 1_000_000_000
  epoch_steps: 100_000
  save_steps: 500_000
  test_episodes: 5
  show_progress: true
  replace_checkpoint: false

hydra:
  run:
    dir: outputs/${now:%Y-%m-%d}/${now:%H-%M-%S}
```

## Agents

| Agent | Type | Policy | Critic | Returns | Exploration |
|---|---|---|---|---|---|
| **DDPG** | Off-policy | Deterministic | Q(s,a) | n-step | Noise module |
| **TD3** | Off-policy | Deterministic | Twin Q(s,a), delayed policy | n-step | Noise module |
| **D4PG** | Off-policy | Deterministic | Categorical Q(s,a) | n-step | Noise module, no target smoothing |
| **TD4** | Off-policy | Deterministic | Twin categorical Q(s,a), delayed policy | n-step | Noise module |
| **SAC** | Off-policy | Stochastic (squashed Gaussian) | Twin Q(s,a) | n-step | Entropy, auto α |
| **MPO** | Off-policy | Stochastic (Gaussian) | Q(s,a) | **1-step only** | Policy sampling + KL duals |
| **PPO** | On-policy | Stochastic (Gaussian) | V(s) | GAE | Policy entropy |

All share `roxie.agents.agent.Agent`, which owns observation normalization, checkpointing, and the `step`/`add`/`update` interface the trainer consumes. Baselines (`Constant`, `NormalRandom`, `UniformRandom`, `OrnsteinUhlenbeck`) are available for sanity-checking an environment.

The off-policy agents pick their buffer from `n_step`: a flashbax **flat buffer** at `n_step: 1`, switching automatically to a **trajectory buffer** (`sample_sequence_length = n_step + 1`, `period=1`) when `n_step > 1`, since n-step targets need consecutive items. The YAML keeps the flat-buffer schema either way — `max_length`/`min_length` are *total* transitions, converted internally to flashbax's per-row time-axis lengths. PPO instead uses a trajectory **queue**, drained every update.

Two caveats when comparing them: **PPO is not replay-ratio comparable** (on-policy — judge it on score-vs-env-steps and score-vs-wall-clock, not gradient steps), and **MPO is ~20× more expensive per gradient step** (20 action samples per state at batch 512). MPO runs 1-step returns because `mpo.py` takes no `n_step`; that is an implementation gap, not a chosen handicap.

## Mocap tracking example

A humanoid motion-capture tracking task on dm_control's CMU Humanoid (V2020), living under [`examples/mocap/`](examples/mocap/) rather than in the core package — the dependency direction is one-way, examples import from `roxie` and never the reverse.

Clips are retargeted CMU data fetched from DeepMind's public HDF5 and cached in `~/.cache/roxie`; no manual conversion step is needed. Each clip is grounded by shifting it down until the lowest foot *collision surface* (not the geom origin — the feet are 25 mm-radius capsules, and grounding on origins buries them) rests on the floor.

The env computes a weighted reward from pose matching, joint velocities, end-effector positions, and split root position/orientation/velocity terms, with early termination on NaN, tracking collapse and root drift. Rotations are exchanged with the network in the **6D continuous representation** (Zhou et al.), never quaternions or Euler angles.

Two features shape the training distribution:

- **Negative mining over start phases.** Uniform random starts spend most of their budget on clip regions already tracked well. The env keeps a per-bin failure *rate* over the clip (a rate, not a count — dying early means later phases are visited less, and a count would mistake that for competence), EMA-smoothed, and biases reset toward the failing bins as a mixture against uniform (`alpha`, kept well below 1: this is a re-weighting, not a curriculum). Watch `mining/effective_bins` — collapse toward 1 means coverage is being lost.
- **GPU clip residency.** `gpu_clip_budget` caps how many clips are resident on the GPU at once, reshuffled per epoch (`clip_swap`). A real swap invalidates in-progress episodes, whose stored clip indices reference the old chunk, so the trainer resets live envs only then. The CPU backend has no such budget — the full dataset always lives in host RAM.

The canonical eval protocol is fixed and deliberate: **start at frame 0, no reset noise, run the clip to its end.** That is the task as stated ("track this clip"), not a sample of it, and `play.py` starts at frame 0 too — so what you watch is what the metric measured.

See [`experiments/mocap/sweep/README.md`](experiments/mocap/sweep/README.md) for the single-clip overfit sweep that compares every agent on identical settings, and the helper scripts alongside the env: `check_envpool_parity.py`, `check_mocap_reward.py`, `check_openloop_tracking.py`, `check_cmu_mocap_data.py`.

## Tests

```bash
uv run pytest tests/
```

Covering models (actor/critic shapes and bounds), loss functions, exploration noise, n-step returns, agent utilities, per-agent behaviour, and — importantly — that every agent config constructs and that no `__init__` keyword is missing from its YAML.
