# CLAUDE.md

Project-specific context for Claude Code when working in this repo.

## Reference Docs

Fetch these on demand when working on related code — don't assume their content, check the live page:

- **JAX** — https://docs.jax.dev/en/latest/ — core dependency (`jax[cuda12]`).
- **flashbax** — https://instadeepai.github.io/flashbax/ — replay buffer backend; only `roxie/utils/memory.py`'s `ReplayManager` should call it directly (see JAX Conventions above).
- **EnvPool** — https://envpool.readthedocs.io/en/latest/ — CPU physics backend, wrapping dm_control's own C++ implementation (not MJX/Warp) behind a gym/dm_env `reset`/`step` API with a native thread pool; only `roxie/environment/loader.py`'s `build_envpool_env` and `roxie/environment/vector.py`'s `EnvPoolVectorEnv` should call it directly. Its pool never exposes `mjModel`/`mjData`, and its `step()` is closed-loop (one call per timestep) — bridged into the fused training `jax.lax.scan` via an `ordered=True` `jax.experimental.io_callback` in `roxie/utils/rollout.py`'s `pool_advance`, which `EnvPoolRollout` binds as its `advance`.
  - Its [XLA interface](https://envpool.readthedocs.io/en/latest/content/xla_interface.html) (`env.xla()`) was built out, measured and **reverted** (2026-09-17) — don't re-attempt without new evidence. The callback cost is per STEP, so it amortizes over `parallel_envs` and is already negligible at benchmark width; the gain only becomes visible at low env counts. It also needs process-global `jax_enable_x64` (every dm_control task declares its action spec float64 in C++ and `send` does no numpy cast), which `mujoco_playground`'s warp FFI cannot tolerate, and it SIGSEGVs on the eval path because `reseed` rebuilds the pool while EnvPool names its FFI target after `id(pool)` and `evaluate` is jitted on a static `self`.
  - Batching the per-step callback — a host-side Python step loop, or one bulk `io_callback` running all T steps — was prototyped and measured, and **does not pay** (2026-09-17). `scripts/profile_envpool_chunk.py` is the measurement: swap only `EnvPoolRollout.advance` and subtract the runs. The callback splits into a FIXED per-crossing cost (~90us) and a host<->device TRANSFER cost (~1.5us per env per step), and only the fixed part is what batching removes — a bulk callback moves the same bytes. T is `steps_between_updates // parallel_envs`, so at the benchmark's 1024 envs T is **2** and the removable cost is 0.18ms of a 71ms iteration (0.3%); at 64 envs 3.6%, at 16 envs 11.2%. Both prototypes then gave that back in the extra per-step dispatches the fused chunk does not pay, netting out at or below the current path. `ordered=False` is a wash too (identical wall time to three figures over 20 chunks), and `ordered=True` must stay regardless: `roll_random`'s actions do not depend on the previous observation, so nothing but the token sequences its callbacks.
  - A `jax.profiler` trace of the same cell confirms XLA's compute lanes are **100% idle for the whole of every callback** — 11 of 12 `tf_XLAEigen` lanes do literally nothing while the 12th runs it. That is NOT reclaimable capacity at benchmark width: `/proc/self/stat` over the same window puts EnvPool's own C++ threads at **10.7 of 12 cores** during exactly that stall (`runtime.cpu_cores` and `env.num_threads` are pinned equal for this reason), so the idle lanes are idle because the cores are taken. At 16 envs the pool only fills 5.6 of 12 and ~19% of the machine really is dead — but the way to spend it is overlapping the replay write behind the next physics step, not batching the crossings, and it is worth nothing at the 1024 envs the grid runs. Two trace caveats: profiling inflates the `python` lane ~6x (a chunk reads 126ms against 21.9ms unprofiled), so only the trace's STRUCTURE is usable, and the ordered-token machinery it makes look expensive (`_add_tokens_to_inputs` -> `device_put` -> `block_until_ready`) is exactly that artifact. EnvPool's threads are not XLA-instrumented and never appear in the trace at all, which is why the occupancy question needs `/proc`.

- **rlax** — https://rlax.readthedocs.io/en/latest/index.html — RL primitives (`td_learning`, `categorical_td_learning`, GAE); used throughout `roxie/losses/` and `roxie/agents/ppo.py`, typically vmapped over the batch/env axis (see JAX Conventions above).
- **MuJoCo** — https://mujoco.readthedocs.io/en/stable/overview.html — physics backend (`mujoco-mjx`); this repo targets MJX, mujoco_warp, and native-CPU MuJoCo across environments (see `pyproject.toml`, `scripts/check_warp.py`, `roxie/utils/native_player.py`).
- **Brax** — https://github.com/google/brax — not a dependency here; useful as a reference JAX physics/RL implementation when comparing approaches.
- **Acme** — https://github.com/google-deepmind/acme — not a dependency here; useful as a reference RL agent architecture (DeepMind) when comparing agent/loop design.
- **Google Python Style Guide** — https://google.github.io/styleguide/pyguide.html — section 3.8 is the docstring and comment contract this repo follows (see Docstrings and Comments below). Check it rather than guessing at section spelling or indentation.

## No Legacy

**roxie is unreleased. Nothing in it is legacy, and nothing needs backward
compatibility.** There are no users on an older version, no checkpoints or run
logs in the wild, and no config spellings to keep honouring.

So when something is renamed, moved or collapsed into another knob, the old form
goes — no alias, no fallback branch, no "still stripped so an old config
replays", no shim that rewrites yesterday's column names. Update the code, the
yamls, the docs and the tests together, and delete the old path in the same
change.

The word "legacy" should not appear in this repo. If it turns up in a comment,
it is either dead code to delete or a live feature described badly: the flat
pair vs trajectory replay layouts, for instance, are both current — `n_step=1`
uses one and `n_step>1` the other.

Compatibility that is NOT legacy, and stays: handling a real difference between
agents or backends (an agent without `obs_stats`, a pool that exposes no
`mjModel`), and refusing a mismatch loudly (`checkpoint.py`'s observation-width
and buffer-geometry checks). That is present-tense variation, not history.

## Docstrings and Comments

**Docstrings are Google-style**, per section 3.8 of the style guide linked
above. That is a rule about STRUCTURE, not about how much you are allowed to
know: see "what does not get deleted" below before touching an existing one.

### The shape

- A one-line summary, one physical line, under 80 characters, ending in a
  period. Descriptive mood (`"""Advances the envs by one step."""`), not
  imperative — except a property or a helper that simply IS a value, which
  takes a noun phrase (`"""The observation the next action is selected
  from."""`), as the guide's `@property` rule has it.
- Then a blank line, then any extended description.
- Then `Args:` / `Returns:` / `Raises:`, in that order, with a **four-space**
  hanging indent and continuation lines indented two further. (The guide allows
  two or four and asks only for consistency; this repo is four.
  `roxie/agents/utils.py`'s `fused_grad_steps` is the one block still on two.)

  ```
  Args:
      state: the pytree to carry. A tuple of them works too and is how an
        agent's side modules keep updating across the fused steps.
      n_steps: static; it is the scan length.

  Returns:
      `(train_state, noise_module, rstate, sums)` — the chunk's carry, and
      the scalars the epoch metrics are built from.
  ```

- **Omit those sections** when the name and the signature already say it and a
  one-line docstring covers the function. The guide says so explicitly, and
  most of this repo's helpers are that case — do not grow `finished_episodes`
  an `Args:` block to satisfy a checklist. `Returns:` is also omitted when the
  summary itself starts with "Returns"/"Yields" and says enough.
- Classes get a one-line summary of what an instance REPRESENTS; public
  attributes go in an `Attributes:` section formatted like `Args:`. A
  `struct.dataclass`'s fields belong there (`ChunkSums`, `RolloutState`).
- Types stay in the signature's annotations, not repeated in the docstring.
- Single backticks for code, never RST-style double backticks.
  `roxie/environment/functional.py` and `roxie/environment/vector.py` are the
  two files still using doubles; convert them when you next touch them.

### What does NOT get deleted

The same guide: *"The final place to have comments is in tricky parts of the
code. If you're going to have to explain it at the next code review, you should
comment it now."* Going Google-style **relocates** this repo's reasoning out of
docstring prose and down to the line it explains — it does not license dropping
it. Anything measured, anything that burned someone, anything that reads as an
arbitrary choice until you know why, stays:

- Why `io_callback` must be `ordered=True` on the EnvPool path.
- Why eval takes a fixed `PRNGKey`, and why `reset_test` reseeds.
- Every `donate_argnums` justification (already required under JAX Conventions).
- Why the chunk folds observation statistics once instead of per step.
- Why `unroll` is clamped to `n_steps`.
- The numbers and dates on anything already tried and reverted.

Put it where it bites, as a block comment above the operation or an end-of-line
comment on the non-obvious one. What leaves the docstring is only the part that
was never contract: the essay framing, the ALL-CAPS banner headings, the
second-person asides.

**Never describe the code.** A comment that restates the line above it is the
one kind that should be deleted outright rather than moved.

### Module docstrings

These keep their orienting job — a module's docstring is how someone finds
their way into `rollout.py` or `functional.py` at all, and Google style has no
rule against a long one. Drop the ALL-CAPS banner headings for plain
paragraphs, and lead with a one-line summary like any other docstring.

## JAX Conventions

This is a JAX-based RL codebase (`roxie/agents/`: PPO, SAC, MPO, TD3, DDPG, D4PG, TD4; losses in `roxie/losses/`; shared JAX utilities in `roxie/utils/`). These are the project's own conventions on top of JAX — follow them rather than reaching for raw JAX primitives.

### PRNG keys
- Variable name is always `key` (or a descriptive suffix like `agent_rng`, `step_key`, `sample_key`, `noise_key`, `perm_key`) — never `rng` or `seed` at call sites.
- `flax.nnx.Rngs` is the general-purpose key-splitting utility across the codebase, not just for module init. `Trainer` holds separate `envs` and `agent` streams (`roxie/utils/trainer.py`), seeded so that resuming a run offsets the `envs` stream but keeps `agent` fixed (see `roxie/train.py`).
- Network param-init keys go through `network_rngs(seed, offset)` in `roxie/agents/utils.py`, with the `offset` convention: actor=0, critic=2, second critic=4 (so twin critics never init identically).
- Bulk key-splitting for scanned learning passes goes through `fused_grad_steps` (`roxie/agents/utils.py`) — it splits once via `jax.random.split(key, n_steps)` and scans. Off-policy agents' `_<agent>_grad_steps` should call this rather than splitting manually per step.
- Within a single step, split with an explicit count and destructure immediately, e.g. `key, sample_key, actor_key, critic_key = jax.random.split(key, 4)`.
- Eval/test rollouts intentionally use a **fixed, non-threaded** key (`jax.random.PRNGKey(0)`, or `PRNGKey(_EVAL_SEED=12345)` for eval env resets in `roxie/utils/rollout.py`) so consecutive evals of the same weights are comparable. Don't thread a fresh key through eval paths.

### jit structure
- Every agent's fused gradient-step entry point is wrapped with `functools.partial(jax.jit, static_argnames=(...), donate_argnums=(0,))`. `hp` (the frozen hyperparams dataclass) is always static; the `TrainState` is always donated. SAC donates additional argnums since it threads `log_alpha_module`/`alpha_optimizer` alongside state.
- Functions that run per-step but not per-pass (e.g. PPO's `_prepare_rollout`) are `jax.jit` but explicitly **not** donated.
- Loss functions are never decorated with `@jax.jit` — they're always called from inside an already-jitted `_<agent>_grad_steps`, so decorating them would just add a nested `pjit`.
- `jax.lax.scan` is the standard tool for "run N steps as one compiled program": fused learning passes, PPO's epoch/minibatch double-scan with early stop via `jax.lax.cond`, the rollout's acting/env-step loop, and warmup random rollout. `jax.lax.while_loop` is used specifically for variable-length episodes in eval, keeping the termination check on-device.
- Per-step conditions that vary across a `lax.scan` (e.g. TD3's `policy_delay`, PPO's early-stop `stopped`) must be threaded as **traced** booleans through `jax.lax.cond`/`nnx.cond`, not Python `if` — the step index is no longer a Python int inside the scan.
- `donate_argnums`/`donate_argnames` is used pervasively; always add a comment justifying it (avoiding transient 2x buffer allocation) when you add a new one.

### Pytrees
- Two mechanisms coexist by design — don't mix them up: `flax.struct.dataclass` for plain data (`Transition`, `ObsStats`, `RolloutState`), vs. `flax.nnx.Module` for anything with live parameters/optimizer state (`TrainState`, `LogAlpha`). `TrainState` is deliberately a pytree rather than an opaque nnx graph node, specifically so it can be passed straight to `jit`/`lax.scan` without `nnx.split`/`nnx.merge`.
- Optional fields on pytree containers are left `None` rather than zero-filled when unused (e.g. `Transition.log_probs=None` for off-policy agents) — relies on `None` being an empty pytree node.
- No custom `jax.tree_util.register_pytree_node` calls anywhere. Any new container that crosses a jit boundary should go through `flax.struct.dataclass` or `nnx.Module`, never a raw dataclass or NamedTuple.
- Hyperparameter containers are always plain `@dataclasses.dataclass(frozen=True)` (not `struct.dataclass`) so they stay hashable for `static_argnames`. The standard parameter name for a static hyperparams object is `hp`.

### vmap
- Used to lift per-transition rlax loss primitives (`rlax.td_learning`, `rlax.categorical_td_learning`) over the batch dimension — always with a comment noting the primitive is defined per scalar transition.
- PPO vmaps `rlax.truncated_generalized_advantage_estimation` over the env axis with explicit `in_axes`.
- The vectorized env layer (`roxie/environment/vector.py`) vmaps each `FuncEnv` method individually with per-argument `in_axes`, rather than vmapping the whole env object at once — the ensemble-of-envs equivalent of ensemble critics.

### Prefer these helpers over raw JAX
- `roxie/agents/utils.py`: `fused_grad_steps` (scan-wrapped learning passes), `soft_update` (Polyak averaging via `nnx.update`/`optax.incremental_update` — never hand-roll this), `reduce_diagnostics`, `network_rngs`.
- `roxie/utils/math.py`: `finite_or_zero` is the only sanctioned NaN-scrub primitive, applied at three documented boundaries (actor output, obs entering running stats, buffer writes). `normalize_obs`/`obs_mean_std`/`update_obs_stats` for the `ObsStats` running-moments pytree. `normalize_samples` is the single place replay samples get normalized — losses never take `obs_mean`/`obs_std` directly.
- `roxie/utils/memory.py`: `ReplayManager` wraps all flashbax buffer interaction; agents should never call `flashbax` directly except inside `build_replay`.

### Device / sharding
- Single-device only — no `pmap`, no `jax.sharding`/`NamedSharding` anywhere. `roxie/utils/checkpoint.py` explicitly restores to host memory rather than any device sharding baked into a checkpoint.

### Float width (`roxie/utils/precision.py`)
- **Pin the dtype at every array-creation site.** `jnp.zeros(...)`, `jax.random.normal(...)`, `jnp.asarray(<numpy>)` and friends take their dtype from JAX's process-global canonicalization when it is left off, not from the surrounding code. Pin to `precision.FLOAT` (float32), or to a neighbouring array's `.dtype` when the point is to match it.
- The repo is deliberately x64-CLEAN even though nothing enables `jax_enable_x64`: the pinning is what keeps an f64 from reaching an f32 replay buffer and firing a flashbax/chex assertion frames away from wherever the width came from.
- Watch third-party promotions, which is where this actually bites. `distrax`'s `entropy()` returns float64 even for float32 params (pinned in `TanhNormal.entropy`), and integer division in `roxie/exploration/schedulers.py` did the same to the noise scale and with it the action. `roxie/utils/checkpoint.py` restores float leaves at the RUN's width, not the checkpoint file's — a resume must not change what the run computes in.

### Debugging / NaN handling
- No `jax.debug.print` or `checkify` usage. NaN handling goes through the `finite_or_zero` scrub pattern (deterministic clamping), not runtime assertions. Plain `print(...)` is reserved for host-side, pre/post-jit status messages (compile timing, agent init) — not for in-jit debugging.
