# Environments

There are exactly two ways to attach an environment to roxie, both shaped like [Gymnasium's `functional_jax_env`](https://gymnasium.farama.org/main/_modules/gymnasium/envs/functional_jax_env/). Nothing downstream — trainer, learner, agents — knows which backend it is driving.

*Back to the [README](../README.md).*

## The two ways in

| | Write a `FuncEnv` | Bring a batched env |
|---|---|---|
| Defined in | [`roxie/environment/functional.py`](../roxie/environment/functional.py) | [`roxie/environment/vector.py`](../roxie/environment/vector.py) |
| For | anything functional and JAX-traceable | anything already batched |
| State | passed explicitly, one env at a time | carried by the caller, batched |
| Examples | MuJoCo Playground, the mocap env, Waymax | EnvPool, the mocap CPU pool |

Write a `FuncEnv` and `JaxVectorEnv` batches it. Bring something that already speaks Gymnasium's 5-tuple and `EnvPoolVectorEnv` fronts it directly — no translation, because that *is* the interface.

## Writing a `FuncEnv`

Stateless: state goes in, state comes out, nothing accumulates on `self`. That is what lets the driver `vmap` the env across a thousand worlds, `jit` a whole step, and `scan` an entire warmup rollout into one dispatch.

```python
class MyEnv(FuncEnv):
    observation_space = functional.unbounded_box(24)
    action_space = functional.box(-1.0, 1.0, shape=(6,))

    def initial(self, rng, params=None) -> State: ...
    def transition(self, state, action, rng, params=None) -> State: ...
    def observation(self, state, rng, params=None) -> Array: ...
    def reward(self, state, action, next_state, rng, params=None) -> Array: ...
    def terminal(self, state, rng, params=None) -> Array: ...
    def truncal(self, state, rng, params=None) -> Array:      # optional
    def transition_info(self, state, action, next_state, params=None) -> dict:
```

Three things are worth knowing before writing one.

**`terminal` is failure only.** An episode that ends without the outcome being bad — a time limit, a reference trajectory running out — is a *truncation*, and every value-based agent in this repo treats the two differently: a termination zeroes the Bellman bootstrap, a truncation keeps it. Put the env's own non-failure cutoff in `truncal` and let the driver combine them. See [backends.md](backends.md#truncation-is-not-termination) for the exact rule.

**`params` is for values that change between calls but must stay traced.** Because it is an argument rather than a closure constant, refreshing it does not recompile. An env that wants to *adapt* its own `params` while training runs — a start-state curriculum that follows where episodes are failing, say — implements three optional hooks and keeps the whole mechanism to itself:

```python
    def init_params(self) -> params                                # once, at startup
    def observe_params(self, params, info, terminated) -> params   # per step, inside the jit
    def epoch_refresh(self, params) -> (params, invalidated)       # per epoch, on the host
```

The driver only carries the value: it takes one from `init_params`, passes it to every call, threads it through `observe_params` inside the jitted step (so that stays a single dispatch), and refreshes it at the epoch boundary — then rebuilds the auto-reset pool from the refreshed value. `observe_params` is handed the batched `transition_info` dict and `terminated` (failure only, done *minus* truncation). `epoch_refresh` is the one call outside a trace, so it may mutate the env; if that leaves in-progress episodes referring to something that no longer exists, it returns `invalidated=True` and the driver resets the live envs. All three default to no-ops, so an env with no such state ignores them entirely — and roxie contains no trace of what any particular env does with them. A batched pool ([below](#bringing-an-already-batched-env)) has no `params` to thread and instead owns its state outright, exposing a single `epoch_refresh()`.

**Per-step metrics ride in `transition_info` under `"metrics"`.** The trainer means them over an epoch and logs them as `train/<key>`.

### Adapting a monolithic-step env

Playground, Brax and Waymax all compute physics, observation, reward and termination in a single `step` call, so the accessors cannot recompute them independently without paying for the physics twice. The recipe is [`MuJoCoFuncEnv`](../roxie/environment/loader.py): `transition` runs the env's own step, and the accessors read the fields back off the returned state. `PlaygroundFuncEnv` applies it to any `mujoco_playground` env in about ten lines, and `MocapTrackingEnv` inherits the same accessors so the two backends cannot drift on what a termination means.

## Bringing an already-batched env

```python
reset(key, params=None, num_envs=None)          -> (VecState, Timestep)
step(state, action, key, reset_pool, params)    -> (VecState, Timestep)
```

`Timestep` is Gymnasium's `(obs, reward, terminated, truncated, info)` and unpacks as such. It holds the **pre-auto-reset** values — the true next observation, which is what the replay buffer must store — while the returned `VecState.obs` is the **post-auto-reset** observation the next action is selected from. Conflating the two stores the reset observation as the terminal transition's `next_obs`.

The state is threaded through the call rather than kept on the driver, which is the one substantive difference from Gymnasium's `FunctionalJaxVectorEnv` and the reason roxie does not use it. Upstream's implementation is unusable here for three independent reasons, all visible in its source:

1. `step` branches on `if jnp.any(self.prev_done):` — a device-to-host sync **every step**. The mocap release cell runs 1e9 env steps.
2. It resets with `self.state.at[to_reset].set(...)`, which assumes the state is a single array. Roxie's states are pytrees (`mjx.Data`); `.at[]` does not exist on a pytree.
3. Keeping the state on `self` makes a `lax.scan` warmup impossible.

So roxie keeps the shape and writes the driver. Auto-reset is a gather from a pre-built pool rather than Gymnasium's `AutoresetMode.NEXT_STEP`, for the reason given in [backends.md](backends.md#auto-reset-the-same-trick-for-two-different-reasons).

## Builders

A builder turns a config block into an `EnvBundle` of two ready-to-drive vector envs — one for training, one for evaluation. It is named by `env.builder` in the experiment yaml (a dotted path resolved with `hydra.utils.get_method`), so the core loops never name a task:

```python
def build_my_env(cfg_env, mode="train", num_envs=1, test_episodes=1) -> EnvBundle:
```

`num_envs` and `test_episodes` are passed in by the trainer rather than read off `cfg_env`, because they are trainer quantities: the same env definition is driven at `env.parallel_envs` worlds for training and at `trainer.test_episodes` for evaluation. `mode` is `"train"` or `"play"`, letting a builder apply playback-specific tweaks (shrinking a GPU clip pool, falling back from a CPU pool to a single jitted world).

The three shipped builders are `build_playground_env` and `build_envpool_env` in [`roxie/environment/loader.py`](../roxie/environment/loader.py), and `build_mocap_env` / `build_mocap_envpool_env` under [`examples/mocap/`](../examples/mocap/).

## Spaces

Envs declare `gymnasium.spaces.Box` observation and action spaces; drivers expose them as `single_observation_space` / `single_action_space`. `train.py` derives `env_obs_size`, `env_action_size`, `action_low` and `action_high` from them — the four arguments it injects into every agent constructor ([configuration.md](configuration.md)). For a MuJoCo env the action bounds come from `mj_model.actuator_ctrlrange`, so the space is exact rather than a convention.

## Tests

[`tests/test_env_protocol.py`](../tests/test_env_protocol.py) pins the interface against a toy `FuncEnv`: the termination/truncation three-way split, the pre-versus-post-auto-reset observation, that `params` stays traced, and that a whole step is jittable and scannable. Those are the properties a refactor of this boundary is most likely to move silently, and moving either of the first two invalidates every number in the release grid.
