"""Vectorized environment drivers for JAX and C++ pool backends.

Provides `JaxVectorEnv` (vmapped `FuncEnv`) and `EnvPoolVectorEnv` 
(C++ thread pool), both presenting an identical batched interface for 
JIT-compiled training loops.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct

from roxie.environment import functional


@struct.dataclass
class VecState:
    """The environment-side carry for the next step.

    Attributes:
        env_state: The internal state of the environment. None for C++ pools.
        obs: The current observation to act upon. 
        steps: Step counters for truncation limits. None for C++ pools.
    """
    env_state: Any
    obs: jax.Array
    steps: Any = None


class Timestep(NamedTuple):
    """The pre-reset transition outcome. Unpacks as 
    `(obs, reward, terminated, truncated, info)`."""
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    truncated: jax.Array
    info: dict


def nonfinite_worlds(obs, reward, xp=jnp):
    """Identifies worlds that produced non-finite observations or rewards."""
    n = obs.shape[0]
    finite_obs = xp.all(xp.isfinite(obs).reshape(n, -1), axis=1)
    return ~(finite_obs & xp.isfinite(reward))


def zero_worlds(mask, x, xp=jnp):
    """Selectively zeroes worlds based on a boolean mask."""
    x = xp.asarray(x)
    return xp.where(mask.reshape((-1,) + (1,) * (x.ndim - 1)), 0.0, x)


class JaxVectorEnv:
    """A `FuncEnv` batched across `num_envs` worlds via `jax.vmap`."""

    def __init__(self, func_env, num_envs: int, max_episode_steps: int = 0):
        self.func_env = func_env
        self.num_envs = int(num_envs)
        self.max_episode_steps = int(max_episode_steps)

        self.single_observation_space = func_env.observation_space
        self.single_action_space = func_env.action_space
        self.metadata = dict(getattr(func_env, "metadata", {}) or {})

        self._v_initial = jax.vmap(func_env.initial, in_axes=(0, None))
        self._v_transition = jax.vmap(func_env.transition, in_axes=(0, 0, 0, None))
        self._v_observation = jax.vmap(func_env.observation, in_axes=(0, 0, None))
        self._v_reward = jax.vmap(func_env.reward, in_axes=(0, 0, 0, 0, None))
        self._v_terminal = jax.vmap(func_env.terminal, in_axes=(0, 0, None))
        self._v_truncal = jax.vmap(func_env.truncal, in_axes=(0, 0, None))
        self._v_state_info = jax.vmap(func_env.state_info, in_axes=(0, None))
        self._v_transition_info = jax.vmap(
            func_env.transition_info, in_axes=(0, 0, 0, None)
        )

    def reset(self, key, params=None, num_envs: int | None = None):
        """Initializes fresh episodes for `num_envs` worlds."""
        n = self.num_envs if num_envs is None else int(num_envs)
        keys = jax.random.split(key, n)
        env_state = self._v_initial(keys, params)
        obs = self._v_observation(env_state, keys, params)
        state = VecState(
            env_state=env_state, obs=obs, steps=jnp.zeros(n, dtype=jnp.int32),
        )
        zeros = jnp.zeros(n, dtype=jnp.float32)
        false = jnp.zeros(n, dtype=jnp.bool_)
        return state, Timestep(
            obs=obs, reward=zeros, terminated=false, truncated=false,
            info=self._v_state_info(env_state, params),
        )

    def step(
        self, state: VecState, action, key, reset_pool: VecState | None = None,
        params=None,
    ):
        """Executes one control step and gathers fresh starts for done worlds."""
        step_key, pool_key = jax.random.split(key)
        keys = jax.random.split(step_key, self.num_envs)

        prev_env_state = state.env_state
        next_env_state = self._v_transition(prev_env_state, action, keys, params)
        reward = self._v_reward(prev_env_state, action, next_env_state, keys, params)
        obs = self._v_observation(next_env_state, keys, params)
        info = self._v_transition_info(
            prev_env_state, action, next_env_state, params,
        )

        terminal = self._v_terminal(next_env_state, keys, params)
        truncal = self._v_truncal(next_env_state, keys, params)

        diverged = nonfinite_worlds(obs, reward)
        obs = zero_worlds(diverged, obs)
        reward = zero_worlds(diverged, reward)

        steps = state.steps + 1
        step_truncated = (
            steps >= self.max_episode_steps if self.max_episode_steps > 0
            else jnp.zeros_like(terminal)
        )
        terminated = jnp.logical_or(
            jnp.logical_and(terminal, jnp.logical_not(truncal)), diverged
        )
        truncated = jnp.logical_or(truncal, step_truncated)
        done = jnp.logical_or(jnp.logical_or(terminated, truncal), step_truncated)

        info = {
            **info,
            "metrics": {
                **{
                    key: zero_worlds(diverged, value)
                    for key, value in info.get("metrics", {}).items()
                },
                "nonfinite": diverged.astype(jnp.float32),
            },
        }

        timestep = Timestep(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated,
            info=info,
        )
        stepped = VecState(env_state=next_env_state, obs=obs, steps=steps)
        if reset_pool is None:
            return stepped, timestep
        return self._autoreset(stepped, done, reset_pool, pool_key), timestep

    def _autoreset(self, stepped: VecState, done, reset_pool: VecState, key):
        """Replaces done worlds with randomly gathered starts from the pool."""
        pool_size = reset_pool.obs.shape[0]
        idx = jax.random.randint(key, (self.num_envs,), 0, pool_size)

        def _leaf(pool_leaf, s):
            if not (isinstance(s, jnp.ndarray) and s.shape[:1] == done.shape):
                return s
            return jnp.where(
                done.reshape(done.shape + (1,) * (s.ndim - 1)), pool_leaf[idx], s,
            )

        return jax.tree.map(_leaf, reset_pool, stepped)


def _dict_obs_keys(space: Any) -> tuple[str, ...] | None:
    """Extracts the observation groups declared by a pool `Dict` space."""
    spaces = getattr(space, "spaces", None)
    if not isinstance(spaces, dict):
        return None
    return tuple(spaces)


def _resolve_obs_keys(space: Any, keys: Any = None) -> tuple[str, ...] | None:
    """Resolves which observation groups should be flattened."""
    declared = _dict_obs_keys(space)
    if keys is None:
        return declared
    keys = tuple(keys)
    if declared is None:
        raise ValueError(
            f"observation groups {keys} were requested, but the pool declares "
            f"one flat observation space ({space!r}), not named groups"
        )
    unknown = [key for key in keys if key not in declared]
    if unknown:
        raise ValueError(
            f"the pool declares no observation group(s) {unknown}; it has "
            f"{list(declared)}"
        )
    return keys


def _as_box(space: Any, unbounded: bool = False, keys: Any = None) -> Any:
    """Normalizes a pool's space object to a `gymnasium.spaces.Box`."""
    keys = _resolve_obs_keys(space, keys)
    if keys is not None:
        size = sum(int(np.prod(space.spaces[k].shape)) for k in keys)
        return functional.unbounded_box(size)
    low, high = getattr(space, "low", None), getattr(space, "high", None)
    if low is None or high is None:
        if not unbounded:
            raise ValueError(f"space {space!r} declares no low/high bounds")
        return functional.unbounded_box(int(np.prod(space.shape)))
    return functional.box(low, high, shape=tuple(space.shape))


class EnvPoolVectorEnv:
    """A C++ pool batched internally via EnvPool."""

    metadata = {"jax": False, "impl": "envpool"}
    _POOL_HOOKS = ("epoch_refresh",)

    def __init__(
        self, pool: Any, num_envs: int | None = None,
        max_episode_steps: int = 1000, rebuild=None,
        obs_keys: Any = None,
    ):
        self.max_episode_steps = int(max_episode_steps)
        self._rebuild = rebuild
        self._num_envs = num_envs
        self._bind_pool(pool)

        self._obs_keys = _resolve_obs_keys(pool.observation_space, obs_keys)
        self.single_observation_space = _as_box(
            pool.observation_space, unbounded=True, keys=self._obs_keys,
        )
        self.single_action_space = _as_box(pool.action_space)

    def _bind_pool(self, pool: Any) -> None:
        self._pool = pool
        self.num_envs = int(
            getattr(pool, "num_envs", None) or self._num_envs or 0
        ) or None
        for name in self._POOL_HOOKS:
            attr = getattr(pool, name, None)
            if attr is not None:
                setattr(self, name, attr)

    def reseed(self, seed: int) -> bool:
        """Rebuilds the pool at `seed` to pin reset reproducibility."""
        if self._rebuild is None:
            return False
        self._bind_pool(self._rebuild(seed))
        return True

    def _flat_obs(self, obs: Any) -> np.ndarray:
        """Flattens the pool's observation into a single contiguous array."""
        if self._obs_keys is None:
            return np.asarray(obs, dtype=np.float32)
        parts = []
        for key in self._obs_keys:
            leaf = obs[key] if isinstance(obs, Mapping) else getattr(obs, key)
            leaf = np.asarray(leaf, dtype=np.float32)
            parts.append(leaf.reshape(leaf.shape[0], -1))
        return np.concatenate(parts, axis=1)

    @property
    def step_spec(self):
        """Defines the shape/dtype signature expected by `io_callback`."""
        n = int(self.num_envs)
        obs_size = int(np.prod(self.single_observation_space.shape))
        f32 = lambda shape: jax.ShapeDtypeStruct(shape, np.float32)  # noqa: E731
        return (
            f32((n, obs_size)),
            f32((n,)),
            jax.ShapeDtypeStruct((n,), np.bool_),
            jax.ShapeDtypeStruct((n,), np.bool_),
            {"nonfinite": f32((n,))},
        )

    def reset(self, key=None, params=None, num_envs: int | None = None):
        """Resets the pool."""
        del key, params, num_envs 
        obs, info = self._pool.reset()
        obs = self._flat_obs(obs)
        n = obs.shape[0]
        self.num_envs = n
        false = np.zeros(n, dtype=bool)
        return VecState(env_state=None, obs=obs), Timestep(
            obs=obs, reward=np.zeros(n, dtype=np.float32),
            terminated=false, truncated=false,
            info=info if isinstance(info, dict) else {},
        )

    def step(self, state: VecState, action, key=None, reset_pool=None, params=None):
        """Executes one step in the C++ pool."""
        del state, key, reset_pool, params
        obs, reward, terminated, truncated, info = self._pool.step(
            np.asarray(action)
        )
        obs = self._flat_obs(obs)
        reward = np.asarray(reward, dtype=np.float32)

        diverged = nonfinite_worlds(obs, reward, np)
        obs = zero_worlds(diverged, obs, np)
        reward = zero_worlds(diverged, reward, np)

        info = dict(info) if isinstance(info, dict) else {}
        info["metrics"] = {
            **info.get("metrics", {}),
            "nonfinite": diverged.astype(np.float32),
        }
        timestep = Timestep(
            obs=obs,
            reward=reward,
            terminated=np.asarray(terminated, dtype=bool),
            truncated=np.asarray(truncated, dtype=bool),
            info=info,
        )
        return VecState(env_state=None, obs=obs), timestep