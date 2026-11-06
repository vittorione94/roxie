"""Vectorized environment drivers for the JIT-compiled training loop.

`JaxVectorEnv` vmaps a `FuncEnv`. `BatchVectorEnv` is the same driver with the
physics swapped for a `mujoco.batch` thread pool behind a host callback.
"""

from __future__ import annotations

import contextlib
import os
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import struct
from jax.experimental import io_callback
from mujoco import batch
from mujoco_playground._src import mjx_env

from roxie.utils.precision import FLOAT


@struct.dataclass
class VecState:
    """The environment-side carry for the next step.

    Attributes:
        env_state: The internal state of the environment.
        obs: The current observation to act upon.
        steps: Step counters for truncation limits.
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


# Every field the dm_control suite tasks read off `data`, and the state a world
# needs to be rebuilt from. `suites.DMC_TASKS` is what this list is good for, and
# `build_playground_env` refuses anything else.
_BATCH_CARRIED = (
    "qpos", "qvel", "act", "qacc_warmstart", "mocap_pos", "mocap_quat", "time",
)
_BATCH_DERIVED = (
    "xpos", "xmat", "xipos", "xanchor", "geom_xpos", "geom_xmat", "site_xpos",
    "subtree_com", "sensordata",
)


@contextlib.contextmanager
def _physics_already_stepped(action, fields, n_substeps):
    """Makes a task's `mjx_env.step` return results the batch already computed."""
    real = mjx_env.step

    def step(model, data, task_action, task_substeps=1):
        del model
        if task_action is not action or task_substeps != n_substeps:
            raise ValueError(
                "impl=batch steps physics before the task runs, so the task must "
                "hand mjx_env.step its action and n_substeps unchanged."
            )
        return data.replace(ctrl=action, **fields)

    mjx_env.step = step
    try:
        yield
    finally:
        mjx_env.step = real


def _available_cores() -> int:
    """The cores this process may run on, which a pinned run narrows."""
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count() or 1


class BatchVectorEnv(JaxVectorEnv):
    """A playground task whose physics runs on a `mujoco.batch` thread pool.

    Everything but the physics is `JaxVectorEnv`: resets, auto-reset and the
    upstream task's observation and reward stay in the compiled program.
    """

    def __init__(self, func_env, num_envs: int, max_episode_steps: int = 0,
                 num_threads: int | None = None):
        super().__init__(func_env, num_envs, max_episode_steps)
        nthread = int(num_threads or _available_cores())
        self._batch = batch.Batch(
            func_env.mj_model, nsim=self.num_envs,
            nthread=min(nthread, self.num_envs),
        )
        self._n_substeps = int(func_env.n_substeps)
        self._rows = {
            name: self._batch.bind(name)
            for name in (*_BATCH_CARRIED, *_BATCH_DERIVED, "ctrl")
        }
        self._spec = {
            name: jax.ShapeDtypeStruct(self._field_shape(name), FLOAT)
            for name in (*_BATCH_CARRIED, *_BATCH_DERIVED)
        }
        self._last = {
            name: np.full(self._spec[name].shape, np.nan, FLOAT)
            for name in _BATCH_CARRIED
        }
        self._v_transition = self._transition

    def _field_shape(self, name: str) -> tuple[int, ...]:
        """The shape MJX gives a field, where the batch's rows hold it flat."""
        shape = self._rows[name].shape
        if name == "time":
            return (self.num_envs,)
        return shape[:-1] + (3, 3) if name.endswith("xmat") else shape

    def _step_physics(self, carried, action):
        """Steps every world, reloading those the program moved since last time.

        The rows stay authoritative at float64 between steps. A world's carried
        float32 fields only differ from what was last returned when the program
        replaced them: an auto-reset, a fresh reset, or a task editing `data`.
        """
        stale = np.zeros(self.num_envs, dtype=bool)
        for name, value in carried.items():
            stale |= (np.asarray(value) != self._last[name]).reshape(
                self.num_envs, -1,
            ).any(axis=1)
        ids = np.flatnonzero(stale)
        if ids.size:
            for name, value in carried.items():
                row = self._rows[name]
                row[ids] = np.asarray(value)[ids].reshape(row[ids].shape)

        self._rows["ctrl"][:] = np.asarray(action)
        # mjSTAGE_NONE, the default: derived fields lag the state by a substep,
        # exactly as they do after `mjx.step`, which is what the tasks were
        # written against.
        self._batch.step(nstep=self._n_substeps)

        out = {
            name: self._rows[name].reshape(self._spec[name].shape).astype(FLOAT)
            for name in self._spec
        }
        self._last = {name: out[name] for name in _BATCH_CARRIED}
        return out

    def _transition(self, state, action, keys, params):
        carried = {name: getattr(state.data, name) for name in _BATCH_CARRIED}
        # Ordered: the rows are stateful, so steps must run in sequence with each
        # other and with the loop feeding them.
        fields = io_callback(
            self._step_physics, self._spec, carried, action, ordered=True,
        )

        def one(world, world_action, key, world_fields):
            with _physics_already_stepped(
                world_action, world_fields, self._n_substeps,
            ):
                return self.func_env.transition(world, world_action, key, params)

        return jax.vmap(one)(state, action, keys, fields)
