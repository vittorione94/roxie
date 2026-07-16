"""EnvPool adapter: wraps an envpool pool to match the roxie env interface.

EnvPool provides C++-vectorized environments for CPU training. This module
bridges envpool's stateful, auto-resetting pool API to the duck-typed interface
the roxie Trainer and agents expect (WrapperState / env_state protocol).

Usage via experiment YAML:
    env:
      builder: roxie.environment.envpool_adapter.build_envpool_env
      task_id: HalfCheetah-v4
      parallel_envs: 32
      test_episodes: 5

The Trainer detects EnvPoolWrapper and routes to its CPU training loop
(_run_envpool), which skips JAX vmap/jit for the env step and delegates
auto-reset to EnvPool's built-in C++ logic.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import jax.numpy as jnp

from roxie.environment.loader import EnvBundle


class EnvPoolEnvState:
    """Duck-type stand-in for mjx_env.State in the EnvPool code path.

    The Trainer and agents access: obs, reward, done, info (with
    "termination"/"truncation" keys), and metrics.
    """

    __slots__ = ("obs", "reward", "done", "info", "metrics")

    def __init__(
        self,
        obs: jnp.ndarray,
        reward: jnp.ndarray,
        done: jnp.ndarray,
        info: dict,
        metrics: dict | None = None,
    ):
        self.obs = obs
        self.reward = reward
        self.done = done
        self.info = info
        self.metrics = metrics if metrics is not None else {}


class EnvPoolWrapperState:
    """Duck-type stand-in for WrapperState in the EnvPool code path."""

    __slots__ = ("env_state",)

    def __init__(self, env_state: EnvPoolEnvState):
        self.env_state = env_state


class EnvPoolWrapper:
    """Adapts an envpool gymnasium pool to the roxie env interface.

    EnvPool handles batching and auto-reset entirely in C++, so no
    jax.vmap or jax.jit wrapping of reset/step is needed. The Trainer
    detects this class and uses a simpler Python-loop training path.

    Args:
        pool: An envpool pool created with env_type="gymnasium".
        max_episode_steps: Episode time-limit used for eval rollouts.
    """

    _backend: str = "envpool"
    # Reported by loader.resolve_loaded_impl (the startup banner): pools have
    # no MJX backend, the "physics impl" is the pool itself.
    _impl: str = "envpool"

    def __init__(self, pool: Any, max_episode_steps: int = 1000):
        self._pool = pool
        self.max_episode_steps = max_episode_steps

        obs_space = pool.observation_space
        act_space = pool.action_space

        self.observation_size: int = int(np.prod(obs_space.shape))
        self.action_size: int = int(np.prod(act_space.shape))
        self.action_low: jnp.ndarray = jnp.array(act_space.low, dtype=jnp.float32)
        self.action_high: jnp.ndarray = jnp.array(act_space.high, dtype=jnp.float32)

    def reset(self) -> EnvPoolWrapperState:
        obs, _ = self._pool.reset()
        n = obs.shape[0]
        false_n = jnp.zeros(n, dtype=jnp.bool_)
        return EnvPoolWrapperState(
            EnvPoolEnvState(
                obs=jnp.array(obs, dtype=jnp.float32),
                reward=jnp.zeros(n, dtype=jnp.float32),
                done=false_n,
                info={"termination": false_n, "truncation": false_n},
            )
        )

    def step(self, state: EnvPoolWrapperState, action: Any) -> EnvPoolWrapperState:
        obs, reward, terminated, truncated, info = self._pool.step(np.asarray(action))
        done = np.logical_or(terminated, truncated)
        # Pools may report per-step env metrics (e.g. the mocap pool's reward
        # components) under info["metrics"]; surface them on the state so the
        # trainer logs them exactly like the JAX path does. Kept as numpy —
        # the trainer reduces them host-side.
        metrics = info.get("metrics", {}) if isinstance(info, dict) else {}
        return EnvPoolWrapperState(
            EnvPoolEnvState(
                obs=jnp.array(obs, dtype=jnp.float32),
                reward=jnp.array(reward, dtype=jnp.float32),
                done=jnp.array(done, dtype=jnp.bool_),
                info={
                    "termination": jnp.array(terminated, dtype=jnp.bool_),
                    "truncation": jnp.array(truncated, dtype=jnp.bool_),
                },
                metrics=metrics,
            )
        )


def build_envpool_env(cfg_env: Any, mode: str = "train") -> EnvBundle:
    """Builder for envpool environments (CPU training path).

    Reads from cfg_env:
      task_id (str):            envpool task identifier, e.g. "HalfCheetah-v4".
      parallel_envs (int):      number of parallel training environments.
      seed (int):               RNG seed (default 0).
      max_episode_steps (int):  episode time-limit (default 1000).
      test_episodes (int):      eval pool size; should match trainer.test_episodes.
      (any other keys are forwarded to envpool.make() as keyword arguments)

    Two independent pools are created — one for training, one for evaluation —
    so eval rollouts never corrupt the training environment state.
    """
    import envpool  # lazy: envpool is an optional dependency

    task_id: str = cfg_env.task_id
    num_envs: int = int(cfg_env.parallel_envs)
    seed: int = int(cfg_env.get("seed", 0))
    max_episode_steps: int = int(cfg_env.get("max_episode_steps", 1000))
    test_episodes: int = int(cfg_env.get("test_episodes", 5))

    _reserved = frozenset({
        "task_id", "parallel_envs", "seed", "max_episode_steps",
        "test_episodes", "builder",
    })
    extra = {k: v for k, v in cfg_env.items() if k not in _reserved}

    train_pool = envpool.make(
        task_id,
        env_type="gymnasium",
        num_envs=num_envs,
        seed=seed,
        max_episode_steps=max_episode_steps,
        **extra,
    )
    test_pool = envpool.make(
        task_id,
        env_type="gymnasium",
        num_envs=test_episodes,
        seed=seed + 1,
        max_episode_steps=max_episode_steps,
        **extra,
    )

    train_env = EnvPoolWrapper(train_pool, max_episode_steps=max_episode_steps)
    test_env = EnvPoolWrapper(test_pool, max_episode_steps=max_episode_steps)

    return EnvBundle(env=train_env, test_env=test_env, env_cfg=None)
