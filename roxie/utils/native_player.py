"""Native MuJoCo CPU execution engine for interactive playback."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np


@runtime_checkable
class NativeSteppable(Protocol):
    """Protocol defining native MuJoCo stepping hooks for environment playback."""

    @property
    def mj_model(self) -> mujoco.MjModel: ...

    @property
    def native_n_substeps(self) -> int: ...

    def native_reset(self, data: mujoco.MjData, key: Any) -> dict: ...

    def native_obs(self, data: mujoco.MjData, info: dict) -> np.ndarray: ...

    def native_control(self, action: Any, info: dict) -> tuple[np.ndarray, dict]: ...

    def native_post(
        self, data: mujoco.MjData, action: Any, ctrl: np.ndarray, info: dict
    ) -> tuple[float, bool, dict, dict]: ...


def _unwrap(env: Any) -> Any:
    """Unwraps nested environment decorators to reach the underlying base environment."""
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


class NativePlayer:
    """Executes single-world environment steps using native MuJoCo C++ physics bindings."""

    def __init__(self, env: Any):
        base = _unwrap(env)
        self._env = base
        self._model = base.mj_model
        self._data = mujoco.MjData(self._model)
        self._n_substeps = int(base.native_n_substeps)
        self._info: dict = {}

    def reset(self, key: Any):
        """Resets the native MuJoCo physics state and returns initial state namespace."""
        self._info = self._env.native_reset(self._data, key)
        obs = self._env.native_obs(self._data, self._info)
        return self._state(obs, 0.0, False, {})

    def step(self, state: Any, action: Any):
        """Steps physics simulation forward and extracts observation/reward payload."""
        ctrl, self._info = self._env.native_control(action, self._info)
        self._data.ctrl[:] = ctrl
        mujoco.mj_step(self._model, self._data, nstep=self._n_substeps)
        reward, done, metrics, self._info = self._env.native_post(
            self._data, action, ctrl, self._info
        )
        obs = self._env.native_obs(self._data, self._info)
        return self._state(obs, reward, done, metrics)

    def _state(self, obs, reward, done, metrics):
        """Constructs a compatible state namespace mirroring functional JAX state."""
        return SimpleNamespace(
            obs=np.asarray(obs, dtype=np.float32),
            data=self._data,
            reward=float(reward),
            done=bool(done),
            metrics={k: float(v) for k, v in metrics.items()},
            info=self._info,
        )


def make_native_player(env: Any):
    """Instantiates a NativePlayer instance if the underlying environment implements NativeSteppable."""
    base = _unwrap(env)
    if isinstance(base, NativeSteppable):
        return NativePlayer(env)
    return None