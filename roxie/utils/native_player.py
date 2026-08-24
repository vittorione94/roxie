"""Generic native-MuJoCo (CPU) playback stepper.

Envs in this repo are written in MJX — a GPU-first, batch-oriented rewrite of
MuJoCo. A single world on CPU is impractically slow there, so for interactive
playback (watch a checkpoint while the GPU is busy training) we step the physics
with native ``mujoco.mj_step`` instead.

This module is env-agnostic: ``NativePlayer`` owns only the native ``MjData`` and
the step orchestration. Everything env-specific — how to seed a fresh episode,
how to turn state into an observation, how to map an action onto ctrl, how to
score/terminate a step — is delegated to a small ``native_*`` protocol the env
implements (see ``NativeSteppable``). Which env (and therefore which native
implementation) is used is decided entirely by the Hydra ``env.builder`` config;
nothing here knows about any particular task.

``reset(key)`` / ``step(state, action)`` mirror the jitted env's signatures and
return an object shaped like the ``mjx_env.State`` that path produces
(``obs``/``data``/``reward``/``done``/``metrics``/``info``), so ``play.py``
drives this through the exact same loop it uses for the MJX path.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable

import mujoco
import numpy as np


@runtime_checkable
class NativeSteppable(Protocol):
    """The hooks an env implements to be playable on native CPU MuJoCo.

    ``info`` is an opaque per-episode dict owned by the env (phase counters,
    filter state, last action, ...); the player carries it between calls but
    never inspects it. Any keys the viewer needs (e.g. the mocap ghost reads
    ``clip_start``/``phase_idx``) simply ride along inside it.
    """

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
    """Peel adapters (``PlaygroundFuncEnv``, ...) off to the base env.

    An env that implements the ``native_*`` protocol itself (mocap) is already
    the base and comes straight back out.
    """
    base = env
    while hasattr(base, "env"):
        base = base.env
    return base


class NativePlayer:
    """Steps a ``NativeSteppable`` env on native CPU MuJoCo for playback."""

    def __init__(self, env: Any):
        base = _unwrap(env)
        self._env = base
        self._model = base.mj_model
        self._data = mujoco.MjData(self._model)
        self._n_substeps = int(base.native_n_substeps)
        self._info: dict = {}

    def reset(self, key: Any):
        self._info = self._env.native_reset(self._data, key)
        obs = self._env.native_obs(self._data, self._info)
        return self._state(obs, 0.0, False, {})

    def step(self, state: Any, action: Any):
        ctrl, self._info = self._env.native_control(action, self._info)
        self._data.ctrl[:] = ctrl
        mujoco.mj_step(self._model, self._data, nstep=self._n_substeps)
        reward, done, metrics, self._info = self._env.native_post(
            self._data, action, ctrl, self._info
        )
        obs = self._env.native_obs(self._data, self._info)
        return self._state(obs, reward, done, metrics)

    def _state(self, obs, reward, done, metrics):
        # Shaped like the `mjx_env.State` the jitted FuncEnv path produces, so
        # `play.py` reads `state.obs` / `state.data` / `state.done` without
        # knowing which stepper produced it.
        return SimpleNamespace(
            obs=np.asarray(obs, dtype=np.float32),
            data=self._data,
            reward=float(reward),
            done=bool(done),
            metrics={k: float(v) for k, v in metrics.items()},
            info=self._info,
        )


def make_native_player(env: Any):
    """Return a ``NativePlayer`` if the env supports native stepping, else None.

    ``play.py`` calls this (via the ``env.player`` dotted path, defaulted in
    play.py) to get a fast, GPU-free stepper. Returns None for any env that does
    not implement the ``native_*`` protocol, so the caller falls back to the
    generic jitted-MJX path.
    """
    base = _unwrap(env)
    if isinstance(base, NativeSteppable):
        return NativePlayer(env)
    return None
