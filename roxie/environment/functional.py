"""Stateless, JAX-transformable environment interface.

Provides `FuncEnv`, a structurally compatible equivalent to 
`gymnasium.functional.FuncEnv` designed for JAX transformations.
"""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

import jax
import numpy as np
from gymnasium import spaces

StateType = TypeVar("StateType")
ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")
ParamsType = TypeVar("ParamsType")


class FuncEnv(Generic[StateType, ObsType, ActType, ParamsType]):
    """Stateless environment interface.

    Subclasses must set `observation_space` and `action_space` and implement
    at least `initial`, `transition`, `observation`, `reward`, and `terminal`.
    Methods must be written for a single environment and contain no Python
    branching on traced values, as they are vmapped by the driver.
    """

    observation_space: spaces.Space
    action_space: spaces.Space
    metadata: dict[str, Any] = {"jax": True}

    def initial(self, rng: Any, params: ParamsType | None = None) -> StateType:
        """Generates the start state of a fresh episode."""
        raise NotImplementedError

    def transition(
        self, state: StateType, action: ActType, rng: Any,
        params: ParamsType | None = None,
    ) -> StateType:
        """Advances the state by one control step."""
        raise NotImplementedError

    def observation(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> ObsType:
        """Extracts the observation seen by the policy."""
        raise NotImplementedError

    def reward(
        self, state: StateType, action: ActType, next_state: StateType, rng: Any,
        params: ParamsType | None = None,
    ) -> jax.Array:
        """Calculates the scalar reward for the transition."""
        raise NotImplementedError

    def terminal(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> jax.Array:
        """Evaluates whether the episode ended in failure.

        Failures zero out the Bellman bootstrap. Non-failure cutoffs must be 
        handled by `truncal`.
        """
        raise NotImplementedError

    def truncal(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> jax.Array:
        """Evaluates whether the episode ended for a non-failure reason.

        The driver ORs this with its own step-limit truncation. Defaults to False.
        """
        return jax.numpy.bool_(False)

    def state_info(
        self, state: StateType, params: ParamsType | None = None,
    ) -> dict[str, Any]:
        """Surfaces diagnostics for a single state on `reset`."""
        return {}

    def transition_info(
        self, state: StateType, action: ActType, next_state: StateType,
        params: ParamsType | None = None,
    ) -> dict[str, Any]:
        """Surfaces diagnostics for one transition on `step`.

        Keys under `"metrics"` are averaged per epoch and logged as `train/<key>`.
        """
        return {}

    def init_params(self) -> ParamsType | None:
        """Initializes traced environment parameters."""
        return None

    def observe_params(
        self, params: ParamsType | None, info: dict[str, Any], terminated: Any,
    ) -> ParamsType | None:
        """Folds one step's outcome into the traced parameters."""
        return params

    def epoch_refresh(self, params: ParamsType | None) -> tuple[Any, bool]:
        """Refreshes parameters on the host once per epoch.

        Returns:
            A tuple of `(params, invalidated)`. If `invalidated` is True, 
            in-progress episodes are reset and discarded.
        """
        return params, False

    def transform(self, func: Callable[[Callable], Callable]) -> None:
        """Applies a JAX transformation to every method in place.

        Note: Mutates the environment instance. Once transformed, it is no 
        longer callable on single states.
        """
        self.initial = func(self.initial)
        self.transition = func(self.transition)
        self.observation = func(self.observation)
        self.reward = func(self.reward)
        self.terminal = func(self.terminal)
        self.truncal = func(self.truncal)
        self.state_info = func(self.state_info)
        self.transition_info = func(self.transition_info)


def box(low, high, shape=None, dtype=np.float32) -> spaces.Box:
    """Creates a `gymnasium.spaces.Box` tolerant of JAX arrays as bounds."""
    low = np.asarray(low, dtype=dtype)
    high = np.asarray(high, dtype=dtype)
    if shape is None:
        shape = np.broadcast_shapes(low.shape, high.shape)
    return spaces.Box(
        low=np.broadcast_to(low, shape).copy(),
        high=np.broadcast_to(high, shape).copy(),
        shape=tuple(shape),
        dtype=dtype,
    )


def unbounded_box(size: int, dtype=np.float32) -> spaces.Box:
    """Creates an unbounded Box space of the specified size."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(int(size),), dtype=dtype)


def space_size(space: spaces.Space) -> int:
    """Calculates the flat scalar size of a Box space."""
    return int(np.prod(space.shape))