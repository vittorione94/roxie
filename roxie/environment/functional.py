"""Writing an environment: a stateless, JAX-transformable interface.

This is roxie's version of ``gymnasium.functional.FuncEnv`` — the same method
set, the same ``params`` convention, the same spaces — so a Gymnasium
``FuncEnv`` subclass satisfies this structurally and drops in unchanged. It is
re-declared here rather than imported because upstream still ships it under
``gymnasium.experimental`` and documents it as "predominantly for internal
use... This API is likely to change". Roxie's *entire* env boundary sits on
this, so pinning it to an API upstream reserves the right to break is a worse
trade than forty lines.

An env written against this class owns no mutable state: every method takes the
state it operates on and returns a new one. That is what lets the driver in
``roxie.environment.vector`` ``vmap`` the env across a thousand worlds, ``jit``
a whole step, and ``scan`` an entire warmup rollout into one dispatch.

``params`` is the escape hatch for values that change between calls but must
stay TRACED — refreshing them must not retrigger a compile. An env that ADAPTS
its own ``params`` as training goes (a start-state curriculum, say) owns that
adaptation itself, through ``init_params`` / ``observe_params`` /
``epoch_refresh``; the driver only carries the value and hands it back. See
those three methods.

One deliberate deviation from Gymnasium: ``truncal``. Gymnasium's driver
hardcodes ``truncated = steps >= time_limit`` and its ``FuncEnv`` has no way to
say "this episode ended for a non-failure reason of the env's own". A motion-
tracking env has exactly that — the reference clip runs out — and it must stay
OUT of ``terminal``, or the critic zeroes the bootstrap at the cutoff and Q
collapses there. It defaults to False, so a plain Gymnasium ``FuncEnv`` is unaffected.
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
    """Stateless environment: state in, state out, nothing on ``self``.

    Subclasses set ``observation_space``/``action_space`` and implement at least
    ``initial``, ``transition``, ``observation``, ``reward`` and ``terminal``.

    Every method takes ``rng`` and optional ``params`` in the same positions as
    Gymnasium's, so the two are interchangeable. Methods are called under
    ``jax.vmap`` by the driver, so they must be written for a SINGLE env and
    contain no Python branching on traced values.
    """

    observation_space: spaces.Space
    action_space: spaces.Space

    # Free-form, Gymnasium-style. Roxie reads ``metadata["impl"]`` for the
    # startup backend banner, which is how the loud "which physics actually
    # loaded" check survives the wrapper chain going away.
    metadata: dict[str, Any] = {"jax": True}

    def initial(self, rng: Any, params: ParamsType | None = None) -> StateType:
        """The start state of a fresh episode."""
        raise NotImplementedError

    def transition(
        self, state: StateType, action: ActType, rng: Any,
        params: ParamsType | None = None,
    ) -> StateType:
        """Advance the state by one control step."""
        raise NotImplementedError

    def observation(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> ObsType:
        """What the policy sees in ``state``."""
        raise NotImplementedError

    def reward(
        self, state: StateType, action: ActType, next_state: StateType, rng: Any,
        params: ParamsType | None = None,
    ) -> jax.Array:
        """Scalar reward for the ``(state, action, next_state)`` transition."""
        raise NotImplementedError

    def terminal(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> jax.Array:
        """Did the episode END IN FAILURE here?

        Failure only — a fall, a NaN, a tracking collapse. Anything that ends an
        episode without the outcome being bad (a time limit, a reference
        trajectory running out) belongs in ``truncal``, because the two are
        treated differently by every value-based agent in the repo: a
        termination zeroes the Bellman bootstrap, a truncation keeps it.
        """
        raise NotImplementedError

    def truncal(
        self, state: StateType, rng: Any, params: ParamsType | None = None,
    ) -> jax.Array:
        """Did the episode end here for a NON-failure reason of the env's own?

        Roxie's one addition to the Gymnasium method set (see the module
        docstring). The driver ORs this with its own step-limit truncation, so an
        env with no internal notion of "ran out" simply leaves it alone.
        """
        return jax.numpy.bool_(False)

    def state_info(
        self, state: StateType, params: ParamsType | None = None,
    ) -> dict[str, Any]:
        """Diagnostics for a single state, surfaced on ``reset``."""
        return {}

    def transition_info(
        self, state: StateType, action: ActType, next_state: StateType,
        params: ParamsType | None = None,
    ) -> dict[str, Any]:
        """Diagnostics for one transition, surfaced on ``step``.

        Per-step env metrics go under the ``"metrics"`` key: the trainer means
        them over each epoch and logs them as ``train/<key>``. Everything else is
        passed through untouched, for consumers that know what to do with it —
        ``observe_params`` is handed this dict verbatim, so an env that adapts
        its ``params`` puts whatever that needs here.
        """
        return {}

    # -- self-adapting ``params`` -------------------------------------------
    #
    # Three optional hooks, all no-ops by default, that let an env change its
    # own ``params`` while training runs without ever owning mutable state
    # inside a trace. The driver just carries the value: it takes one from
    # ``init_params``, passes it to every ``initial``/``transition``/... call,
    # threads it through ``observe_params`` on each step (INSIDE the jitted
    # step, so that stays one dispatch), and refreshes it once per epoch. Since
    # ``params`` is a traced argument throughout, none of that recompiles.

    def init_params(self) -> ParamsType | None:
        """The initial ``params``, or None for an env that needs none."""
        return None

    def observe_params(
        self, params: ParamsType | None, info: dict[str, Any], terminated: Any,
    ) -> ParamsType | None:
        """Fold one step's outcome into ``params``. Runs under ``jit``/``vmap``
        with BATCHED arguments — ``info`` is this step's ``transition_info`` and
        ``terminated`` is failure only, done MINUS truncation, so a non-failure
        cutoff is never mistaken for one.
        """
        return params

    def epoch_refresh(self, params: ParamsType | None) -> tuple[Any, bool]:
        """Once per epoch, on the host: return ``(params, invalidated)``.

        Free to mutate the env — this is the one call that is not inside a
        trace — but a mutation that leaves IN-PROGRESS episodes referring to
        data that no longer exists must report ``invalidated=True``, which tells
        the driver to reset the live envs and the trainer to drop their
        part-scored episodes.
        """
        return params, False

    def transform(self, func: Callable[[Callable], Callable]) -> None:
        """Apply a JAX transformation to every method, in place.

        ``func_env.transform(jax.vmap)`` is how the driver batches an env across
        worlds. Note this MUTATES the env — Gymnasium does the same — so an env
        that is transformed is no longer callable on single states.
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
    """A ``gymnasium.spaces.Box``, tolerant of JAX arrays as bounds.

    ``spaces.Box`` requires numpy; MuJoCo action bounds arrive as either
    ``mj_model.actuator_ctrlrange`` columns (numpy) or device arrays (from an
    env that already moved them), so normalize here rather than at each call
    site.
    """
    low = np.asarray(low, dtype=dtype)
    high = np.asarray(high, dtype=dtype)
    if shape is None:
        shape = np.broadcast_shapes(low.shape, high.shape)
    # Broadcast explicitly: gymnasium accepts a PYTHON scalar bound against an
    # explicit shape but rejects a 0-d ndarray, and `np.asarray` above has
    # already made every bound an ndarray.
    return spaces.Box(
        low=np.broadcast_to(low, shape).copy(),
        high=np.broadcast_to(high, shape).copy(),
        shape=tuple(shape),
        dtype=dtype,
    )


def unbounded_box(size: int, dtype=np.float32) -> spaces.Box:
    """The observation space for an env that declares no observation bounds —
    which is every MuJoCo env in this repo."""
    return spaces.Box(low=-np.inf, high=np.inf, shape=(int(size),), dtype=dtype)


def space_size(space: spaces.Space) -> int:
    """Flat size of a Box, which is what the agents' ``in_features`` want."""
    return int(np.prod(space.shape))
