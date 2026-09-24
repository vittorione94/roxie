import abc

import jax
import jax.numpy as jnp

from roxie.utils.precision import FLOAT


class DecaySchedule(abc.ABC):
    """Abstract base class for noise decay schedules.

    Every schedule returns `FLOAT` rather than whatever its arithmetic promoted
    to: `step / decay_steps` is integer division, and the result scales the
    exploration noise, so an unpinned schedule widens the action away from the
    dtype the replay buffer was allocated with.
    """

    @abc.abstractmethod
    def __call__(self, initial_value: float, step: int) -> float:
        """Compute the decayed value at the given step."""
        pass


class LinearDecay(DecaySchedule):
    """Linear decay schedule."""

    def __init__(self, decay_steps: int, final_value: float = 0.0):
        self.decay_steps = decay_steps
        self.final_value = final_value

    def __call__(self, initial_value: float, step: int) -> float:
        decay_fraction = step / self.decay_steps
        decayed_value = (
            initial_value - (initial_value - self.final_value) * decay_fraction
        )
        # Both branches pinned: `lax.cond` requires them to agree, and a bare
        # Python float takes the canonical width rather than this one.
        return jax.lax.cond(
            step >= self.decay_steps,
            lambda _: jnp.asarray(self.final_value, dtype=FLOAT),
            lambda _: jnp.asarray(decayed_value, dtype=FLOAT),
            operand=None,
        )


class ExponentialDecay(DecaySchedule):
    """Exponential decay schedule."""

    def __init__(self, decay_rate: float, decay_steps: int, min_value: float = 0.01):
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.min_value = min_value

    def __call__(self, initial_value: float, step: int) -> float:
        decayed = initial_value * (self.decay_rate ** (step / self.decay_steps))
        return jnp.maximum(decayed, self.min_value).astype(FLOAT)


class ConstantSchedule(DecaySchedule):
    """Constant schedule (no decay)."""

    def __call__(self, initial_value: float, step: int) -> float:
        return jnp.asarray(initial_value, dtype=FLOAT)
