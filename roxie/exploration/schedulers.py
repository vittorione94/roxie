import abc

import jax
import jax.numpy as jnp


# Base classes for decay schedules
class DecaySchedule(abc.ABC):
    """Abstract base class for noise decay schedules."""

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
        return jax.lax.cond(
            step >= self.decay_steps,
            lambda _: self.final_value,
            lambda _: decayed_value,
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
        return jnp.maximum(decayed, self.min_value)


class ConstantSchedule(DecaySchedule):
    """Constant schedule (no decay)."""

    def __call__(self, initial_value: float, step: int) -> float:
        return initial_value
