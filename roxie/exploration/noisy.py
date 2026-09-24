"""Exploration noise modules for action and parameter space perturbation."""

import abc
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.exploration.schedulers import ConstantSchedule, DecaySchedule
from roxie.utils.precision import FLOAT


class NoiseModule(nnx.Module):
    """Abstract base class for action-space exploration noise modules."""

    def __init__(
        self,
        action_shape: tuple,
        initial_noise_scale: float = 0.1,
        decay_schedule: Optional[DecaySchedule] = None,
    ):
        self.action_shape = action_shape
        self.initial_noise_scale = initial_noise_scale
        self.decay_schedule = decay_schedule or ConstantSchedule()
        self.step_count = nnx.Variable(0)

    @abc.abstractmethod
    def sample_noise(
        self, key: jax.Array, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        """Samples unscaled noise for the current step."""
        pass

    def get_current_scale(self) -> float:
        """Computes the active noise scale based on the decay schedule."""
        return self.decay_schedule(
            self.initial_noise_scale, self.step_count.get_value()
        )

    def reset_noise(self):
        """Resets internal state buffers on episode boundaries."""
        pass

    def add_noise(
        self, actions: jnp.ndarray, key: jax.Array, evaluation: bool = False
    ) -> jnp.ndarray:
        """Applies scaled noise to actions and increments step counters."""
        if evaluation:
            return actions

        current_scale = self.get_current_scale()
        noise = self.sample_noise(key, shape=actions.shape) * current_scale

        self.step_count.set_value(
            self.step_count.get_value()
            + (actions.shape[0] if actions.ndim > 1 else 1)
        )

        return actions + noise


class GaussianNoise(NoiseModule):
    """Uncorrelated zero-mean Gaussian action noise."""

    def sample_noise(
        self, key: jax.Array, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        """Samples independent Gaussian noise from standard normal distribution."""
        shape = shape or self.action_shape
        return jax.random.normal(key, shape, dtype=FLOAT)

    def hyperparameters(self) -> dict:
        """Returns hyperparameters for logging."""
        return {
            "name": "Gaussian",
            "initial_noise_scale": self.initial_noise_scale,
            "decay_schedule": self.decay_schedule.__class__.__name__,
            "action_shape": self.action_shape,
        }


class OrnsteinUhlenbeckNoise(NoiseModule):
    """Temporally correlated Ornstein-Uhlenbeck action noise process."""

    def __init__(
        self,
        action_shape: tuple,
        initial_noise_scale: float = 0.1,
        decay_schedule: Optional[DecaySchedule] = None,
        theta: float = 0.15,
        dt: float = 1.0,
        clip: float = 2.0,
        mu: float = 0.0,
    ):
        super().__init__(action_shape, initial_noise_scale, decay_schedule)
        self.theta = theta
        self.dt = dt
        self.clip = clip
        self.mu = mu
        self.noise_state = nnx.Variable(jnp.zeros(action_shape, dtype=FLOAT))

    def sample_noise(
        self, key: jax.Array, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        """Samples next state in the mean-reverting OU noise process."""
        current_noise = self.noise_state.get_value()

        mean_reversion = self.theta * self.dt * (self.mu - current_noise)

        shape = shape or self.action_shape
        gaussian = jnp.clip(
            jax.random.normal(key, shape, dtype=FLOAT), -self.clip, self.clip
        )
        random_component = jnp.sqrt(2.0 * self.theta * self.dt) * gaussian

        new_noise = current_noise + mean_reversion + random_component
        self.noise_state.set_value(new_noise)

        return new_noise

    def reset_noise(self):
        """Resets the noise state to zero."""
        self.noise_state.set_value(jnp.zeros(self.action_shape, dtype=FLOAT))

    def hyperparameters(self) -> dict:
        """Returns hyperparameters for logging."""
        return {
            "name": "Ornstein-Uhlenbeck",
            "initial_noise_scale": self.initial_noise_scale,
            "decay_schedule": self.decay_schedule.__class__.__name__,
            "action_shape": self.action_shape,
            "theta": self.theta,
            "dt": self.dt,
            "clip": self.clip,
            "mu": self.mu,
        }


class ParameterNoise(nnx.Module):
    """Exploration noise applied directly to neural network weights."""

    def __init__(
        self,
        initial_stddev: float = 0.1,
        desired_action_stddev: float = 0.1,
        adaptation_coefficient: float = 1.01,
        decay_schedule: Optional[DecaySchedule] = None,
    ):
        self.initial_stddev = initial_stddev
        self.desired_action_stddev = desired_action_stddev
        self.adaptation_coefficient = adaptation_coefficient
        self.decay_schedule = decay_schedule or ConstantSchedule()

        self.current_stddev = nnx.Variable(initial_stddev)
        self.step_count = nnx.Variable(0)

    def perturb_parameters(
        self, model: nnx.Module, key: jax.Array
    ) -> nnx.Module:
        """Returns a copy of the input model with perturbed parameters."""
        import copy

        perturbed_model = copy.deepcopy(model)

        current_scale = self.decay_schedule(
            self.current_stddev.get_value(), self.step_count.get_value()
        )

        params = nnx.state(perturbed_model, nnx.Param)

        def add_param_noise(param, key):
            noise = jax.random.normal(
                key, param.shape, dtype=param.dtype
            ) * current_scale
            return param + noise

        keys = jax.random.split(key, len(jax.tree_leaves(params)))
        key_tree = jax.tree_unflatten(jax.tree_structure(params), keys)

        noisy_params = jax.tree_map(add_param_noise, params, key_tree)
        nnx.update(perturbed_model, noisy_params)

        return perturbed_model

    def adapt_noise(self, action_distance: float):
        """Adapts parameter standard deviation based on measured action distance."""
        if action_distance > self.desired_action_stddev:
            self.current_stddev.set_value(
                self.current_stddev.get_value() / self.adaptation_coefficient
            )
        else:
            self.current_stddev.set_value(
                self.current_stddev.get_value() * self.adaptation_coefficient
            )

        self.step_count.set_value(self.step_count.get_value() + 1)

    def hyperparameters(self) -> dict:
        """Returns hyperparameters for logging."""
        return {
            "name": "ParameterNoise",
            "initial_stddev": self.initial_stddev,
            "desired_action_stddev": self.desired_action_stddev,
            "adaptation_coefficient": self.adaptation_coefficient,
            "decay_schedule": self.decay_schedule.__class__.__name__,
        }


class AdaptiveNoise(NoiseModule):
    """Action noise scale that dynamically adapts to empirical action variance."""

    def __init__(
        self,
        action_shape: tuple,
        initial_noise_scale: float = 0.1,
        decay_schedule: Optional[DecaySchedule] = None,
        adaptation_rate: float = 0.01,
        target_variance: float = 0.1,
        window_size: int = 100,
    ):
        super().__init__(action_shape, initial_noise_scale, decay_schedule)
        self.adaptation_rate = adaptation_rate
        self.target_variance = target_variance
        self.window_size = window_size

        self.action_buffer = nnx.Variable(
            jnp.zeros((window_size,) + action_shape, dtype=FLOAT)
        )
        self.buffer_index = nnx.Variable(0)
        self.buffer_full = nnx.Variable(False)

        self.adaptive_scale = nnx.Variable(1.0)

    def update_action_history(self, actions: jnp.ndarray):
        """Appends recent actions to the rolling history buffer."""
        idx = self.buffer_index.get_value() % self.window_size
        self.action_buffer.set_value(
            self.action_buffer.get_value().at[idx].set(actions)
        )
        self.buffer_index.set_value(self.buffer_index.get_value() + 1)

        if self.buffer_index.get_value() >= self.window_size:
            self.buffer_full.set_value(True)

    def compute_action_variance(self) -> float:
        """Calculates variance across recent action history."""
        if not self.buffer_full.get_value():
            return self.target_variance

        actions = self.action_buffer.get_value()
        return jnp.var(actions)

    def adapt_scale(self):
        """Adjusts adaptive multiplier to match target action variance."""
        current_variance = self.compute_action_variance()

        if current_variance < self.target_variance:
            self.adaptive_scale.set_value(
                self.adaptive_scale.get_value() * (1 + self.adaptation_rate)
            )
        else:
            self.adaptive_scale.set_value(
                self.adaptive_scale.get_value() * (1 - self.adaptation_rate)
            )

        self.adaptive_scale.set_value(
            jnp.clip(self.adaptive_scale.get_value(), 0.1, 10.0)
        )

    def sample_noise(self, key: jax.Array) -> jnp.ndarray:
        """Samples standard Gaussian action noise."""
        return jax.random.normal(key, self.action_shape, dtype=FLOAT)

    def add_noise(
        self, actions: jnp.ndarray, key: jax.Array, evaluation: bool = False
    ) -> jnp.ndarray:
        """Applies variance-adapted noise to actions."""
        if evaluation:
            return actions

        self.update_action_history(actions)
        self.adapt_scale()

        base_scale = self.get_current_scale()
        effective_scale = base_scale * self.adaptive_scale.get_value()

        noise = self.sample_noise(key) * effective_scale
        self.step_count.set_value(self.step_count.get_value() + 1)

        return actions + noise
