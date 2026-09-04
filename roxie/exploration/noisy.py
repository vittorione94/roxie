import abc
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.exploration.schedulers import ConstantSchedule, DecaySchedule


class NoiseModule(nnx.Module):
    """Abstract base class for noise modules."""

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
        self, key: jax.random.PRNGKey, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        """Sample noise for the current step."""
        pass

    def get_current_scale(self) -> float:
        """Get the current noise scale based on decay schedule."""
        return self.decay_schedule(self.initial_noise_scale, self.step_count.value)

    def reset_noise(self):
        """Reset the noise module state."""
        pass

    def add_noise(
        self, actions: jnp.ndarray, key: jax.random.PRNGKey, evaluation: bool = False
    ) -> jnp.ndarray:
        """Add noise to actions and increment step counter."""
        if evaluation:
            return actions

        current_scale = self.get_current_scale()
        noise = self.sample_noise(key, shape=actions.shape) * current_scale

        # Advanced by env frames, not by 1 per call, so `decay_steps` is in the
        # same unit as the trainer's `steps` at any parallel_envs.
        self.step_count.value += actions.shape[0] if actions.ndim > 1 else 1

        return actions + noise


class GaussianNoise(NoiseModule):
    """Gaussian (white) noise module."""

    def sample_noise(
        self, key: jax.random.PRNGKey, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        shape = shape or self.action_shape
        return jax.random.normal(key, shape)

    def hyperparameters(self) -> dict:
        """Return hyperparameters for logging."""
        return {
            "name": "Gaussian",
            "initial_noise_scale": self.initial_noise_scale,
            "decay_schedule": self.decay_schedule.__class__.__name__,
            "action_shape": self.action_shape,
        }


class OrnsteinUhlenbeckNoise(NoiseModule):
    """Ornstein-Uhlenbeck (temporally correlated) noise module."""

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
        self.noise_state = nnx.Variable(jnp.zeros(action_shape))

    def sample_noise(
        self, key: jax.random.PRNGKey, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        # The sqrt(2*theta*dt) increment holds the stationary std at ~1.0 for
        # any theta/dt, so `initial_noise_scale` is the actual noise std. A bare
        # sqrt(dt) would couple amplitude to the mean-reversion rate — which the
        # OU-as-policy agent in roxie.agents.basic deliberately keeps.
        current_noise = self.noise_state.value

        mean_reversion = self.theta * self.dt * (self.mu - current_noise)

        shape = shape or self.action_shape
        gaussian = jnp.clip(jax.random.normal(key, shape), -self.clip, self.clip)
        random_component = jnp.sqrt(2.0 * self.theta * self.dt) * gaussian

        new_noise = current_noise + mean_reversion + random_component
        self.noise_state.value = new_noise

        return new_noise

    def reset_noise(self):
        """Reset the noise state (useful at episode boundaries)."""
        self.noise_state.value = jnp.zeros(self.action_shape)

    def hyperparameters(self) -> dict:
        """Return hyperparameters for logging."""
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
    """Parameter noise for parameter space exploration."""

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
        self, model: nnx.Module, key: jax.random.PRNGKey
    ) -> nnx.Module:
        """Create a perturbed copy of the model parameters."""
        import copy

        perturbed_model = copy.deepcopy(model)

        current_scale = self.decay_schedule(
            self.current_stddev.value, self.step_count.value
        )

        params = nnx.state(perturbed_model, nnx.Param)

        def add_param_noise(param, key):
            noise = jax.random.normal(key, param.shape) * current_scale
            return param + noise

        keys = jax.random.split(key, len(jax.tree_leaves(params)))
        key_tree = jax.tree_unflatten(jax.tree_structure(params), keys)

        noisy_params = jax.tree_map(add_param_noise, params, key_tree)
        nnx.update(perturbed_model, noisy_params)

        return perturbed_model

    def adapt_noise(self, action_distance: float):
        """Adapt noise based on the distance between clean and noisy actions."""
        if action_distance > self.desired_action_stddev:
            self.current_stddev.value /= self.adaptation_coefficient
        else:
            self.current_stddev.value *= self.adaptation_coefficient

        self.step_count.value += 1

    def hyperparameters(self) -> dict:
        """Return hyperparameters for logging."""
        return {
            "name": "ParameterNoise",
            "initial_stddev": self.initial_stddev,
            "desired_action_stddev": self.desired_action_stddev,
            "adaptation_coefficient": self.adaptation_coefficient,
            "decay_schedule": self.decay_schedule.__class__.__name__,
        }


class AdaptiveNoise(NoiseModule):
    """Adaptive noise that adjusts based on recent action variance."""

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

        self.action_buffer = nnx.Variable(jnp.zeros((window_size,) + action_shape))
        self.buffer_index = nnx.Variable(0)
        self.buffer_full = nnx.Variable(False)

        self.adaptive_scale = nnx.Variable(1.0)

    def update_action_history(self, actions: jnp.ndarray):
        """Update the rolling buffer of actions."""
        idx = self.buffer_index.value % self.window_size
        self.action_buffer.value = self.action_buffer.value.at[idx].set(actions)
        self.buffer_index.value += 1

        if self.buffer_index.value >= self.window_size:
            self.buffer_full.value = True

    def compute_action_variance(self) -> float:
        """Compute variance of recent actions."""
        if not self.buffer_full.value:
            return self.target_variance

        actions = self.action_buffer.value
        return jnp.var(actions)

    def adapt_scale(self):
        """Adapt the noise scale based on action variance."""
        current_variance = self.compute_action_variance()

        if current_variance < self.target_variance:
            self.adaptive_scale.value *= 1 + self.adaptation_rate
        else:
            self.adaptive_scale.value *= 1 - self.adaptation_rate

        self.adaptive_scale.value = jnp.clip(self.adaptive_scale.value, 0.1, 10.0)

    def sample_noise(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.normal(key, self.action_shape)

    def add_noise(
        self, actions: jnp.ndarray, key: jax.random.PRNGKey, evaluation: bool = False
    ) -> jnp.ndarray:
        """Add adaptive noise to actions."""
        if evaluation:
            return actions

        self.update_action_history(actions)

        self.adapt_scale()

        base_scale = self.get_current_scale()

        effective_scale = base_scale * self.adaptive_scale.value

        noise = self.sample_noise(key) * effective_scale

        self.step_count.value += 1

        return actions + noise
