import abc
from typing import Optional

import jax
import jax.numpy as jnp
from flax import nnx

from roxie.exploration.schedulers import ConstantSchedule, DecaySchedule


# Base noise module
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

        # Advance the decay clock by the number of environment frames collected
        # this call (the batch / env dimension), not by 1 per iteration. This
        # measures the schedule in env steps -- the same unit as the trainer's
        # `steps`/`memory_warmup` budgets -- so `decay_steps` is independent of
        # how many parallel envs are used (same behaviour at 1 env or 4000).
        self.step_count.value += actions.shape[0] if actions.ndim > 1 else 1

        return actions + noise


# Gaussian noise
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


# Ornstein-Uhlenbeck noise
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
        # Initialize the noise state
        self.noise_state = nnx.Variable(jnp.zeros(action_shape))

    def sample_noise(
        self, key: jax.random.PRNGKey, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        # OU process: dx = theta * (mu - x) * dt + sigma * dW
        # Discretized: x_t = x_{t-1} + theta * (mu - x_{t-1}) * dt + sqrt(dt) * noise
        # Same parameterization as roxie.agents.basic.OrnsteinUhlenbeck, so
        # `theta` is the continuous-time mean-reversion rate in both.
        current_noise = self.noise_state.value

        # Mean reversion term
        mean_reversion = self.theta * self.dt * (self.mu - current_noise)

        # Random component (gaussian sample clipped per step, then scaled by
        # sqrt(dt) -- same order as roxie.agents.basic.OrnsteinUhlenbeck).
        shape = shape or self.action_shape
        gaussian = jnp.clip(jax.random.normal(key, shape), -self.clip, self.clip)
        random_component = jnp.sqrt(self.dt) * gaussian

        # Update noise state
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


# Parameter noise (for parameter space exploration)
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

        # Add noise to all parameters
        params = nnx.state(perturbed_model, nnx.Param)

        def add_param_noise(param, key):
            noise = jax.random.normal(key, param.shape) * current_scale
            return param + noise

        # Split keys for each parameter
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


# Composite noise (combine multiple noise sources)
class CompositeNoise(NoiseModule):
    """Combine multiple noise sources."""

    def __init__(
        self, noise_modules: list[NoiseModule], weights: Optional[list[float]] = None
    ):
        # We'll use the first module's settings as defaults
        first_module = noise_modules[0]
        super().__init__(
            first_module.action_shape,
            first_module.initial_noise_scale,
            first_module.decay_schedule,
        )

        self.noise_modules = noise_modules
        self.weights = weights or [1.0] * len(noise_modules)

        if len(self.weights) != len(noise_modules):
            raise ValueError("Number of weights must match number of noise modules")

    def sample_noise(
        self, key: jax.random.PRNGKey, shape: Optional[tuple] = None
    ) -> jnp.ndarray:
        shape = shape or self.action_shape
        keys = jax.random.split(key, len(self.noise_modules))
        combined_noise = jnp.zeros(self.action_shape)

        for i, (module, weight) in enumerate(zip(self.noise_modules, self.weights)):
            noise = module.sample_noise(keys[i], shape=shape)
            combined_noise += weight * noise

        return combined_noise

    def add_noise(
        self, actions: jnp.ndarray, key: jax.random.PRNGKey, evaluation: bool = False
    ) -> jnp.ndarray:
        """Override to update all submodules."""
        if evaluation:
            return actions

        # Advance every submodule's decay clock (and our own) by the number of
        # environment frames in this call, so the schedule is measured in env
        # steps and stays independent of the parallel env count.
        inc = actions.shape[0] if actions.ndim > 1 else 1
        for module in self.noise_modules:
            module.step_count.value += inc
        self.step_count.value += inc

        current_scale = self.get_current_scale()
        noise = self.sample_noise(key) * current_scale

        return actions + noise


# Adaptive noise that adjusts based on action variance
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

        # Rolling buffer of recent actions
        self.action_buffer = nnx.Variable(jnp.zeros((window_size,) + action_shape))
        self.buffer_index = nnx.Variable(0)
        self.buffer_full = nnx.Variable(False)

        # Adaptive scale factor
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
            # Not enough data yet
            return self.target_variance

        actions = self.action_buffer.value
        return jnp.var(actions)

    def adapt_scale(self):
        """Adapt the noise scale based on action variance."""
        current_variance = self.compute_action_variance()

        # Simple proportional adaptation
        if current_variance < self.target_variance:
            self.adaptive_scale.value *= 1 + self.adaptation_rate
        else:
            self.adaptive_scale.value *= 1 - self.adaptation_rate

        # Keep scale positive and bounded
        self.adaptive_scale.value = jnp.clip(self.adaptive_scale.value, 0.1, 10.0)

    def sample_noise(self, key: jax.random.PRNGKey) -> jnp.ndarray:
        return jax.random.normal(key, self.action_shape)

    def add_noise(
        self, actions: jnp.ndarray, key: jax.random.PRNGKey, evaluation: bool = False
    ) -> jnp.ndarray:
        """Add adaptive noise to actions."""
        if evaluation:
            return actions

        # Update action history
        self.update_action_history(actions)

        # Adapt scale
        self.adapt_scale()

        # Get current base scale from decay schedule
        base_scale = self.get_current_scale()

        # Apply adaptive scaling
        effective_scale = base_scale * self.adaptive_scale.value

        noise = self.sample_noise(key) * effective_scale

        # Increment step counter
        self.step_count.value += 1

        return actions + noise
