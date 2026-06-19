import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from roxie.exploration.noisy import (
    GaussianNoise,
    OrnsteinUhlenbeckNoise,
    CompositeNoise,
)
from roxie.exploration.schedulers import (
    ConstantSchedule,
    LinearDecay,
    ExponentialDecay,
)


class TestConstantSchedule:
    def test_returns_initial_value(self):
        schedule = ConstantSchedule()
        assert schedule(1.0, 0) == 1.0
        assert schedule(1.0, 1000) == 1.0
        assert schedule(0.5, 500) == 0.5


class TestLinearDecay:
    def test_at_start(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        result = schedule(1.0, 0)
        assert jnp.isclose(result, 1.0)

    def test_at_end(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        result = schedule(1.0, 100)
        assert jnp.isclose(result, 0.0)

    def test_midpoint(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        result = schedule(1.0, 50)
        assert jnp.isclose(result, 0.5)

    def test_past_end(self):
        schedule = LinearDecay(decay_steps=100, final_value=0.1)
        result = schedule(1.0, 200)
        assert jnp.isclose(result, 0.1)


class TestExponentialDecay:
    def test_at_start(self):
        schedule = ExponentialDecay(decay_rate=0.5, decay_steps=100)
        result = schedule(1.0, 0)
        assert jnp.isclose(result, 1.0)

    def test_decay(self):
        schedule = ExponentialDecay(decay_rate=0.5, decay_steps=100)
        result = schedule(1.0, 100)
        assert jnp.isclose(result, 0.5)

    def test_min_value(self):
        schedule = ExponentialDecay(decay_rate=0.1, decay_steps=10, min_value=0.05)
        result = schedule(1.0, 100)
        assert jnp.isclose(result, 0.05)


class TestGaussianNoise:
    def test_noise_shape(self, rng_key):
        action_shape = (4,)
        noise = GaussianNoise(action_shape=action_shape)
        sample = noise.sample_noise(rng_key)
        assert sample.shape == action_shape

    def test_noise_shape_custom(self, rng_key):
        noise = GaussianNoise(action_shape=(4,))
        sample = noise.sample_noise(rng_key, shape=(8, 4))
        assert sample.shape == (8, 4)

    def test_add_noise_evaluation(self, rng_key):
        action_shape = (4,)
        noise = GaussianNoise(action_shape=action_shape)
        actions = jnp.ones(action_shape)
        noisy_actions = noise.add_noise(actions, rng_key, evaluation=True)
        assert jnp.allclose(noisy_actions, actions)

    def test_add_noise_training(self, rng_key):
        action_shape = (4,)
        noise = GaussianNoise(action_shape=action_shape, initial_noise_scale=0.5)
        actions = jnp.ones(action_shape)
        noisy_actions = noise.add_noise(actions, rng_key, evaluation=False)
        assert not jnp.allclose(noisy_actions, actions)

    def test_step_counter_increments(self, rng_key):
        noise = GaussianNoise(action_shape=(4,))
        actions = jnp.ones((4,))
        assert noise.step_count.value == 0
        noise.add_noise(actions, rng_key, evaluation=False)
        assert noise.step_count.value == 1

    def test_step_counter_no_increment_eval(self, rng_key):
        noise = GaussianNoise(action_shape=(4,))
        actions = jnp.ones((4,))
        noise.add_noise(actions, rng_key, evaluation=True)
        assert noise.step_count.value == 0

    def test_hyperparameters(self):
        noise = GaussianNoise(action_shape=(4,), initial_noise_scale=0.2)
        hp = noise.hyperparameters()
        assert hp["name"] == "Gaussian"
        assert hp["initial_noise_scale"] == 0.2
        assert hp["action_shape"] == (4,)

    def test_with_linear_decay(self, rng_key):
        schedule = LinearDecay(decay_steps=100, final_value=0.0)
        noise = GaussianNoise(
            action_shape=(4,), initial_noise_scale=1.0, decay_schedule=schedule
        )
        scale_start = noise.get_current_scale()
        assert jnp.isclose(scale_start, 1.0)


class TestOrnsteinUhlenbeckNoise:
    def test_noise_shape(self, rng_key):
        action_shape = (4,)
        noise = OrnsteinUhlenbeckNoise(action_shape=action_shape)
        sample = noise.sample_noise(rng_key)
        assert sample.shape == action_shape

    def test_temporal_correlation(self, rng_key):
        action_shape = (4,)
        noise = OrnsteinUhlenbeckNoise(action_shape=action_shape, damping=0.15)
        samples = []
        for i in range(10):
            key = jax.random.fold_in(rng_key, i)
            samples.append(noise.sample_noise(key))
        first = samples[0]
        last = samples[-1]
        assert not jnp.allclose(first, last)

    def test_reset_noise(self, rng_key):
        action_shape = (4,)
        noise = OrnsteinUhlenbeckNoise(action_shape=action_shape)
        noise.sample_noise(rng_key)
        assert not jnp.allclose(noise.noise_state.value, jnp.zeros(action_shape))
        noise.reset_noise()
        assert jnp.allclose(noise.noise_state.value, jnp.zeros(action_shape))

    def test_hyperparameters(self):
        noise = OrnsteinUhlenbeckNoise(
            action_shape=(4,), initial_noise_scale=0.3, damping=0.2, mu=0.1
        )
        hp = noise.hyperparameters()
        assert hp["name"] == "Ornstein-Uhlenbeck"
        assert hp["damping"] == 0.2
        assert hp["mu"] == 0.1


class TestCompositeNoise:
    def test_combines_noise_sources(self, rng_key):
        action_shape = (4,)
        g1 = GaussianNoise(action_shape=action_shape, initial_noise_scale=0.1)
        g2 = GaussianNoise(action_shape=action_shape, initial_noise_scale=0.2)
        composite = CompositeNoise(noise_modules=[g1, g2], weights=[0.5, 0.5])
        sample = composite.sample_noise(rng_key)
        assert sample.shape == action_shape

    def test_weight_validation(self):
        action_shape = (4,)
        g1 = GaussianNoise(action_shape=action_shape)
        g2 = GaussianNoise(action_shape=action_shape)
        with pytest.raises(ValueError):
            CompositeNoise(noise_modules=[g1, g2], weights=[1.0])

    def test_add_noise_eval(self, rng_key):
        action_shape = (4,)
        g1 = GaussianNoise(action_shape=action_shape)
        g2 = GaussianNoise(action_shape=action_shape)
        composite = CompositeNoise(noise_modules=[g1, g2])
        actions = jnp.ones(action_shape)
        result = composite.add_noise(actions, rng_key, evaluation=True)
        assert jnp.allclose(result, actions)
