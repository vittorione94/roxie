import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roxie.agents.agent import Agent, ObsStats
from roxie.agents.utils import Transition, serialize_bound


class TestScaleToEnv:
    def test_center(self):
        x = jnp.array([0.0])
        result = Agent.scale_to_env(x, jnp.array([-1.0]), jnp.array([1.0]))
        assert jnp.isclose(result, 0.0)

    def test_low_bound(self):
        x = jnp.array([-1.0])
        result = Agent.scale_to_env(x, jnp.array([-2.0]), jnp.array([2.0]))
        assert jnp.isclose(result, -2.0)

    def test_high_bound(self):
        x = jnp.array([1.0])
        result = Agent.scale_to_env(x, jnp.array([-2.0]), jnp.array([2.0]))
        assert jnp.isclose(result, 2.0)

    def test_asymmetric_bounds(self):
        x = jnp.array([0.0])
        result = Agent.scale_to_env(x, jnp.array([0.0]), jnp.array([10.0]))
        assert jnp.isclose(result, 5.0)

    def test_batch(self):
        x = jnp.array([[-1.0, 0.0, 1.0]])
        low = jnp.array([0.0, 0.0, 0.0])
        high = jnp.array([1.0, 1.0, 1.0])
        result = Agent.scale_to_env(x, low, high)
        expected = jnp.array([[0.0, 0.5, 1.0]])
        assert jnp.allclose(result, expected)


class TestObsStats:
    def test_init_obs_stats(self):
        stats = Agent.init_obs_stats(4)
        assert jnp.isclose(stats.count, 0.0)
        assert stats.sum.shape == (4,)
        assert stats.sumsq.shape == (4,)
        assert jnp.allclose(stats.sum, 0.0)
        assert jnp.allclose(stats.sumsq, 0.0)

    def test_update_obs_stats(self):
        stats = Agent.init_obs_stats(3)
        batch = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        updated = Agent.update_obs_stats(stats, batch)
        assert jnp.isclose(updated.count, 2.0)
        assert jnp.allclose(updated.sum, jnp.array([5.0, 7.0, 9.0]))

    def test_obs_mean_std(self):
        stats = Agent.init_obs_stats(2)
        batch = jnp.array([[2.0, 4.0], [4.0, 6.0], [6.0, 8.0]])
        stats = Agent.update_obs_stats(stats, batch)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        assert jnp.allclose(mean, jnp.array([4.0, 6.0]))
        expected_var = jnp.array([(4+16+36)/3 - 16, (16+36+64)/3 - 36])
        expected_std = jnp.sqrt(expected_var + 1e-8)
        assert jnp.allclose(std, expected_std, atol=1e-5)

    def test_obs_mean_std_empty(self):
        stats = Agent.init_obs_stats(3)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        assert jnp.allclose(mean, 0.0)

    def test_normalize_obs(self):
        x = jnp.array([10.0, 20.0])
        mean = jnp.array([5.0, 10.0])
        std = jnp.array([5.0, 5.0])
        result = Agent.normalize_obs(x, mean, std, clip=5.0)
        expected = jnp.array([1.0, 2.0])
        assert jnp.allclose(result, expected)

    def test_normalize_obs_clipping(self):
        x = jnp.array([100.0])
        mean = jnp.array([0.0])
        std = jnp.array([1.0])
        result = Agent.normalize_obs(x, mean, std, clip=3.0)
        assert jnp.isclose(result, 3.0)

    def test_full_pipeline(self):
        stats = Agent.init_obs_stats(2)
        for i in range(100):
            batch = jnp.array([[float(i), float(i) * 2]])
            stats = Agent.update_obs_stats(stats, batch)
        mean, std = Agent.obs_mean_std(stats, eps=1e-8)
        normalized = Agent.normalize_obs(jnp.array([50.0, 100.0]), mean, std, clip=5.0)
        assert jnp.all(jnp.abs(normalized) <= 5.0)


class TestTransition:
    def test_creation(self):
        t = Transition(
            observation=jnp.zeros(4),
            action=jnp.zeros(2),
            reward=jnp.array(1.0),
            terminal=jnp.array(False),
        )
        assert t.observation.shape == (4,)
        assert t.action.shape == (2,)
        assert t.log_probs is None
        assert t.value is None

    def test_with_optional_fields(self):
        t = Transition(
            observation=jnp.zeros(4),
            action=jnp.zeros(2),
            reward=jnp.array(1.0),
            terminal=jnp.array(False),
            log_probs=jnp.array(-0.5),
            value=jnp.array(3.0),
        )
        assert t.log_probs is not None
        assert t.value is not None


class TestSerializeBound:
    def test_scalar_float(self):
        result = serialize_bound(1.5)
        assert result == 1.5

    def test_jax_scalar(self):
        result = serialize_bound(jnp.array(2.0))
        assert result == 2.0
        assert isinstance(result, float)

    def test_numpy_scalar(self):
        result = serialize_bound(np.float32(3.0))
        assert result == 3.0
        assert isinstance(result, float)
