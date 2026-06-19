import jax
import jax.numpy as jnp
import pytest
from flax import nnx

from roxie.models.actors import DeterministicActor, StochasticActor
from roxie.models.critics import QCritic, VCritic, TwinCritic


class TestDeterministicActor:
    def test_output_shape(self, rngs, obs_dim, action_dim, batch_size, hidden_features):
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((batch_size, obs_dim))
        out = actor(x)
        assert out.shape == (batch_size, action_dim)

    def test_output_bounded(self, rngs, obs_dim, action_dim, hidden_features, rng_key):
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jax.random.normal(rng_key, (32, obs_dim)) * 10
        out = actor(x)
        assert jnp.all(out >= -1.0)
        assert jnp.all(out <= 1.0)

    def test_single_hidden_layer(self, rngs, obs_dim, action_dim):
        actor = DeterministicActor(
            in_features=obs_dim, features=[32], action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((4, obs_dim))
        out = actor(x)
        assert out.shape == (4, action_dim)

    def test_layer_norm(self, rngs, obs_dim, action_dim, hidden_features, batch_size):
        actor = DeterministicActor(
            in_features=obs_dim,
            features=hidden_features,
            action_dim=action_dim,
            rngs=rngs,
            use_layer_norm=True,
        )
        x = jnp.ones((batch_size, obs_dim))
        out = actor(x)
        assert out.shape == (batch_size, action_dim)

    def test_single_sample(self, rngs, obs_dim, action_dim, hidden_features):
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((1, obs_dim))
        out = actor(x)
        assert out.shape == (1, action_dim)

    def test_deterministic(self, rngs, obs_dim, action_dim, hidden_features):
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((4, obs_dim))
        out1 = actor(x)
        out2 = actor(x)
        assert jnp.allclose(out1, out2)


class TestStochasticActor:
    def test_returns_distribution(self, rngs, obs_dim, action_dim, hidden_features, batch_size):
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((batch_size, obs_dim))
        dist = actor(x)
        assert hasattr(dist, "sample")
        assert hasattr(dist, "log_prob")

    def test_sample_shape(self, rngs, obs_dim, action_dim, hidden_features, batch_size, rng_key):
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((batch_size, obs_dim))
        dist = actor(x)
        sample = dist.sample(seed=rng_key)
        assert sample.shape == (batch_size, action_dim)

    def test_log_prob_shape(self, rngs, obs_dim, action_dim, hidden_features, batch_size, rng_key):
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((batch_size, obs_dim))
        dist = actor(x)
        sample = dist.sample(seed=rng_key)
        log_prob = dist.log_prob(sample)
        assert log_prob.shape == (batch_size,)

    def test_entropy_shape(self, rngs, obs_dim, action_dim, hidden_features, batch_size):
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jnp.ones((batch_size, obs_dim))
        dist = actor(x)
        entropy = dist.entropy()
        assert entropy.shape == (batch_size,)

    def test_std_clamping(self, rngs, obs_dim, action_dim, hidden_features, rng_key):
        std_min, std_max = 0.01, 0.5
        actor = StochasticActor(
            in_features=obs_dim,
            features=hidden_features,
            action_dim=action_dim,
            rngs=rngs,
            std_min=std_min,
            std_max=std_max,
        )
        x = jax.random.normal(rng_key, (32, obs_dim)) * 100
        dist = actor(x)
        std = dist.stddev()
        assert jnp.all(std >= std_min)
        assert jnp.all(std <= std_max + 1e-5)

    def test_layer_norm(self, rngs, obs_dim, action_dim, hidden_features, batch_size):
        actor = StochasticActor(
            in_features=obs_dim,
            features=hidden_features,
            action_dim=action_dim,
            rngs=rngs,
            use_layer_norm=True,
        )
        x = jnp.ones((batch_size, obs_dim))
        dist = actor(x)
        assert hasattr(dist, "sample")


class TestQCritic:
    def test_output_shape(self, rngs, obs_dim, action_dim, batch_size, hidden_features):
        critic = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs
        )
        obs = jnp.ones((batch_size, obs_dim))
        actions = jnp.ones((batch_size, action_dim))
        out = critic(obs, actions)
        assert out.shape == (batch_size, 1)

    def test_layer_norm(self, rngs, obs_dim, action_dim, batch_size, hidden_features):
        critic = QCritic(
            in_features=obs_dim + action_dim,
            features=hidden_features,
            rngs=rngs,
            use_layer_norm=True,
        )
        obs = jnp.ones((batch_size, obs_dim))
        actions = jnp.ones((batch_size, action_dim))
        out = critic(obs, actions)
        assert out.shape == (batch_size, 1)

    def test_deterministic_inference(self, rngs, obs_dim, action_dim, hidden_features):
        critic = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs
        )
        obs = jnp.ones((4, obs_dim))
        actions = jnp.ones((4, action_dim))
        out1 = critic(obs, actions, training=False)
        out2 = critic(obs, actions, training=False)
        assert jnp.allclose(out1, out2)


class TestVCritic:
    def test_output_shape(self, rngs, obs_dim, batch_size, hidden_features):
        critic = VCritic(
            in_features=obs_dim, features=hidden_features, rngs=rngs
        )
        obs = jnp.ones((batch_size, obs_dim))
        out = critic(obs)
        assert out.shape == (batch_size, 1)

    def test_layer_norm(self, rngs, obs_dim, batch_size, hidden_features):
        critic = VCritic(
            in_features=obs_dim,
            features=hidden_features,
            rngs=rngs,
            use_layer_norm=True,
        )
        obs = jnp.ones((batch_size, obs_dim))
        out = critic(obs)
        assert out.shape == (batch_size, 1)


class TestTwinCritic:
    def test_output_shapes(self, obs_dim, action_dim, batch_size, hidden_features):
        rngs1 = nnx.Rngs(params=0, dropout=1)
        rngs2 = nnx.Rngs(params=2, dropout=3)
        c1 = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs1
        )
        c2 = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs2
        )
        twin = TwinCritic(c1, c2)
        obs = jnp.ones((batch_size, obs_dim))
        actions = jnp.ones((batch_size, action_dim))
        q1, q2 = twin(obs, actions)
        assert q1.shape == (batch_size, 1)
        assert q2.shape == (batch_size, 1)

    def test_critics_differ(self, obs_dim, action_dim, hidden_features, rng_key):
        rngs1 = nnx.Rngs(params=0, dropout=1)
        rngs2 = nnx.Rngs(params=2, dropout=3)
        c1 = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs1
        )
        c2 = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs2
        )
        twin = TwinCritic(c1, c2)
        obs = jax.random.normal(rng_key, (8, obs_dim))
        actions = jax.random.normal(rng_key, (8, action_dim))
        q1, q2 = twin(obs, actions)
        assert not jnp.allclose(q1, q2)
