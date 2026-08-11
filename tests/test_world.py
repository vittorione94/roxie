import copy

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

from roxie.models.critics import QCritic, TwinCritic
from roxie.models.world import (
    TOLD,
    Encoder,
    LatentDynamics,
    RewardPredictor,
    simnorm,
)


@pytest.fixture
def latent_dim():
    return 12


@pytest.fixture
def told(obs_dim, action_dim, latent_dim, hidden_features):
    """A full TOLD model, wired exactly the way the agent will wire it: the Q
    heads are the existing QCritic/TwinCritic, just over latents."""
    return TOLD(
        encoder=Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=latent_dim,
            rngs=nnx.Rngs(params=0, dropout=1),
        ),
        dynamics=LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=nnx.Rngs(params=2, dropout=3),
        ),
        reward=RewardPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=nnx.Rngs(params=4, dropout=5),
        ),
        critic=TwinCritic(
            QCritic(
                in_features=latent_dim + action_dim,
                features=hidden_features,
                rngs=nnx.Rngs(params=6, dropout=7),
            ),
            QCritic(
                in_features=latent_dim + action_dim,
                features=hidden_features,
                rngs=nnx.Rngs(params=8, dropout=9),
            ),
        ),
    )


class TestSimNorm:
    def test_groups_sum_to_one(self, rng_key):
        x = jax.random.normal(rng_key, (4, 12)) * 5
        out = simnorm(x, group_size=4)
        assert out.shape == x.shape
        groups = out.reshape(4, 3, 4)
        assert jnp.allclose(jnp.sum(groups, axis=-1), 1.0, atol=1e-5)
        assert jnp.all(out >= 0.0)

    def test_preserves_leading_dims(self, rng_key):
        x = jax.random.normal(rng_key, (2, 5, 8))
        assert simnorm(x, group_size=2).shape == (2, 5, 8)


class TestEncoder:
    def test_output_shape(self, rngs, obs_dim, latent_dim, batch_size, hidden_features):
        enc = Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=latent_dim,
            rngs=rngs,
        )
        z = enc(jnp.ones((batch_size, obs_dim)))
        assert z.shape == (batch_size, latent_dim)

    def test_extra_leading_dims(self, rngs, obs_dim, latent_dim, hidden_features):
        """The planner batches over (samples, envs); the nets must pass any
        number of leading axes through untouched."""
        enc = Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=latent_dim,
            rngs=rngs,
        )
        z = enc(jnp.ones((7, 5, obs_dim)))
        assert z.shape == (7, 5, latent_dim)

    def test_layer_norm(self, rngs, obs_dim, latent_dim, batch_size, hidden_features):
        enc = Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=latent_dim,
            rngs=rngs,
            use_layer_norm=True,
        )
        assert enc(jnp.ones((batch_size, obs_dim))).shape == (batch_size, latent_dim)

    def test_simnorm_latent_is_normalized(
        self, rngs, obs_dim, batch_size, hidden_features, rng_key
    ):
        enc = Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=12,
            rngs=rngs,
            simnorm_group=4,
        )
        z = enc(jax.random.normal(rng_key, (batch_size, obs_dim)) * 10)
        groups = z.reshape(batch_size, 3, 4)
        assert jnp.allclose(jnp.sum(groups, axis=-1), 1.0, atol=1e-5)

    def test_simnorm_group_must_divide_latent(self, rngs, obs_dim, hidden_features):
        with pytest.raises(ValueError):
            Encoder(
                in_features=obs_dim,
                features=hidden_features,
                latent_dim=10,
                rngs=rngs,
                simnorm_group=4,
            )

    def test_deterministic(self, rngs, obs_dim, latent_dim, hidden_features):
        enc = Encoder(
            in_features=obs_dim,
            features=hidden_features,
            latent_dim=latent_dim,
            rngs=rngs,
        )
        x = jnp.ones((4, obs_dim))
        assert jnp.allclose(enc(x), enc(x))


class TestLatentDynamics:
    def test_output_shape(self, rngs, latent_dim, action_dim, batch_size, hidden_features):
        dyn = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
        )
        z = jnp.ones((batch_size, latent_dim))
        a = jnp.ones((batch_size, action_dim))
        assert dyn(z, a).shape == (batch_size, latent_dim)

    def test_extra_leading_dims(self, rngs, latent_dim, action_dim, hidden_features):
        dyn = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
        )
        z = jnp.ones((7, 5, latent_dim))
        a = jnp.ones((7, 5, action_dim))
        assert dyn(z, a).shape == (7, 5, latent_dim)

    def test_action_changes_output(
        self, rngs, latent_dim, action_dim, hidden_features, rng_key
    ):
        dyn = LatentDynamics(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
        )
        z = jax.random.normal(rng_key, (8, latent_dim))
        a1 = jnp.ones((8, action_dim))
        a2 = -jnp.ones((8, action_dim))
        assert not jnp.allclose(dyn(z, a1), dyn(z, a2))

    def test_simnorm_output_is_normalized(
        self, rngs, action_dim, hidden_features, rng_key
    ):
        dyn = LatentDynamics(
            latent_dim=12,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
            simnorm_group=4,
        )
        z = jax.random.normal(rng_key, (8, 12))
        a = jnp.ones((8, action_dim))
        groups = dyn(z, a).reshape(8, 3, 4)
        assert jnp.allclose(jnp.sum(groups, axis=-1), 1.0, atol=1e-5)


class TestRewardPredictor:
    def test_output_shape(self, rngs, latent_dim, action_dim, batch_size, hidden_features):
        rew = RewardPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
        )
        z = jnp.ones((batch_size, latent_dim))
        a = jnp.ones((batch_size, action_dim))
        assert rew(z, a).shape == (batch_size, 1)

    def test_extra_leading_dims(self, rngs, latent_dim, action_dim, hidden_features):
        rew = RewardPredictor(
            latent_dim=latent_dim,
            action_dim=action_dim,
            features=hidden_features,
            rngs=rngs,
        )
        z = jnp.ones((7, 5, latent_dim))
        a = jnp.ones((7, 5, action_dim))
        assert rew(z, a).shape == (7, 5, 1)


class TestTOLD:
    def test_component_shapes(self, told, obs_dim, action_dim, latent_dim, batch_size):
        obs = jnp.ones((batch_size, obs_dim))
        a = jnp.ones((batch_size, action_dim))

        z = told.encode(obs)
        assert z.shape == (batch_size, latent_dim)
        assert told.next(z, a).shape == (batch_size, latent_dim)
        assert told.predict_reward(z, a).shape == (batch_size, 1)

        q1, q2 = told.q(z, a)
        assert q1.shape == (batch_size, 1)
        assert q2.shape == (batch_size, 1)
        assert not jnp.allclose(q1, q2)

    def test_step_matches_components(self, told, obs_dim, action_dim, batch_size):
        obs = jnp.ones((batch_size, obs_dim))
        a = jnp.ones((batch_size, action_dim))
        z = told.encode(obs)
        z_next, r = told.step(z, a)
        assert jnp.allclose(z_next, told.next(z, a))
        assert jnp.allclose(r, told.predict_reward(z, a))

    def test_imagined_rollout_under_jit(
        self, told, obs_dim, action_dim, latent_dim, batch_size
    ):
        """The planner's inner loop: scan the dynamics forward from one encode.
        Exercises the model under jit with a traced carry, which is how it will
        actually be used."""
        horizon = 4

        @nnx.jit
        def rollout(model, obs, actions):
            z = model.encode(obs)

            def body(z, a):
                z, r = model.step(z, a)
                return z, r

            return jax.lax.scan(body, z, actions)

        obs = jnp.ones((batch_size, obs_dim))
        actions = jnp.ones((horizon, batch_size, action_dim))
        z_final, rewards = rollout(told, obs, actions)
        assert z_final.shape == (batch_size, latent_dim)
        assert rewards.shape == (horizon, batch_size, 1)

    def test_split_merge_roundtrip(self, told, obs_dim, action_dim, batch_size):
        """TOLD lives in TrainState.critic, so it has to survive the
        nnx.split/merge that Agent.save/load is built on."""
        graphdef, state = nnx.split(told)
        restored = nnx.merge(graphdef, state)

        obs = jnp.ones((batch_size, obs_dim))
        a = jnp.ones((batch_size, action_dim))
        z = told.encode(obs)
        assert jnp.allclose(restored.encode(obs), z)
        assert jnp.allclose(restored.next(z, a), told.next(z, a))
        assert jnp.allclose(restored.predict_reward(z, a), told.predict_reward(z, a))

    def test_soft_target_update(self, told, obs_dim, batch_size):
        """The whole model (encoder included) must soft-update through the same
        deepcopy + optax.incremental_update path the other agents use."""
        target = copy.deepcopy(told)

        # Perturb the live model so the update has something to move toward.
        kernel = told.encoder.output_layer.kernel
        before = jnp.asarray(target.encoder.output_layer.kernel.value)
        told.encoder.output_layer.kernel.value = kernel.value + 1.0

        new_tensors = nnx.state(told, nnx.Param)
        old_tensors = nnx.state(target, nnx.Param)
        nnx.update(
            target,
            optax.incremental_update(
                new_tensors=new_tensors, old_tensors=old_tensors, step_size=0.5
            ),
        )

        expected = before + 0.5
        assert jnp.allclose(target.encoder.output_layer.kernel.value, expected)
        # Target still runs after the update.
        assert target.encode(jnp.ones((batch_size, obs_dim))).shape[0] == batch_size
