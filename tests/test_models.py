import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import distrax

from roxie.models.actors import DeterministicActor, StochasticActor, TanhNormal
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

    def test_forward_returns_pre_activation(
        self, rngs, obs_dim, action_dim, hidden_features, rng_key
    ):
        """`forward` exposes the pre-tanh logits, and `__call__` is its action."""
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jax.random.normal(rng_key, (8, obs_dim))
        action, pre_activation = actor.forward(x)
        assert pre_activation.shape == (8, action_dim)
        assert jnp.allclose(action, jnp.tanh(pre_activation))
        assert jnp.allclose(action, actor(x))

    def test_output_init_starts_unsaturated(self, obs_dim, action_dim, rng_key):
        """The small final-layer init keeps the logits in tanh's linear region.

        Regression guard for the saturation collapse: at the default init scale
        the policy must start with a live gradient (1 - a^2 near 1), so the
        pre-activation penalty only has to hold it there.
        """
        actor = DeterministicActor(
            in_features=obs_dim,
            features=[256, 256],
            action_dim=action_dim,
            rngs=nnx.Rngs(params=0, dropout=1),
            use_layer_norm=True,
        )
        x = jax.random.normal(rng_key, (256, obs_dim))
        action, pre_activation = actor.forward(x)
        assert jnp.mean(jnp.abs(pre_activation)) < 0.5
        assert jnp.mean(1.0 - action ** 2) > 0.9

    def test_output_init_scale_controls_logit_magnitude(
        self, obs_dim, action_dim, rng_key
    ):
        def mean_abs_logit(scale):
            actor = DeterministicActor(
                in_features=obs_dim,
                features=[256, 256],
                action_dim=action_dim,
                rngs=nnx.Rngs(params=0, dropout=1),
                use_layer_norm=True,
                output_init_scale=scale,
            )
            return jnp.mean(jnp.abs(actor.forward(jax.random.normal(
                rng_key, (256, obs_dim)))[1]))

        # variance_scaling scales the VARIANCE, so 100x scale ~ 10x the logits.
        small, large = mean_abs_logit(0.01), mean_abs_logit(1.0)
        assert large > 5.0 * small


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


class TestTanhSquashedActor:
    """`squash=True` must bound actions and keep the density self-consistent."""

    OBS_DIM = 8
    ACT_DIM = 4

    @classmethod
    def _actor(cls, squash):
        OBS_DIM, ACT_DIM = cls.OBS_DIM, cls.ACT_DIM
        return StochasticActor(
            in_features=OBS_DIM, features=[32, 32], action_dim=ACT_DIM,
            rngs=nnx.Rngs(params=0, dropout=1), squash=squash, std_max=5.0,
        )

    def test_default_is_unsquashed(self):
        """SAC/MPO share this class -- the default must not change behaviour."""
        d = self._actor(False)(jnp.zeros((4, self.OBS_DIM)))
        assert isinstance(d, distrax.MultivariateNormalDiag)

    def test_samples_and_mean_are_in_range(self):
        d = self._actor(True)(jax.random.normal(jax.random.PRNGKey(0), (64, self.OBS_DIM)))
        a = d.sample(seed=jax.random.PRNGKey(1))
        assert jnp.all(jnp.abs(a) < 1.0)
        assert jnp.all(jnp.abs(d.mean()) < 1.0)

    def test_log_prob_roundtrips_through_the_squash(self):
        """log_prob(a) must match the value returned alongside the sample --
        this is what makes PPO's stored old_log_probs consistent with the ratio
        recomputed later."""
        d = self._actor(True)(jax.random.normal(jax.random.PRNGKey(0), (64, self.OBS_DIM)))
        a, lp = d.sample_and_log_prob(seed=jax.random.PRNGKey(1))
        assert jnp.allclose(lp, d.log_prob(a), atol=1e-3)

    def test_log_prob_roundtrips_at_saturating_scale(self):
        """Regression: the round-trip must hold in the tail, not just near 0.

        A freshly initialised actor emits sigma ~ 0.7, so `u` never reaches the
        arctanh clip and the test above passes even when `sample_and_log_prob`
        scores the raw `u`. At the configured std_max=5 a large share of draws
        saturate tanh in float32, the action stops identifying its own `u`, and
        the two log-probs diverge by whole nats -- which PPO would read as KL
        and clipping on an update that has not changed the policy yet.
        """
        n = 8192
        d = TanhNormal(
            jnp.zeros((n, self.ACT_DIM)), jnp.full((n, self.ACT_DIM), 5.0))
        a, lp = d.sample_and_log_prob(seed=jax.random.PRNGKey(0))

        # The regime this test exists for: the sample must actually saturate.
        assert float(jnp.mean(jnp.abs(a) >= 1.0 - 1e-6)) > 0.05

        assert jnp.allclose(lp, d.log_prob(a), atol=1e-3)

        # What it costs PPO: recomputing the ratio against an unchanged policy
        # must be a no-op, so approx_kl and clip_frac stay at zero.
        ratio = jnp.exp(d.log_prob(a) - lp)
        approx_kl = jnp.mean((ratio - 1.0) - jnp.log(ratio))
        assert float(approx_kl) < 1e-5
        assert float(jnp.mean(jnp.abs(ratio - 1.0) > 0.2)) == 0.0

    def test_entropy_has_an_interior_maximum_in_sigma(self):
        """The whole point of squashing, for an entropy bonus.

        An unsquashed Normal's entropy is const + sum(log sigma): unbounded and
        monotonically increasing, so the bonus pays forever to inflate sigma and
        the clipped-away spread costs nothing. Squashed, large sigma pushes
        tanh(u) onto the two atoms at +-1, so the differential entropy PEAKS
        (near sigma ~ 1) and then falls. The bonus therefore has an interior
        optimum and stops driving sigma upward.
        """
        key = jax.random.PRNGKey(0)
        loc = jnp.zeros((4096, self.ACT_DIM))
        sq = lambda s: float(jnp.mean(
            TanhNormal(loc, jnp.full((4096, self.ACT_DIM), s)).entropy(seed=key)))
        plain = lambda s: float(jnp.mean(
            distrax.MultivariateNormalDiag(
                loc, jnp.full((4096, self.ACT_DIM), s)).entropy()))

        # squashed: rises to a peak, then decreases
        assert sq(1.0) > sq(0.25)
        assert sq(1.0) > sq(5.0) > sq(10.0)
        # plain: strictly increasing over the same range
        assert plain(0.25) < plain(1.0) < plain(5.0) < plain(10.0)
