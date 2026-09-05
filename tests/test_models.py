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
        assert hasattr(dist, "log_prob_from_pre")
        # Pinned deliberately: an action-keyed density cannot be correct here
        # (see `test_recovering_u_from_the_action_would_be_wrong`), so the name
        # must not exist for a caller to reach for.
        assert not hasattr(dist, "log_prob")

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
        _sample, pre = dist.sample_from_pre(seed=rng_key)
        log_prob = dist.log_prob_from_pre(pre)
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
    """`StochasticActor` bounds its actions and keeps the density usable there.

    The squash is unconditional -- SAC and MPO read the tanh log-prob correction
    straight off `TanhNormal`, and PPO relies on the bounded mean -- so these
    also stand in for the removed `squash=False` branch.
    """

    OBS_DIM = 8
    ACT_DIM = 4

    @classmethod
    def _actor(cls):
        OBS_DIM, ACT_DIM = cls.OBS_DIM, cls.ACT_DIM
        return StochasticActor(
            in_features=OBS_DIM, features=[32, 32], action_dim=ACT_DIM,
            rngs=nnx.Rngs(params=0, dropout=1), std_max=5.0,
        )

    def test_actor_always_squashes(self):
        """No flag re-introduces an unbounded Normal: every agent's loss now
        calls `TanhNormal`-only methods, so one would build a broken agent."""
        d = self._actor()(jnp.zeros((4, self.OBS_DIM)))
        assert isinstance(d, TanhNormal)
        assert not isinstance(d, distrax.MultivariateNormalDiag)

    def test_samples_and_mean_are_in_range(self):
        d = self._actor()(jax.random.normal(jax.random.PRNGKey(0), (64, self.OBS_DIM)))
        a = d.sample(seed=jax.random.PRNGKey(1))
        assert jnp.all(jnp.abs(a) < 1.0)
        assert jnp.all(jnp.abs(d.mean()) < 1.0)

    def test_sample_shape_draws_a_leading_axis(self):
        """MPO's E-step draws S actions per state; `sample_from_pre` must carry
        the same shape as `sample` so the pre-activations line up with them."""
        d = self._actor()(jnp.zeros((6, self.OBS_DIM)))
        a = d.sample(seed=jax.random.PRNGKey(0), sample_shape=(5,))
        assert a.shape == (5, 6, self.ACT_DIM)

        a2, u = d.sample_from_pre(seed=jax.random.PRNGKey(0), sample_shape=(5,))
        assert a2.shape == u.shape == (5, 6, self.ACT_DIM)
        # Same seed, same draw: the two entry points must not diverge.
        assert jnp.allclose(a, a2)
        assert jnp.allclose(a2, jnp.tanh(u))

    def test_gaussian_params_are_exposed_under_distrax_names(self):
        """MPO's decoupled trust region reads `loc` / `scale_diag` off whatever
        the actor returns; tanh is a bijection, so those PRE-squash parameters
        are what the KL is (exactly) defined on."""
        d = self._actor()(jax.random.normal(jax.random.PRNGKey(0), (16, self.OBS_DIM)))
        assert d.loc.shape == d.scale_diag.shape == (16, self.ACT_DIM)
        assert jnp.allclose(d.mean(), jnp.tanh(d.loc))
        assert jnp.allclose(d.stddev(), d.scale_diag)

    def test_log_prob_from_pre_stays_finite_where_tanh_saturates(self):
        """Every stored density is scored from `u`, so saturation must be a
        non-event: at sigma 5 most draws land on the rail, and one -inf
        log-prob NaNs a whole ratio."""
        n = 8192
        d = TanhNormal(
            jnp.zeros((n, self.ACT_DIM)), jnp.full((n, self.ACT_DIM), 5.0))
        a, u = d.sample_from_pre(seed=jax.random.PRNGKey(0))

        # The regime this test exists for: the draws must actually saturate.
        assert float(jnp.mean(jnp.abs(a) >= 1.0 - 1e-6)) > 0.05

        lp = d.log_prob_from_pre(u)
        assert jnp.all(jnp.isfinite(lp))
        # Deterministic in `u`, which is what makes PPO's first-pass ratio
        # exactly 1.
        assert jnp.array_equal(lp, d.log_prob_from_pre(u))

    def test_recovering_u_from_the_action_would_be_wrong(self):
        """Why `TanhNormal` has no `log_prob(action)` overload at all.

        An action-keyed density would have to recover `u` through an arctanh
        clipped short of 1.0, scoring every draw past that rail (~7.25) as if
        it had landed exactly on it. Near zero the two agree; in the tail they
        are whole nats apart, and it is the recovered value that is wrong --
        MPO's M-step would fit the online mean to a rail rather than to the
        draw, and PPO's ratio would stop tracking its own policy.
        """
        # One action dimension, so `log_prob`'s sum over it is exactly the
        # per-draw quantity and a draw is interior or saturated as a whole.
        n = 8192
        d = TanhNormal(jnp.zeros((n, 1)), jnp.full((n, 1), 5.0))
        a, u = d.sample_from_pre(seed=jax.random.PRNGKey(0))

        # What the removed overload did, spelled out here so the trap stays
        # documented without leaving a callable version of it in the class.
        via_action = d.log_prob_from_pre(
            jnp.arctanh(jnp.clip(a, -(1.0 - 1e-6), 1.0 - 1e-6))
        )

        interior = jnp.squeeze(jnp.abs(u) < 1.0, axis=-1)
        assert float(jnp.mean(interior)) > 0.05
        assert jnp.allclose(
            d.log_prob_from_pre(u)[interior], via_action[interior], atol=1e-3
        )

        saturated = jnp.squeeze(jnp.abs(u) > 8.0, axis=-1)
        assert float(jnp.mean(saturated)) > 0.05
        gap = jnp.abs(d.log_prob_from_pre(u) - via_action)[saturated]
        assert float(jnp.min(gap)) > 1.0

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
