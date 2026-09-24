"""The network blocks, at the level the agents and the configs use them.

Shape and plumbing checks are collapsed one per block: they all fail together
for the same reason (a layer wired wrong), and re-asserting `out.shape` once per
constructor flag only multiplies the count. What each block PROMISES beyond its
shape — the small output init that keeps a fresh actor unsaturated, and the
squashed distribution `TestTanhSquashedActor` pins — is a test each.
"""

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

import distrax

from roxie.models.actors import DeterministicActor, StochasticActor, TanhNormal
from roxie.models.critics import QCritic, VCritic, TwinCritic


@pytest.fixture
def rngs2():
    """A second init stream, for the twin critic's other head."""
    return nnx.Rngs(params=2, dropout=3)


class TestDeterministicActor:
    @pytest.mark.parametrize("features", [[32], [64, 64]])
    def test_maps_a_batch_to_a_bounded_action_deterministically(
        self, rngs, obs_dim, action_dim, features, rng_key
    ):
        """Every geometry the configs use, and the three things they all owe:
        the declared shape, the tanh bound, and no hidden per-call state."""
        actor = DeterministicActor(
            in_features=obs_dim, features=features, action_dim=action_dim, rngs=rngs,
        )
        # Deliberately far outside the training range: the bound is a property
        # of the squash, not of the inputs.
        x = jax.random.normal(rng_key, (32, obs_dim), dtype=jnp.float32) * 10

        out = actor(x)
        assert out.shape == (32, action_dim)
        assert jnp.all(out >= -1.0) and jnp.all(out <= 1.0)
        assert jnp.allclose(out, actor(x)), "a second call disagreed"
        # A single sample is the eval path's batch, and must not be special.
        assert actor(x[:1]).shape == (1, action_dim)

    def test_forward_returns_pre_activation(
        self, rngs, obs_dim, action_dim, hidden_features, rng_key
    ):
        """`forward` exposes the pre-tanh logits, and `__call__` is its action."""
        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim, rngs=rngs
        )
        x = jax.random.normal(rng_key, (8, obs_dim), dtype=jnp.float32)
        action, pre_activation = actor.forward(x)
        assert pre_activation.shape == (8, action_dim)
        assert jnp.allclose(action, jnp.tanh(pre_activation))
        assert jnp.allclose(action, actor(x))

    def test_output_init_starts_unsaturated_and_scales_with_the_knob(
        self, obs_dim, action_dim, rng_key
    ):
        """The small final-layer init keeps the logits in tanh's linear region.

        Regression guard for the saturation collapse: at the default init scale
        the policy must start with a live gradient (1 - a^2 near 1), so the
        pre-activation penalty only has to hold it there. `output_init_scale` is
        the knob that sets it, so it has to move the logits when turned.
        """
        def actor(scale=None):
            kwargs = {} if scale is None else {"output_init_scale": scale}
            return DeterministicActor(
                in_features=obs_dim, features=[256, 256], action_dim=action_dim,
                rngs=nnx.Rngs(params=0, dropout=1), use_layer_norm=True, **kwargs,
            )

        x = jax.random.normal(rng_key, (256, obs_dim), dtype=jnp.float32)
        action, pre_activation = actor().forward(x)
        assert jnp.mean(jnp.abs(pre_activation)) < 0.5
        assert jnp.mean(1.0 - action ** 2) > 0.9

        # variance_scaling scales the VARIANCE, so 100x scale ~ 10x the logits.
        logits = lambda scale: jnp.mean(jnp.abs(actor(scale).forward(x)[1]))
        assert logits(1.0) > 5.0 * logits(0.01)


class TestStochasticActor:
    def test_returns_a_pre_activation_keyed_distribution_over_the_batch(
        self, rngs, obs_dim, action_dim, batch_size, hidden_features, rng_key
    ):
        """Sample, density and entropy all come back batch-shaped — and the
        density is keyed on the PRE-TANH draw. `log_prob` is pinned absent
        deliberately: an action-keyed density cannot be correct here (see
        `test_recovering_u_from_the_action_would_be_wrong`), so the name must
        not exist for a caller to reach for.
        """
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim,
            rngs=rngs,
        )
        dist = actor(jnp.ones((batch_size, obs_dim), dtype=jnp.float32))

        assert not hasattr(dist, "log_prob")
        sample, pre = dist.sample_from_pre(seed=rng_key)
        assert sample.shape == (batch_size, action_dim)
        assert jnp.allclose(sample, dist.sample(seed=rng_key))
        assert dist.log_prob_from_pre(pre).shape == (batch_size,)
        assert dist.entropy().shape == (batch_size,)

    def test_unsquashed_actor_presents_the_same_interface(
        self, rngs, obs_dim, action_dim, batch_size, hidden_features, rng_key
    ):
        """`squash=False` swaps the distribution, not the contract.

        PPO acts and scores through the same five calls whichever it gets, so a
        `ClippedNormal` missing one of them fails inside a trace rather than here.
        """
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim,
            rngs=rngs, squash=False,
        )
        dist = actor(jnp.ones((batch_size, obs_dim), dtype=jnp.float32))

        assert not hasattr(dist, "log_prob")
        sample, pre = dist.sample_from_pre(seed=rng_key)
        assert sample.shape == (batch_size, action_dim)
        assert dist.log_prob_from_pre(pre).shape == (batch_size,)
        # Exact, so it takes no draw — but it still accepts the seed the loss
        # passes, because `TanhNormal` needs one.
        assert dist.entropy(seed=rng_key).shape == (batch_size,)
        assert dist.entropy().dtype == jnp.float32

    def test_unsquashed_density_carries_no_tanh_jacobian(
        self, rngs, obs_dim, action_dim, hidden_features, rng_key
    ):
        """The point of the swap, stated as a number.

        `TanhNormal.log_prob_from_pre` subtracts `sum(log(1 - tanh(u)^2))`, which
        grows as `2|u|` per dimension — that is what makes a PPO ratio explode as
        the mean walks past the rail (`test_losses.py::TestRatioSurvivesSaturation`).
        Clipping instead leaves the plain Gaussian, whose density is flat in `u`
        far from the mean, so the same drift costs a bounded log-ratio.
        """
        obs = jnp.ones((4, obs_dim), dtype=jnp.float32)

        def logp(squash, mean):
            actor = StochasticActor(
                in_features=obs_dim, features=hidden_features,
                action_dim=action_dim, rngs=nnx.Rngs(0), squash=squash,
            )
            actor.output_layer.kernel[...] = jnp.zeros_like(
                actor.output_layer.kernel[...]
            )
            actor.output_layer.bias[...] = jnp.full_like(
                actor.output_layer.bias[...], mean
            )
            dist = actor(obs)
            _, pre = dist.sample_from_pre(seed=rng_key)
            return float(jnp.mean(dist.log_prob_from_pre(pre)))

        # Past the float32 tanh rail at |u| >= 8, where the Jacobian term is the
        # whole density; the Gaussian is translation-invariant there and the
        # squashed one is not.
        assert logp(False, 1.0) == pytest.approx(logp(False, 40.0), abs=1.0)
        assert logp(True, 40.0) - logp(True, 1.0) > 100.0

    def test_log_std_param_is_exp_of_the_parameter_and_unclipped(
        self, rngs, obs_dim, action_dim, hidden_features
    ):
        """`log_std_param` makes the parameter log sigma itself.

        Under the softplus branch d(log sigma)/d(param) is sigmoid(raw)/sigma,
        which is 0.82 at sigma=0.4 and drifts as sigma moves; here it is exactly
        1, which is what the mjbatch reference's free `log_std` parameter gives
        and what makes Adam move the spread at the learning rate.
        """
        actor = StochasticActor(
            in_features=obs_dim, features=hidden_features, action_dim=action_dim,
            rngs=rngs, state_dependent_std=False, log_std_param=True,
            init_std=0.4, std_min=1e-2, std_max=5.0,
        )
        assert float(actor.log_std[...][0]) == pytest.approx(jnp.log(0.4), abs=1e-6)
        obs = jnp.ones((4, obs_dim), dtype=jnp.float32)
        assert float(actor(obs).scale_diag[0, 0]) == pytest.approx(0.4, abs=1e-5)

        # Unclipped: std_max must NOT bind under this parameterization.
        actor.log_std[...] = jnp.full_like(actor.log_std[...], jnp.log(9.0))
        assert float(actor(obs).scale_diag[0, 0]) == pytest.approx(9.0, rel=1e-5)

    def test_log_std_param_rejects_a_state_dependent_spread(
        self, rngs, obs_dim, action_dim, hidden_features
    ):
        """One shared log sigma and a per-state head are different models; asking
        for both is a config error, not something to silently resolve."""
        with pytest.raises(ValueError, match="state_dependent_std"):
            StochasticActor(
                in_features=obs_dim, features=hidden_features,
                action_dim=action_dim, rngs=rngs,
                state_dependent_std=True, log_std_param=True,
            )

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
        x = jax.random.normal(rng_key, (32, obs_dim), dtype=jnp.float32) * 100
        std = actor(x).stddev()
        assert jnp.all(std >= std_min)
        assert jnp.all(std <= std_max + 1e-5)


class TestCritics:
    def test_q_critic_scores_a_state_action_pair(
        self, rngs, obs_dim, action_dim, batch_size, hidden_features
    ):
        critic = QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs,
        )
        obs = jnp.ones((batch_size, obs_dim), dtype=jnp.float32)
        actions = jnp.ones((batch_size, action_dim), dtype=jnp.float32)

        out = critic(obs, actions, training=False)
        assert out.shape == (batch_size, 1)
        assert jnp.allclose(out, critic(obs, actions, training=False))

    def test_v_critic_scores_a_state(self, rngs, obs_dim, batch_size, hidden_features):
        """PPO's critic takes no action input."""
        critic = VCritic(in_features=obs_dim, features=hidden_features, rngs=rngs)
        out = critic(jnp.ones((batch_size, obs_dim), dtype=jnp.float32))
        assert out.shape == (batch_size, 1)

    def test_layer_norm_is_wired_on_every_block(
        self, rngs, obs_dim, action_dim, batch_size, hidden_features
    ):
        """`use_layer_norm: true` is in every shipped config; all three blocks
        have to accept it and keep their declared shape. A block that dropped
        the flag would be a silently un-normalized network, not an error."""
        obs = jnp.ones((batch_size, obs_dim), dtype=jnp.float32)
        actions = jnp.ones((batch_size, action_dim), dtype=jnp.float32)
        norm = dict(use_layer_norm=True, rngs=rngs)

        actor = DeterministicActor(
            in_features=obs_dim, features=hidden_features,
            action_dim=action_dim, **norm,
        )
        assert actor(obs).shape == (batch_size, action_dim)
        assert StochasticActor(
            in_features=obs_dim, features=hidden_features,
            action_dim=action_dim, **norm,
        )(obs).mean().shape == (batch_size, action_dim)
        assert QCritic(
            in_features=obs_dim + action_dim, features=hidden_features, **norm,
        )(obs, actions).shape == (batch_size, 1)
        assert VCritic(
            in_features=obs_dim, features=hidden_features, **norm,
        )(obs).shape == (batch_size, 1)

    def test_twin_heads_score_separately(
        self, rngs, rngs2, obs_dim, action_dim, batch_size, hidden_features, rng_key
    ):
        """A clipped double-Q min over two IDENTICAL critics is just a single
        critic, so the two heads must be initialized apart."""
        twin = TwinCritic(
            QCritic(in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs),
            QCritic(in_features=obs_dim + action_dim, features=hidden_features, rngs=rngs2),
        )
        obs = jax.random.normal(rng_key, (batch_size, obs_dim), dtype=jnp.float32)
        actions = jax.random.normal(rng_key, (batch_size, action_dim), dtype=jnp.float32)

        q1, q2 = twin(obs, actions)
        assert q1.shape == q2.shape == (batch_size, 1)
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
        d = self._actor()(jnp.zeros((4, self.OBS_DIM), dtype=jnp.float32))
        assert isinstance(d, TanhNormal)
        assert not isinstance(d, distrax.MultivariateNormalDiag)

    def test_samples_and_mean_are_in_range(self):
        d = self._actor()(jax.random.normal(
            jax.random.PRNGKey(0), (64, self.OBS_DIM), dtype=jnp.float32
        ))
        a = d.sample(seed=jax.random.PRNGKey(1))
        assert jnp.all(jnp.abs(a) < 1.0)
        assert jnp.all(jnp.abs(d.mean()) < 1.0)

    def test_sample_shape_draws_a_leading_axis(self):
        """MPO's E-step draws S actions per state; `sample_from_pre` must carry
        the same shape as `sample` so the pre-activations line up with them."""
        d = self._actor()(jnp.zeros((6, self.OBS_DIM), dtype=jnp.float32))
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
        d = self._actor()(jax.random.normal(
            jax.random.PRNGKey(0), (16, self.OBS_DIM), dtype=jnp.float32
        ))
        assert d.loc.shape == d.scale_diag.shape == (16, self.ACT_DIM)
        assert jnp.allclose(d.mean(), jnp.tanh(d.loc))
        assert jnp.allclose(d.stddev(), d.scale_diag)

    def test_log_prob_from_pre_stays_finite_where_tanh_saturates(self):
        """Every stored density is scored from `u`, so saturation must be a
        non-event: at sigma 5 most draws land on the rail, and one -inf
        log-prob NaNs a whole ratio."""
        n = 2048
        d = TanhNormal(
            jnp.zeros(
                (n, self.ACT_DIM), dtype=jnp.float32
            ), jnp.full((n, self.ACT_DIM), 5.0, dtype=jnp.float32))
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
        n = 2048
        d = TanhNormal(jnp.zeros(
            (n, 1), dtype=jnp.float32
        ), jnp.full((n, 1), 5.0, dtype=jnp.float32))
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
        loc = jnp.zeros((2048, self.ACT_DIM), dtype=jnp.float32)
        sq = lambda s: float(jnp.mean(
            TanhNormal(loc, jnp.full((2048, self.ACT_DIM), s)).entropy(seed=key)))
        plain = lambda s: float(jnp.mean(
            distrax.MultivariateNormalDiag(
                loc, jnp.full((2048, self.ACT_DIM), s)).entropy()))

        # squashed: rises to a peak, then decreases
        assert sq(1.0) > sq(0.25)
        assert sq(1.0) > sq(5.0) > sq(10.0)
        # plain: strictly increasing over the same range
        assert plain(0.25) < plain(1.0) < plain(5.0) < plain(10.0)
