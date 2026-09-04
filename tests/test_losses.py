import copy

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

# Import agents first to avoid circular import
import roxie.agents  # noqa: F401
from roxie.agents.agent import Agent
from roxie.agents.sac import LogAlpha
from roxie.losses.actor_losses import (
    ACTOR_DIAGNOSTIC_KEYS,
    ddpg_actor_loss_fn,
    ppo_loss_fn,
    pre_activation_penalty,
    sac_actor_loss_fn,
    sac_alpha_loss_fn,
    td3_actor_loss_fn,
)
from roxie.losses.critic_losses import (
    _smoothed_target_actions,
    ddpg_critic_loss_fn,
    ppo_critic_loss_fn,
    sac_critic_loss_fn,
)
from roxie.models.actors import DeterministicActor, StochasticActor
from roxie.models.critics import QCritic, VCritic, TwinCritic


OBS_DIM = 8
ACT_DIM = 3
BATCH = 16
FEATURES = [32, 32]


@pytest.fixture
def det_actor():
    rngs = nnx.Rngs(params=0, dropout=1)
    return DeterministicActor(
        in_features=OBS_DIM, features=FEATURES, action_dim=ACT_DIM, rngs=rngs
    )


@pytest.fixture
def stoch_actor():
    rngs = nnx.Rngs(params=0, dropout=1)
    return StochasticActor(
        in_features=OBS_DIM, features=FEATURES, action_dim=ACT_DIM, rngs=rngs
    )


@pytest.fixture
def det_critic():
    rngs = nnx.Rngs(params=0, dropout=1)
    return QCritic(
        in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs
    )


@pytest.fixture
def stoch_critic():
    rngs = nnx.Rngs(params=0, dropout=1)
    return VCritic(in_features=OBS_DIM, features=FEATURES, rngs=rngs)


@pytest.fixture
def twin_critic():
    rngs1 = nnx.Rngs(params=0, dropout=1)
    rngs2 = nnx.Rngs(params=2, dropout=3)
    c1 = QCritic(in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs1)
    c2 = QCritic(in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs2)
    return TwinCritic(c1, c2)


@pytest.fixture
def ddpg_samples():
    """A repacked batch as the losses see it: observations ALREADY normalized.

    Every loss function takes them that way — the agent runs the batch through
    `Agent.normalize_samples` once per gradient step (see `TestSampleNormalization`),
    so no loss takes obs_mean/obs_std/obs_clip arguments.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "observations": jax.random.normal(k1, (BATCH, OBS_DIM)),
        "actions": jax.random.normal(k2, (BATCH, ACT_DIM)),
        "rewards": jax.random.normal(k3, (BATCH,)),
        "next_observations": jax.random.normal(k4, (BATCH, OBS_DIM)),
        # Per-sample bootstrap coefficient (gamma^b, 0 at terminals) — every
        # off-policy critic loss consumes this instead of gamma + terminal
        # flags. `terminals` is kept for MPO, which still reads it directly.
        "bootstrap": 0.99 * jnp.ones(BATCH),
        "terminals": jnp.zeros(BATCH, dtype=jnp.bool_),
    }


@pytest.fixture
def action_bounds():
    return {
        "action_low": jnp.full(ACT_DIM, -1.0),
        "action_high": jnp.full(ACT_DIM, 1.0),
    }


class TestSampleNormalization:
    """`Agent.normalize_samples` is the ONE place observations are normalized on
    the learning path, so every loss can take them pre-normalized."""

    def test_normalizes_both_observation_entries(self, ddpg_samples):
        mean = jnp.full(OBS_DIM, 2.0)
        std = jnp.full(OBS_DIM, 4.0)
        out = Agent.normalize_samples(ddpg_samples, mean, std, clip=5.0)
        for k in ("observations", "next_observations"):
            assert jnp.allclose(out[k], (ddpg_samples[k] - mean) / std)

    def test_leaves_the_rest_of_the_batch_alone(self, ddpg_samples):
        out = Agent.normalize_samples(
            ddpg_samples, jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), clip=5.0
        )
        assert out.keys() == ddpg_samples.keys()
        for k in ("actions", "rewards", "bootstrap", "terminals"):
            assert jnp.array_equal(out[k], ddpg_samples[k])

    def test_disabled_is_a_true_passthrough(self, ddpg_samples):
        """Off means off — the clip goes too.

        With normalization disabled the running stats are never updated and
        degrade to mean 0 / std 1, so clipping anyway would silently squash raw
        observations into +/- obs_clip. Guard it with observations well outside
        the clip bound.
        """
        raw = dict(ddpg_samples)
        raw["observations"] = ddpg_samples["observations"] * 100.0
        raw["next_observations"] = ddpg_samples["next_observations"] * 100.0
        out = Agent.normalize_samples(
            raw, jnp.zeros(OBS_DIM), jnp.ones(OBS_DIM), clip=5.0, enabled=False
        )
        assert jnp.array_equal(out["observations"], raw["observations"])
        assert jnp.max(jnp.abs(out["observations"])) > 5.0


class TestDDPGActorLoss:
    def test_returns_scalar(self, det_actor, det_critic, ddpg_samples, action_bounds):
        loss, aux = ddpg_actor_loss_fn(
            det_actor,
            det_critic,
            ddpg_samples,
            action_bounds["action_low"],
            action_bounds["action_high"],
            0.0,
        )
        assert loss.shape == ()
        assert set(aux) == {*ACTOR_DIAGNOSTIC_KEYS, "pre_act_penalty"}

    def test_finite(self, det_actor, det_critic, ddpg_samples, action_bounds):
        loss, _aux = ddpg_actor_loss_fn(
            det_actor,
            det_critic,
            ddpg_samples,
            action_bounds["action_low"],
            action_bounds["action_high"],
            0.0,
        )
        assert jnp.isfinite(loss)


class TestPreActivationPenalty:
    """The counterweight to DPG's unbounded outward push on the actor logits."""

    def test_free_inside_the_threshold(self):
        u = jnp.array([[-1.0, -0.4, 0.0, 0.9, 1.0]])
        assert pre_activation_penalty(u) == 0.0

    def test_grows_quadratically_in_the_overshoot(self):
        # relu(|u| - 1)^2 over a single element.
        assert jnp.allclose(pre_activation_penalty(jnp.array([[3.0]])), 4.0)
        assert jnp.allclose(pre_activation_penalty(jnp.array([[-3.0]])), 4.0)

    def test_scales_with_action_dim_not_averaged_over_it(self):
        """The reduction is sum-over-dims, mean-over-batch — NOT a plain mean.

        Averaging over action dims silently divided `pre_activation_coef` by
        action_dim (56 on the CMU humanoid), which is what made the hinge inert
        in the CMU_006_13 run while still looking configured. Guard it: the
        per-logit gradient must not depend on how many logits there are.
        """
        one = jnp.array([[3.0]])
        many = jnp.full((1, 56), 3.0)
        assert jnp.allclose(pre_activation_penalty(many), 56.0 * 4.0)
        g_one = jax.grad(pre_activation_penalty)(one)
        g_many = jax.grad(pre_activation_penalty)(many)
        assert jnp.allclose(g_one[0, 0], g_many[0, 0])

    def test_batch_is_averaged_not_summed(self):
        """Batch size must not change the penalty's weight against the DPG term
        (which is itself a batch mean)."""
        small = jnp.full((4, 3), 3.0)
        large = jnp.full((512, 3), 3.0)
        assert jnp.allclose(pre_activation_penalty(small), pre_activation_penalty(large))

    def test_gradient_survives_tanh_saturation(self):
        """The whole point: a live gradient where -dQ/du has underflowed.

        At |u| = 12 the tanh derivative is ~1e-10, so the DPG term can no
        longer move the policy; the penalty's gradient is linear in the
        overshoot and still pulls inward.
        """
        u = jnp.array([[12.0, -12.0]])
        assert jnp.mean(1.0 - jnp.tanh(u) ** 2) < 1e-9
        grad = jax.grad(pre_activation_penalty)(u)
        # Points back toward zero, with real magnitude.
        assert jnp.all(grad * jnp.sign(u) > 0)
        assert jnp.min(jnp.abs(grad)) > 1.0


class TestTD3ActorLoss:
    def _call(self, actor, critic, samples, bounds, coef):
        """Returns the full `(loss, aux)` pair the agent differentiates with
        `has_aux=True`."""
        return td3_actor_loss_fn(
            actor,
            critic,
            samples,
            bounds["action_low"],
            bounds["action_high"],
            coef,
        )

    def _loss(self, actor, critic, samples, bounds, coef):
        return self._call(actor, critic, samples, bounds, coef)[0]

    def test_zero_coef_is_plain_dpg(
        self, det_actor, twin_critic, ddpg_samples, action_bounds
    ):
        loss = self._loss(det_actor, twin_critic, ddpg_samples, action_bounds, 0.0)
        obs = ddpg_samples["observations"]
        q1, _ = twin_critic(obs, det_actor(obs))
        assert loss.shape == ()
        assert jnp.allclose(loss, -jnp.mean(q1), atol=1e-5)

    def test_penalty_only_charges_when_saturated(
        self, det_actor, twin_critic, ddpg_samples, action_bounds
    ):
        """At the default (small) init the logits are inside the threshold, so
        the penalty is inert; it must bite once the logits are driven out."""
        base = self._loss(det_actor, twin_critic, ddpg_samples, action_bounds, 0.0)
        unsaturated = self._loss(
            det_actor, twin_critic, ddpg_samples, action_bounds, 1e-2
        )
        assert jnp.allclose(base, unsaturated, atol=1e-6)

        saturated = copy.deepcopy(det_actor)
        params = nnx.state(saturated, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(saturated, params)
        with_penalty = self._loss(
            saturated, twin_critic, ddpg_samples, action_bounds, 1e-2
        )
        without = self._loss(saturated, twin_critic, ddpg_samples, action_bounds, 0.0)
        assert with_penalty > without

    def test_gradient_pulls_a_saturated_actor_back(
        self, det_actor, twin_critic, ddpg_samples, action_bounds
    ):
        """Regression guard for the CMU_006_13 collapse: a saturated actor must
        still receive a gradient that reduces |pre-activation|."""
        actor = copy.deepcopy(det_actor)
        params = nnx.state(actor, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(actor, params)

        obs = ddpg_samples["observations"]
        before = jnp.mean(jnp.abs(actor.forward(obs)[1]))
        assert before > 5.0  # genuinely saturated to start with
        # ...and deep enough into tanh's flat region that the DPG term is
        # heavily attenuated (a healthy actor sits around 0.6-0.7 here).
        assert jnp.mean(1.0 - actor(obs) ** 2) < 0.1

        # Same call shape as TD3._grad_step: differentiate w.r.t. arg 0 with the
        # critic passed explicitly, so nnx owns its (dropout) rng state.
        optimizer = nnx.Optimizer(actor, optax.adam(1e-2), wrt=nnx.Param)
        for _ in range(50):
            grads, _aux = nnx.grad(td3_actor_loss_fn, has_aux=True)(
                actor,
                twin_critic,
                ddpg_samples,
                action_bounds["action_low"],
                action_bounds["action_high"],
                1e-2,
            )
            optimizer.update(actor, grads)

        after = jnp.mean(jnp.abs(actor.forward(obs)[1]))
        assert after < before


class TestTD3ActorDiagnostics:
    """The `td3/` saturation metrics must move BEFORE the score does — that is
    the whole reason they exist, so pin their direction."""

    def _aux(self, actor, critic, samples, bounds):
        return TestTD3ActorLoss()._call(actor, critic, samples, bounds, 1e-2)[1]

    def test_healthy_actor_reports_live_gradient(
        self, det_actor, twin_critic, ddpg_samples, action_bounds
    ):
        aux = self._aux(det_actor, twin_critic, ddpg_samples, action_bounds)
        # Small output init keeps the logits in tanh's linear region.
        assert float(aux["pre_act_abs"]) < 1.0
        assert float(aux["tanh_grad"]) > 0.5
        assert float(aux["sat_frac"]) < 0.05
        assert float(aux["pre_act_penalty"]) == 0.0

    def test_saturated_actor_is_flagged(
        self, det_actor, twin_critic, ddpg_samples, action_bounds
    ):
        actor = copy.deepcopy(det_actor)
        params = nnx.state(actor, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(actor, params)

        healthy = self._aux(det_actor, twin_critic, ddpg_samples, action_bounds)
        saturated = self._aux(actor, twin_critic, ddpg_samples, action_bounds)
        assert saturated["pre_act_abs"] > healthy["pre_act_abs"]
        assert saturated["pre_act_max"] >= saturated["pre_act_abs"]
        # Healthy is <0.05 (asserted above), so 0.5 separates the two regimes
        # with room to spare without pinning the fixture's exact geometry.
        assert float(saturated["sat_frac"]) > 0.5
        # The dead-gradient signature: this is the number to watch in the logs.
        assert float(saturated["tanh_grad"]) < 0.05
        assert float(saturated["pre_act_penalty"]) > 0.0


class TestDDPGCriticLoss:
    def test_returns_scalar(self, det_critic, det_actor, ddpg_samples, action_bounds):
        target_actor = copy.deepcopy(det_actor)
        target_critic = copy.deepcopy(det_critic)
        key = jax.random.PRNGKey(0)
        loss, _aux = ddpg_critic_loss_fn(
            det_critic,
            target_actor,
            target_critic,
            ddpg_samples,
            key,
            0.1,
            0.1,
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == ()

    def test_non_negative(self, det_critic, det_actor, ddpg_samples, action_bounds):
        target_actor = copy.deepcopy(det_actor)
        target_critic = copy.deepcopy(det_critic)
        key = jax.random.PRNGKey(0)
        loss, _aux = ddpg_critic_loss_fn(
            det_critic,
            target_actor,
            target_critic,
            ddpg_samples,
            key,
            0.1,
            0.1,
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss >= 0.0


class TestPPOLoss:
    @pytest.fixture
    def ppo_data(self, stoch_actor, stoch_critic):
        key = jax.random.PRNGKey(0)
        num_envs, seq_len = 4, 10
        obs = jax.random.normal(key, (num_envs, seq_len, OBS_DIM))
        dist = stoch_actor(obs)
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        values = stoch_critic(obs).squeeze(-1)
        advantages = jax.random.normal(key, (num_envs, seq_len - 1))
        return obs, actions, log_probs, values, advantages

    def _ppo_actor_loss(self, actor, obs, actions, log_probs, advantages, **kw):
        return ppo_loss_fn(
            actor,
            obs,
            actions,
            log_probs,
            jnp.full(ACT_DIM, -1.0),
            jnp.full(ACT_DIM, 1.0),
            advantages,
            clip_epsilon=kw.get("clip_epsilon", 0.2),
            entropy_coef=kw.get("entropy_coef", 0.01),
            key=jax.random.PRNGKey(0),
        )

    def test_actor_loss_scalar(self, stoch_actor, ppo_data):
        obs, actions, log_probs, values, advantages = ppo_data
        loss, (approx_kl, clip_frac) = self._ppo_actor_loss(
            stoch_actor, obs, actions, log_probs, advantages
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert approx_kl.shape == () and clip_frac.shape == ()

    def test_trust_region_diagnostics_are_zero_on_first_pass(self, stoch_actor, ppo_data):
        """`log_probs` in the fixture come from this very actor, so the ratio is
        exactly 1: approx_kl and clip_frac must both be 0. This is the property
        that makes them a drift measurement -- anything non-zero on pass 1 means
        the policy already moved (or the observations were normalized
        differently) between acting and learning.
        """
        obs, actions, log_probs, values, advantages = ppo_data
        _, (approx_kl, clip_frac) = self._ppo_actor_loss(
            stoch_actor, obs, actions, log_probs, advantages
        )
        assert float(approx_kl) == pytest.approx(0.0, abs=1e-6)
        assert float(clip_frac) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("stale_log_probs", [-1e4, -1e6, 1e4, 1e6])
    def test_a_diverged_ratio_cannot_produce_a_nan_gradient(
        self, stoch_actor, ppo_data, stale_log_probs
    ):
        """The AcrobotSwingup/warp_gpu release run died here.

        `old_log_probs` far from `logp_new` is not hypothetical for a squashed
        policy: `TanhNormal.log_prob` pins a saturated sample's `u` at the
        arctanh rail and divides the z-score by a `std` free to fall to
        `std_min`, so log-probs of order 1e4+ are ordinary and their DIFFERENCES
        pass `exp`'s float32 overflow at 88 easily.

        The forward loss is no witness -- the clip caps it at a healthy-looking
        `(1 + clip_eps) * advantage` -- so this asserts on the GRADIENT, which
        is where `jnp.minimum`'s zero cotangent used to meet an `inf` ratio and
        produce `0 * inf = NaN`. One NaN element is terminal downstream:
        `clip_by_global_norm` rescales by 1 / global_norm and spreads it over
        every parameter in the tree, with no path back.
        """
        obs, actions, log_probs, _values, advantages = ppo_data
        diverged = jnp.full_like(log_probs, stale_log_probs)

        (loss, (approx_kl, clip_frac)), grads = nnx.value_and_grad(
            lambda m: self._ppo_actor_loss(m, obs, actions, diverged, advantages),
            has_aux=True,
        )(stoch_actor)

        assert jnp.isfinite(loss)
        for leaf in jax.tree.leaves(grads):
            assert jnp.isfinite(leaf).all()

        # And the trust region must still be able to see the divergence:
        # `NaN > target_kl` is False, which is how the early stop switched
        # itself off for the last 487M steps of that run.
        assert jnp.isfinite(approx_kl) and approx_kl > 1.0
        assert float(clip_frac) == pytest.approx(1.0)

    def test_critic_loss_scalar(self, stoch_critic, ppo_data):
        obs, actions, log_probs, values, advantages = ppo_data
        returns = values[:, :-1] + advantages
        loss = ppo_critic_loss_fn(stoch_critic, obs, returns)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_critic_loss_reaches_zero_on_own_prediction(self, stoch_critic, ppo_data):
        """A plain regression target: handed the critic's own output back, the
        loss must be ~0. The previous target (`V_old + normalized_advantage`)
        could not do this -- it floored at var(normalized adv) ~ 1.0 regardless
        of the critic, which is exactly what every training run logged.
        """
        obs, _, _, _, _ = ppo_data
        returns = stoch_critic(obs)[:, :-1, 0]
        assert ppo_critic_loss_fn(stoch_critic, obs, returns) < 1e-6


class TestSACLosses:
    def test_actor_loss(self, stoch_actor, twin_critic, ddpg_samples, action_bounds):
        key = jax.random.PRNGKey(0)
        alpha = 0.2
        loss, (log_probs, _aux) = sac_actor_loss_fn(
            stoch_actor,
            twin_critic,
            alpha,
            ddpg_samples,
            key,
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert log_probs.shape == (BATCH,)

    def test_critic_loss(self, stoch_actor, twin_critic, ddpg_samples, action_bounds):
        target_twin = copy.deepcopy(twin_critic)
        key = jax.random.PRNGKey(0)
        loss, _aux = sac_critic_loss_fn(
            twin_critic,
            stoch_actor,
            target_twin,
            ddpg_samples,
            0.2,
            key,
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_alpha_loss(self):
        log_alpha = LogAlpha(init_value=0.0)
        log_probs = jnp.array([-1.0, -2.0, -0.5])
        target_entropy = -3.0
        loss = sac_alpha_loss_fn(log_alpha, log_probs, target_entropy)
        assert loss.shape == ()
        assert jnp.isfinite(loss)


class TestTargetSmoothingUnits:
    """`target_policy_noise` / `target_noise_clip` live in the actor's [-1, 1]
    output space — the units the TD3 paper defines them in, and the ones
    `NoiseModule.add_noise` already explores in. Scaling them by the env action
    span instead silently doubles both on any [-1, 1] env, which is what made
    TD3's smoothing kernel 4x its own exploration noise.
    """

    N = 4096

    def _bounds(self, span):
        return jnp.full((ACT_DIM,), -span / 2), jnp.full((ACT_DIM,), span / 2)

    def _applied_noise(self, actor, obs, low, high, sigma, clip, seed=1):
        """The smoothing actually applied, expressed back in [-1, 1] units."""
        smoothed, _ = _smoothed_target_actions(
            actor, obs, jax.random.PRNGKey(seed), sigma, clip, low, high
        )
        # `scale_to_env` is affine and increasing, so it inverts exactly.
        smoothed_unit = 2.0 * (smoothed - low) / (high - low) - 1.0
        return smoothed_unit - actor(obs)

    @pytest.mark.parametrize("span", [2.0, 20.0])
    def test_sigma_is_span_independent(self, det_actor, span):
        obs = jax.random.normal(jax.random.PRNGKey(0), (self.N, OBS_DIM))
        low, high = self._bounds(span)
        # Clip deliberately wide enough to be inert, so this measures sigma alone.
        applied = self._applied_noise(det_actor, obs, low, high, 0.2, 10.0)
        assert float(jnp.std(applied)) == pytest.approx(0.2, rel=0.05)

    @pytest.mark.parametrize("span", [2.0, 20.0])
    def test_clip_is_span_independent(self, det_actor, span):
        obs = jax.random.normal(jax.random.PRNGKey(0), (self.N, OBS_DIM))
        low, high = self._bounds(span)
        # Sigma >> clip, so essentially every sample is pinned to the clip.
        applied = self._applied_noise(det_actor, obs, low, high, 1.0, 0.1)
        assert float(jnp.max(jnp.abs(applied))) <= 0.1 + 1e-4

    def test_zero_noise_is_a_no_op(self, det_actor):
        """DDPG and D4PG run this same helper with the noise off; it must return
        the bare target action, not merely a small perturbation of it."""
        obs = jax.random.normal(jax.random.PRNGKey(0), (BATCH, OBS_DIM))
        low, high = self._bounds(2.0)
        smoothed, clip_frac = _smoothed_target_actions(
            det_actor, obs, jax.random.PRNGKey(1), 0.0, 0.0, low, high
        )
        expected = Agent.scale_to_env(det_actor(obs), low, high)
        assert jnp.allclose(smoothed, expected, atol=1e-6)
        assert float(clip_frac) == 0.0
