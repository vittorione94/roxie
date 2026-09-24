import copy

import jax
import jax.numpy as jnp
import optax
import pytest
from flax import nnx

# Import agents first to avoid circular import
import roxie.agents  # noqa: F401
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
    wasserstein_blend_logits,
)
from roxie.models.actors import DeterministicActor, StochasticActor
from roxie.models.critics import QCritic, VCritic, TwinCritic
from roxie.utils.math import normalize_samples, scale_to_env


OBS_DIM = 8
ACT_DIM = 3
BATCH = 16
FEATURES = [32, 32]


# Module-scoped on purpose: these are read-only here — every test that perturbs
# a network `deepcopy`s it first — and rebuilding a network per test is what a
# loss-function file spends its time on otherwise.


@pytest.fixture(scope="module")
def det_actor():
    rngs = nnx.Rngs(params=0, dropout=1)
    return DeterministicActor(
        in_features=OBS_DIM, features=FEATURES, action_dim=ACT_DIM, rngs=rngs
    )


@pytest.fixture(scope="module")
def stoch_actor():
    rngs = nnx.Rngs(params=0, dropout=1)
    return StochasticActor(
        in_features=OBS_DIM, features=FEATURES, action_dim=ACT_DIM, rngs=rngs
    )


@pytest.fixture(scope="module")
def det_critic():
    rngs = nnx.Rngs(params=0, dropout=1)
    return QCritic(
        in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs
    )


@pytest.fixture(scope="module")
def stoch_critic():
    rngs = nnx.Rngs(params=0, dropout=1)
    return VCritic(in_features=OBS_DIM, features=FEATURES, rngs=rngs)


@pytest.fixture(scope="module")
def twin_critic():
    rngs1 = nnx.Rngs(params=0, dropout=1)
    rngs2 = nnx.Rngs(params=2, dropout=3)
    c1 = QCritic(in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs1)
    c2 = QCritic(in_features=OBS_DIM + ACT_DIM, features=FEATURES, rngs=rngs2)
    return TwinCritic(c1, c2)


@pytest.fixture(scope="module")
def ddpg_samples():
    """A repacked batch as the losses see it: observations ALREADY normalized.

    Every loss function takes them that way — the agent runs the batch through
    `normalize_samples` once per gradient step (see `TestSampleNormalization`),
    so no loss takes obs_mean/obs_std/obs_clip arguments.
    """
    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)
    return {
        "observations": jax.random.normal(k1, (BATCH, OBS_DIM), dtype=jnp.float32),
        "actions": jax.random.normal(k2, (BATCH, ACT_DIM), dtype=jnp.float32),
        "rewards": jax.random.normal(k3, (BATCH,), dtype=jnp.float32),
        "next_observations": jax.random.normal(k4, (BATCH, OBS_DIM), dtype=jnp.float32),
        # Per-sample bootstrap coefficient (gamma^b, 0 at terminals) — every
        # off-policy critic loss consumes this instead of gamma + terminal
        # flags. `terminals` is kept for MPO, which still reads it directly.
        "bootstrap": 0.99 * jnp.ones(BATCH, dtype=jnp.float32),
        "terminals": jnp.zeros(BATCH, dtype=jnp.bool_),
    }


@pytest.fixture(scope="module")
def action_bounds():
    return {
        "action_low": jnp.full(ACT_DIM, -1.0, dtype=jnp.float32),
        "action_high": jnp.full(ACT_DIM, 1.0, dtype=jnp.float32),
    }


class TestSampleNormalization:
    """`normalize_samples` is the ONE place observations are normalized on
    the learning path, so every loss can take them pre-normalized."""

    def test_normalizes_both_observation_entries(self, ddpg_samples):
        mean = jnp.full(OBS_DIM, 2.0, dtype=jnp.float32)
        std = jnp.full(OBS_DIM, 4.0, dtype=jnp.float32)
        out = normalize_samples(ddpg_samples, mean, std, clip=5.0)
        for k in ("observations", "next_observations"):
            assert jnp.allclose(out[k], (ddpg_samples[k] - mean) / std)

    def test_leaves_the_rest_of_the_batch_alone(self, ddpg_samples):
        out = normalize_samples(
            ddpg_samples, jnp.zeros(
                OBS_DIM, dtype=jnp.float32
            ), jnp.ones(OBS_DIM, dtype=jnp.float32), clip=5.0
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
        out = normalize_samples(
            raw, jnp.zeros(
                OBS_DIM, dtype=jnp.float32
            ), jnp.ones(OBS_DIM, dtype=jnp.float32), clip=5.0, enabled=False
        )
        assert jnp.array_equal(out["observations"], raw["observations"])
        assert jnp.max(jnp.abs(out["observations"])) > 5.0


class TestDDPGActorLoss:
    def test_is_a_finite_scalar_with_the_declared_diagnostics(
        self, det_actor, det_critic, ddpg_samples, action_bounds
    ):
        loss, aux = ddpg_actor_loss_fn(
            det_actor,
            det_critic,
            ddpg_samples,
            action_bounds["action_low"],
            action_bounds["action_high"],
            0.0,
        )
        assert loss.shape == () and jnp.isfinite(loss)
        assert set(aux) == {*ACTOR_DIAGNOSTIC_KEYS, "pre_act_penalty"}


class TestPreActivationPenalty:
    """The counterweight to DPG's unbounded outward push on the actor logits."""

    def test_free_inside_the_threshold(self):
        u = jnp.array([[-1.0, -0.4, 0.0, 0.9, 1.0]], dtype=jnp.float32)
        assert pre_activation_penalty(u) == 0.0

    def test_grows_quadratically_in_the_overshoot(self):
        # relu(|u| - 1)^2 over a single element.
        assert jnp.allclose(pre_activation_penalty(jnp.array(
            [[3.0]], dtype=jnp.float32
        )), 4.0)
        assert jnp.allclose(pre_activation_penalty(jnp.array(
            [[-3.0]], dtype=jnp.float32
        )), 4.0)

    def test_scales_with_action_dim_not_averaged_over_it(self):
        """The reduction is sum-over-dims, mean-over-batch — NOT a plain mean.

        Averaging over action dims silently divided `pre_activation_coef` by
        action_dim (56 on the CMU humanoid), which is what made the hinge inert
        in the CMU_006_13 run while still looking configured. Guard it: the
        per-logit gradient must not depend on how many logits there are.
        """
        one = jnp.array([[3.0]], dtype=jnp.float32)
        many = jnp.full((1, 56), 3.0, dtype=jnp.float32)
        assert jnp.allclose(pre_activation_penalty(many), 56.0 * 4.0)
        g_one = jax.grad(pre_activation_penalty)(one)
        g_many = jax.grad(pre_activation_penalty)(many)
        assert jnp.allclose(g_one[0, 0], g_many[0, 0])

    def test_batch_is_averaged_not_summed(self):
        """Batch size must not change the penalty's weight against the DPG term
        (which is itself a batch mean)."""
        small = jnp.full((4, 3), 3.0, dtype=jnp.float32)
        large = jnp.full((512, 3), 3.0, dtype=jnp.float32)
        assert jnp.allclose(pre_activation_penalty(small), pre_activation_penalty(large))

    def test_gradient_survives_tanh_saturation(self):
        """The whole point: a live gradient where -dQ/du has underflowed.

        At |u| = 12 the tanh derivative is ~1e-10, so the DPG term can no
        longer move the policy; the penalty's gradient is linear in the
        overshoot and still pulls inward.
        """
        u = jnp.array([[12.0, -12.0]], dtype=jnp.float32)
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
        params["output_layer"]["kernel"][...] *= 200.0
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
        params["output_layer"]["kernel"][...] *= 200.0
        nnx.update(actor, params)

        obs = ddpg_samples["observations"]
        before = jnp.mean(jnp.abs(actor.forward(obs)[1]))
        assert before > 5.0  # genuinely saturated to start with
        # ...and deep enough into tanh's flat region that the DPG term is
        # heavily attenuated (a healthy actor sits around 0.6-0.7 here).
        assert jnp.mean(1.0 - actor(obs) ** 2) < 0.1

        # Same call shape as TD3._td3_grad_step: differentiate w.r.t. arg 0 with the
        # critic passed explicitly, so nnx owns its (dropout) rng state.
        optimizer = nnx.Optimizer(actor, optax.adam(1e-1), wrt=nnx.Param)
        for _ in range(10):
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
        params["output_layer"]["kernel"][...] *= 200.0
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
    def test_is_a_non_negative_scalar(
        self, det_critic, det_actor, ddpg_samples, action_bounds
    ):
        """A squared TD error: scalar, finite and never below zero."""
        loss, _aux = ddpg_critic_loss_fn(
            det_critic,
            copy.deepcopy(det_actor),
            copy.deepcopy(det_critic),
            ddpg_samples,
            jax.random.PRNGKey(0),
            0.1,
            0.1,
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == () and jnp.isfinite(loss) and loss >= 0.0


class TestPPOLoss:
    @pytest.fixture(scope="module")
    def ppo_data(self, stoch_actor, stoch_critic):
        key = jax.random.PRNGKey(0)
        # Flat over `(env, time)`, as `_prepare_rollout` hands it over: the
        # slicing to the GAE horizon and the flattening both happen there, so
        # the losses see one transition axis and nothing else.
        num_transitions = 36
        obs = jax.random.normal(key, (num_transitions, OBS_DIM), dtype=jnp.float32)
        dist = stoch_actor(obs)
        # As the agent stores them: the pre-tanh draw and the density scored
        # from it, never the squashed action re-inverted.
        _actions, pre_actions = dist.sample_from_pre(seed=key)
        log_probs = dist.log_prob_from_pre(pre_actions)
        values = stoch_critic(obs).squeeze(-1)
        advantages = jax.random.normal(key, (num_transitions,), dtype=jnp.float32)
        return obs, pre_actions, log_probs, values, advantages

    def _ppo_actor_loss(self, actor, obs, pre_actions, log_probs, advantages, **kw):
        return ppo_loss_fn(
            actor,
            obs,
            pre_actions,
            log_probs,
            advantages,
            clip_epsilon=kw.get("clip_epsilon", 0.2),
            entropy_coef=kw.get("entropy_coef", 0.01),
            key=jax.random.PRNGKey(0),
        )

    def test_actor_loss_scalar(self, stoch_actor, ppo_data):
        obs, pre_actions, log_probs, values, advantages = ppo_data
        loss, aux = self._ppo_actor_loss(
            stoch_actor, obs, pre_actions, log_probs, advantages
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert aux["approx_kl"].shape == () and aux["clip_frac"].shape == ()

    def test_trust_region_diagnostics_are_zero_on_first_pass(self, stoch_actor, ppo_data):
        """`log_probs` in the fixture come from this very actor, so the ratio is
        exactly 1: approx_kl and clip_frac must both be 0. This is the property
        that makes them a drift measurement -- anything non-zero on pass 1 means
        the policy already moved (or the observations were normalized
        differently) between acting and learning.
        """
        obs, pre_actions, log_probs, values, advantages = ppo_data
        _, aux = self._ppo_actor_loss(
            stoch_actor, obs, pre_actions, log_probs, advantages
        )
        assert float(aux["approx_kl"]) == pytest.approx(0.0, abs=1e-6)
        assert float(aux["clip_frac"]) == pytest.approx(0.0, abs=1e-6)

    @pytest.mark.parametrize("stale_log_probs", [-1e6, 1e6])
    def test_a_diverged_ratio_cannot_produce_a_nan_gradient(
        self, stoch_actor, ppo_data, stale_log_probs
    ):
        """The AcrobotSwingup/warp_gpu release run died here.

        `old_log_probs` far from `logp_new` is not hypothetical for a squashed
        policy: the z-score is divided by a `std` free to fall to `std_min`, so
        log-probs of order 1e4+ are ordinary and their DIFFERENCES pass `exp`'s
        float32 overflow at 88 easily. Scoring from the stored pre-tanh draw
        (`TestRatioSurvivesSaturation`) is what stops the loss REACHING that
        regime; this stays as the backstop for when it gets there anyway.

        The forward loss is no witness -- the clip caps it at a healthy-looking
        `(1 + clip_eps) * advantage` -- so this asserts on the GRADIENT, which
        is where `jnp.minimum`'s zero cotangent used to meet an `inf` ratio and
        produce `0 * inf = NaN`. One NaN element is terminal downstream:
        `clip_by_global_norm` rescales by 1 / global_norm and spreads it over
        every parameter in the tree, with no path back.
        """
        obs, pre_actions, log_probs, _values, advantages = ppo_data
        diverged = jnp.full_like(log_probs, stale_log_probs)

        (loss, aux), grads = nnx.value_and_grad(
            lambda m: self._ppo_actor_loss(m, obs, pre_actions, diverged, advantages),
            has_aux=True,
        )(stoch_actor)

        assert jnp.isfinite(loss)
        for leaf in jax.tree.leaves(grads):
            assert jnp.isfinite(leaf).all()

        # And the trust region must still be able to see the divergence:
        # `NaN > target_kl` is False, which is how the early stop switched
        # itself off for the last 487M steps of that run.
        assert jnp.isfinite(aux["approx_kl"]) and aux["approx_kl"] > 1.0
        assert float(aux["clip_frac"]) == pytest.approx(1.0)

    def test_critic_loss_scalar(self, stoch_critic, ppo_data):
        obs, pre_actions, log_probs, values, advantages = ppo_data
        returns = values + advantages
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
        returns = stoch_critic(obs)[..., 0]
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
            1.0,
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
            1.0,
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0

    def test_alpha_loss(self):
        log_alpha = LogAlpha(init_value=0.0)
        log_probs = jnp.array([-1.0, -2.0, -0.5], dtype=jnp.float32)
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
        obs = jax.random.normal(
            jax.random.PRNGKey(0), (self.N, OBS_DIM), dtype=jnp.float32
        )
        low, high = self._bounds(span)
        # Clip deliberately wide enough to be inert, so this measures sigma alone.
        applied = self._applied_noise(det_actor, obs, low, high, 0.2, 10.0)
        assert float(jnp.std(applied)) == pytest.approx(0.2, rel=0.05)

    @pytest.mark.parametrize("span", [2.0, 20.0])
    def test_clip_is_span_independent(self, det_actor, span):
        obs = jax.random.normal(
            jax.random.PRNGKey(0), (self.N, OBS_DIM), dtype=jnp.float32
        )
        low, high = self._bounds(span)
        # Sigma >> clip, so essentially every sample is pinned to the clip.
        applied = self._applied_noise(det_actor, obs, low, high, 1.0, 0.1)
        assert float(jnp.max(jnp.abs(applied))) <= 0.1 + 1e-4

    def test_zero_noise_is_a_no_op(self, det_actor):
        """DDPG and D4PG run this same helper with the noise off; it must return
        the bare target action, not merely a small perturbation of it."""
        obs = jax.random.normal(
            jax.random.PRNGKey(0), (BATCH, OBS_DIM), dtype=jnp.float32
        )
        low, high = self._bounds(2.0)
        smoothed, clip_frac = _smoothed_target_actions(
            det_actor, obs, jax.random.PRNGKey(1), 0.0, 0.0, low, high
        )
        expected = scale_to_env(det_actor(obs), low, high)
        assert jnp.allclose(smoothed, expected, atol=1e-6)
        assert float(clip_frac) == 0.0


class TestRatioSurvivesSaturation:
    """The AcrobotSwingup/AcrobotSwingupSparse release runs died here.

    `tanh` is not invertible in float32: it rounds to exactly 1.0 for |u| >= 8,
    so `TanhNormal.log_prob` -- which recovers `u` with an arctanh clipped at
    1 - 1e-6 -- maps every saturated draw onto the same rail, u ~ 7.2477.
    Nothing bounds the pre-tanh mean, so once it walks past that rail the stored
    density is evaluated at a FIXED point in the far tail and its sensitivity to
    the parameters is set by how far the mean has drifted, not by how far the
    policy actually moved.

    In the release runs that put `approx_kl` at ~1e8 (the ratio pinned at
    exp(_MAX_LOG_RATIO)) from 6.6M steps onward: `target_kl` then tripped on the
    first minibatch of every single rollout, and PPO ran the remaining 493M
    steps at 1 gradient step per rollout against a configured 64.

    The fix is to score both sides from the stored pre-tanh draw. These tests
    pin the property that makes it work -- the ratio tracks the policy, not the
    saturation.
    """

    # Past the arctanh rail, where the old path lost `u` entirely.
    SATURATED_MEAN = 20.0
    # One Adam step at the benchmark's actor_learning_rate.
    STEP = 3e-4

    @pytest.fixture(scope="module")
    def saturated(self, stoch_actor):
        """An actor forced to emit a pre-tanh mean well past the rail."""
        actor = copy.deepcopy(stoch_actor)
        actor.output_layer.kernel[...] = jnp.zeros_like(actor.output_layer.kernel[...])
        actor.output_layer.bias[...] = jnp.full_like(
            actor.output_layer.bias[...], self.SATURATED_MEAN
        )
        return actor

    # Flat over `(env, time)`, as the losses now take it.
    TRANSITIONS = 40

    def _rollout(self, actor, key):
        obs = jax.random.normal(key, (self.TRANSITIONS, OBS_DIM), dtype=jnp.float32)
        dist = actor(obs)
        _, pre_actions = dist.sample_from_pre(seed=key)
        return obs, pre_actions, dist.log_prob_from_pre(pre_actions)

    def _kl(self, actor, obs, pre_actions, old_log_probs):
        _, aux = ppo_loss_fn(
            actor, obs, pre_actions, old_log_probs,
            jax.random.normal(
                jax.random.PRNGKey(1), (self.TRANSITIONS,), dtype=jnp.float32
            ),
            clip_epsilon=0.2, entropy_coef=0.01, key=jax.random.PRNGKey(0),
        )
        return float(aux["approx_kl"])

    def test_the_fixture_really_saturates(self, saturated):
        """Guards every assertion below against passing vacuously: if the mean
        stayed inside the rail there would be no pathology to regress on."""
        obs = jax.random.normal(
            jax.random.PRNGKey(0), (self.TRANSITIONS, OBS_DIM), dtype=jnp.float32
        )
        dist = saturated(obs)
        _, pre_actions = dist.sample_from_pre(seed=jax.random.PRNGKey(0))
        assert float(jnp.min(jnp.abs(pre_actions))) > 8.0
        # ... and that the squash really has destroyed `u` at that magnitude.
        assert float(jnp.max(jnp.abs(jnp.tanh(pre_actions)))) == 1.0

    def test_first_pass_ratio_is_one_when_saturated(self, saturated):
        obs, pre_actions, log_probs = self._rollout(
            saturated, jax.random.PRNGKey(0)
        )
        assert self._kl(saturated, obs, pre_actions, log_probs) == pytest.approx(
            0.0, abs=1e-6
        )

    def test_kl_tracks_the_policy_step_not_the_saturation(self, saturated):
        """The regression proper.

        A single optimizer-sized step on the mean must cost a KL the trust
        region can budget with. Scored through the arctanh instead, the same
        step moved the density by ~5e4 nats, `approx_kl` came out at
        exp(_MAX_LOG_RATIO) ~ 4.85e8, and `target_kl` tripped immediately.
        """
        obs, pre_actions, log_probs = self._rollout(
            saturated, jax.random.PRNGKey(0)
        )
        stepped = copy.deepcopy(saturated)
        stepped.output_layer.bias[...] = saturated.output_layer.bias[...] + self.STEP

        kl = self._kl(stepped, obs, pre_actions, log_probs)
        assert 0.0 < kl < 1e-3, f"one {self.STEP} step cost approx_kl {kl}"

    def test_kl_is_independent_of_how_far_past_the_rail_the_mean_sits(
        self, stoch_actor
    ):
        """The same step must cost the same KL at any saturation level.

        This is the property the arctanh path could not have: there the cost
        grew with (mean - 7.2477), so a policy that kept drifting kept raising
        its own KL without moving any further per step.
        """
        kls = []
        for mean in (1.0, 8.0, 40.0):
            actor = copy.deepcopy(stoch_actor)
            actor.output_layer.kernel[...] = jnp.zeros_like(
                actor.output_layer.kernel[...]
            )
            actor.output_layer.bias[...] = jnp.full_like(
                actor.output_layer.bias[...], mean
            )
            obs, pre_actions, log_probs = self._rollout(
                actor, jax.random.PRNGKey(0)
            )
            stepped = copy.deepcopy(actor)
            stepped.output_layer.bias[...] = (
                actor.output_layer.bias[...] + self.STEP
            )
            kls.append(self._kl(stepped, obs, pre_actions, log_probs))

        assert max(kls) == pytest.approx(min(kls), rel=0.1), kls


class TestWassersteinBlend:
    """`twin_q_weight` on TD4 blends two categorical critics.

    Mixing their densities is the obvious implementation and the wrong one: it
    adds a mode and inflates the spread, which compounds over Bellman backups.
    The release grid caught it — TD4 escaped HumanoidStand at `w=1.0` and never
    escaped at `w=0.5` (2026-09-18).
    """

    ATOMS = jnp.linspace(0.0, 50.0, 51)

    def _spike(self, index):
        return jnp.where(jnp.arange(51) == index, 20.0, -20.0)[None, :]

    def _mean_std(self, logits):
        probs = jax.nn.softmax(logits, axis=-1)[0]
        mean = jnp.sum(probs * self.ATOMS)
        return mean, jnp.sqrt(jnp.sum(probs * (self.ATOMS - mean) ** 2))

    def test_blending_two_spikes_gives_one_spike_between_them(self):
        blended = wasserstein_blend_logits(
            self._spike(10), self._spike(30), 0.5, self.ATOMS
        )
        mean, std = self._mean_std(blended)
        assert float(mean) == pytest.approx(20.0, abs=1e-3)
        # A density mixture would put half the mass at each spike: std 10.
        assert float(std) == pytest.approx(0.0, abs=1e-3)

    def test_the_weight_sweeps_the_mean_between_the_two_heads(self):
        means = [
            float(self._mean_std(
                wasserstein_blend_logits(
                    self._spike(10), self._spike(30), w, self.ATOMS
                )
            )[0])
            for w in (1.0, 0.75, 0.5, 0.25, 0.0)
        ]
        assert means == pytest.approx([10.0, 15.0, 20.0, 25.0, 30.0], abs=1e-3)

    def test_spread_is_preserved_rather_than_inflated(self):
        def gaussian(centre):
            return jnp.log(
                jax.nn.softmax(-((jnp.arange(51) - centre) ** 2) / 40.0)[None, :]
                + 1e-12
            )

        _, spread = self._mean_std(gaussian(15.0))
        _, blended = self._mean_std(
            wasserstein_blend_logits(gaussian(15.0), gaussian(35.0), 0.5, self.ATOMS)
        )
        assert float(blended) == pytest.approx(float(spread), rel=0.05)
