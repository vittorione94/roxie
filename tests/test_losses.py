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
    ddpg_actor_loss_fn,
    ppo_loss_fn,
    pre_activation_penalty,
    sac_actor_loss_fn,
    sac_alpha_loss_fn,
    td3_actor_loss_fn,
)
from roxie.losses.critic_losses import ddpg_critic_loss_fn, ppo_critic_loss_fn, sac_critic_loss_fn
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
def norm_params():
    return {
        "obs_mean": jnp.zeros(OBS_DIM),
        "obs_std": jnp.ones(OBS_DIM),
        "obs_clip": 5.0,
    }


@pytest.fixture
def action_bounds():
    return {
        "action_low": jnp.full(ACT_DIM, -1.0),
        "action_high": jnp.full(ACT_DIM, 1.0),
    }


class TestDDPGActorLoss:
    def test_returns_scalar(self, det_actor, det_critic, ddpg_samples, norm_params, action_bounds):
        loss = ddpg_actor_loss_fn(
            det_actor,
            det_critic,
            ddpg_samples,
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == ()

    def test_finite(self, det_actor, det_critic, ddpg_samples, norm_params, action_bounds):
        loss = ddpg_actor_loss_fn(
            det_actor,
            det_critic,
            ddpg_samples,
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
            action_bounds["action_low"],
            action_bounds["action_high"],
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
    def _call(self, actor, critic, samples, norm, bounds, coef):
        """Returns the full `(loss, aux)` pair the agent differentiates with
        `has_aux=True`."""
        return td3_actor_loss_fn(
            actor,
            critic,
            samples,
            norm["obs_mean"],
            norm["obs_std"],
            norm["obs_clip"],
            bounds["action_low"],
            bounds["action_high"],
            coef,
        )

    def _loss(self, actor, critic, samples, norm, bounds, coef):
        return self._call(actor, critic, samples, norm, bounds, coef)[0]

    def test_zero_coef_is_plain_dpg(
        self, det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
    ):
        loss = self._loss(
            det_actor, twin_critic, ddpg_samples, norm_params, action_bounds, 0.0
        )
        obs = Agent.normalize_obs(
            ddpg_samples["observations"],
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
        )
        q1, _ = twin_critic(obs, det_actor(obs))
        assert loss.shape == ()
        assert jnp.allclose(loss, -jnp.mean(q1), atol=1e-5)

    def test_penalty_only_charges_when_saturated(
        self, det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
    ):
        """At the default (small) init the logits are inside the threshold, so
        the penalty is inert; it must bite once the logits are driven out."""
        base = self._loss(
            det_actor, twin_critic, ddpg_samples, norm_params, action_bounds, 0.0
        )
        unsaturated = self._loss(
            det_actor, twin_critic, ddpg_samples, norm_params, action_bounds, 1e-2
        )
        assert jnp.allclose(base, unsaturated, atol=1e-6)

        saturated = copy.deepcopy(det_actor)
        params = nnx.state(saturated, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(saturated, params)
        with_penalty = self._loss(
            saturated, twin_critic, ddpg_samples, norm_params, action_bounds, 1e-2
        )
        without = self._loss(
            saturated, twin_critic, ddpg_samples, norm_params, action_bounds, 0.0
        )
        assert with_penalty > without

    def test_gradient_pulls_a_saturated_actor_back(
        self, det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
    ):
        """Regression guard for the CMU_006_13 collapse: a saturated actor must
        still receive a gradient that reduces |pre-activation|."""
        actor = copy.deepcopy(det_actor)
        params = nnx.state(actor, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(actor, params)

        obs = Agent.normalize_obs(
            ddpg_samples["observations"],
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
        )
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
                norm_params["obs_mean"],
                norm_params["obs_std"],
                norm_params["obs_clip"],
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

    def _aux(self, actor, critic, samples, norm, bounds):
        return TestTD3ActorLoss()._call(actor, critic, samples, norm, bounds, 1e-2)[1]

    def test_healthy_actor_reports_live_gradient(
        self, det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
    ):
        aux = self._aux(det_actor, twin_critic, ddpg_samples, norm_params, action_bounds)
        # Small output init keeps the logits in tanh's linear region.
        assert float(aux["pre_act_abs"]) < 1.0
        assert float(aux["tanh_grad"]) > 0.5
        assert float(aux["sat_frac"]) < 0.05
        assert float(aux["pre_act_penalty"]) == 0.0

    def test_saturated_actor_is_flagged(
        self, det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
    ):
        actor = copy.deepcopy(det_actor)
        params = nnx.state(actor, nnx.Param)
        params["output_layer"]["kernel"].value *= 200.0
        nnx.update(actor, params)

        healthy = self._aux(
            det_actor, twin_critic, ddpg_samples, norm_params, action_bounds
        )
        saturated = self._aux(
            actor, twin_critic, ddpg_samples, norm_params, action_bounds
        )
        assert saturated["pre_act_abs"] > healthy["pre_act_abs"]
        assert saturated["pre_act_max"] >= saturated["pre_act_abs"]
        # Healthy is <0.05 (asserted above), so 0.5 separates the two regimes
        # with room to spare without pinning the fixture's exact geometry.
        assert float(saturated["sat_frac"]) > 0.5
        # The dead-gradient signature: this is the number to watch in the logs.
        assert float(saturated["tanh_grad"]) < 0.05
        assert float(saturated["pre_act_penalty"]) > 0.0


class TestDDPGCriticLoss:
    def test_returns_scalar(self, det_critic, det_actor, ddpg_samples, norm_params, action_bounds):
        target_actor = copy.deepcopy(det_actor)
        target_critic = copy.deepcopy(det_critic)
        key = jax.random.PRNGKey(0)
        loss = ddpg_critic_loss_fn(
            det_critic,
            target_actor,
            target_critic,
            ddpg_samples,
            key,
            0.1,
            0.1,
            action_bounds["action_low"],
            action_bounds["action_high"],
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
        )
        assert loss.shape == ()

    def test_non_negative(self, det_critic, det_actor, ddpg_samples, norm_params, action_bounds):
        target_actor = copy.deepcopy(det_actor)
        target_critic = copy.deepcopy(det_critic)
        key = jax.random.PRNGKey(0)
        loss = ddpg_critic_loss_fn(
            det_critic,
            target_actor,
            target_critic,
            ddpg_samples,
            key,
            0.1,
            0.1,
            action_bounds["action_low"],
            action_bounds["action_high"],
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
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
    def test_actor_loss(self, stoch_actor, twin_critic, ddpg_samples, norm_params, action_bounds):
        key = jax.random.PRNGKey(0)
        alpha = 0.2
        loss, log_probs = sac_actor_loss_fn(
            stoch_actor,
            twin_critic,
            alpha,
            ddpg_samples,
            key,
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
            action_bounds["action_low"],
            action_bounds["action_high"],
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert log_probs.shape == (BATCH,)

    def test_critic_loss(self, stoch_actor, twin_critic, ddpg_samples, norm_params, action_bounds):
        target_twin = copy.deepcopy(twin_critic)
        key = jax.random.PRNGKey(0)
        loss = sac_critic_loss_fn(
            twin_critic,
            stoch_actor,
            target_twin,
            ddpg_samples,
            0.2,
            key,
            action_bounds["action_low"],
            action_bounds["action_high"],
            norm_params["obs_mean"],
            norm_params["obs_std"],
            norm_params["obs_clip"],
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
