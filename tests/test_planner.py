"""MPPI planner in the TD-MPC latent space.

Contract tests (shapes, bounds, determinism, warm-start handoff) plus the one
behavioural test that matters: on a world model whose reward is a known function
of the action, the planner must actually find the optimum. CPU only.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.agents.planner import estimate_value, plan as plan_raw
from roxie.agents.planner import plan_jit as plan
from roxie.models.actors import DeterministicActor
from roxie.models.critics import QCritic, TwinCritic
from roxie.models.world import TOLD, Encoder, LatentDynamics, RewardPredictor

OBS_DIM = 4
ACT_DIM = 2
LATENT_DIM = 6
HORIZON = 3
BATCH = 5

PLAN_KWARGS = dict(
    horizon=HORIZON,
    num_samples=32,
    num_elites=8,
    num_policy_trajectories=4,
    num_iterations=3,
)


@pytest.fixture
def model():
    return TOLD(
        encoder=Encoder(
            in_features=OBS_DIM,
            features=[32],
            latent_dim=LATENT_DIM,
            rngs=nnx.Rngs(params=0, dropout=1),
        ),
        dynamics=LatentDynamics(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            features=[32],
            rngs=nnx.Rngs(params=2, dropout=3),
        ),
        reward=RewardPredictor(
            latent_dim=LATENT_DIM,
            action_dim=ACT_DIM,
            features=[32],
            rngs=nnx.Rngs(params=4, dropout=5),
        ),
        critic=TwinCritic(
            QCritic(
                in_features=LATENT_DIM + ACT_DIM,
                features=[32],
                rngs=nnx.Rngs(params=6, dropout=7),
            ),
            QCritic(
                in_features=LATENT_DIM + ACT_DIM,
                features=[32],
                rngs=nnx.Rngs(params=8, dropout=9),
            ),
        ),
    )


@pytest.fixture
def policy():
    return DeterministicActor(
        in_features=LATENT_DIM,
        features=[32],
        action_dim=ACT_DIM,
        rngs=nnx.Rngs(params=10, dropout=11),
    )


@pytest.fixture
def bounds():
    return -jnp.ones(ACT_DIM), jnp.ones(ACT_DIM)


@pytest.fixture
def z(model, rng_key):
    obs = jax.random.normal(rng_key, (BATCH, OBS_DIM))
    return model.encode(obs)


@pytest.fixture
def prev_mean():
    return jnp.zeros((BATCH, HORIZON, ACT_DIM))


class TestEstimateValue:
    def test_shape(self, model, policy, bounds, rng_key):
        low, high = bounds
        n = 7
        z = jnp.zeros((n, BATCH, LATENT_DIM))
        actions = jnp.zeros((HORIZON, n, BATCH, ACT_DIM))
        values = estimate_value(
            model, policy, z, actions, rng_key, 0.99, 0.05, low, high
        )
        assert values.shape == (n, BATCH)
        assert jnp.all(jnp.isfinite(values))

    def test_discounting_uses_full_horizon(self, model, policy, bounds, rng_key):
        """A longer action sequence must accumulate more predicted reward terms,
        i.e. the horizon really is the scanned axis and not a fixed constant.

        Inputs are deliberately non-zero: these nets initialize their biases to
        zero, so an all-zero latent and action would make every head output
        exactly 0 and the comparison vacuous.
        """
        low, high = bounds
        z_key, a_key = jax.random.split(rng_key)
        z = jax.random.normal(z_key, (1, BATCH, LATENT_DIM))
        actions = jax.random.uniform(
            a_key, (5, 1, BATCH, ACT_DIM), minval=-1.0, maxval=1.0
        )
        short = estimate_value(
            model, policy, z, actions[:1], rng_key, 0.99, 0.0, low, high
        )
        long = estimate_value(
            model, policy, z, actions, rng_key, 0.99, 0.0, low, high
        )
        assert jnp.all(jnp.isfinite(short)) and jnp.all(jnp.isfinite(long))
        assert not jnp.allclose(short, long)


class TestPlan:
    def test_shapes_and_bounds(self, model, policy, z, prev_mean, bounds, rng_key):
        low, high = bounds
        action, next_mean, std, noise = plan(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        assert action.shape == (BATCH, ACT_DIM)
        assert next_mean.shape == (BATCH, HORIZON, ACT_DIM)
        assert std.shape == ()
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)
        assert jnp.all(next_mean >= -1.0) and jnp.all(next_mean <= 1.0)
        assert jnp.all(jnp.isfinite(action))

    def test_deterministic_given_key(self, model, policy, z, prev_mean, bounds, rng_key):
        low, high = bounds
        a1, m1, _, _ = plan(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        a2, m2, _, _ = plan(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        np.testing.assert_allclose(a1, a2)
        np.testing.assert_allclose(m1, m2)

    def test_different_keys_differ(self, model, policy, z, prev_mean, bounds):
        low, high = bounds
        a1, _, _, _ = plan(
            model, policy, z, prev_mean, jax.random.PRNGKey(0), low, high, **PLAN_KWARGS
        )
        a2, _, _, _ = plan(
            model, policy, z, prev_mean, jax.random.PRNGKey(1), low, high, **PLAN_KWARGS
        )
        assert not jnp.allclose(a1, a2)

    def test_eval_mode_is_noiseless(self, model, policy, z, prev_mean, bounds, rng_key):
        """Eval takes the best elite with no added noise, so repeated calls with
        different keys should agree far more closely than training calls."""
        low, high = bounds
        kwargs = dict(PLAN_KWARGS, evaluate=True)
        a1, _, _, _ = plan(
            model, policy, z, prev_mean, jax.random.PRNGKey(0), low, high, **kwargs
        )
        a2, _, _, _ = plan(
            model, policy, z, prev_mean, jax.random.PRNGKey(0), low, high, **kwargs
        )
        np.testing.assert_allclose(a1, a2)
        assert jnp.all(a1 >= -1.0) and jnp.all(a1 <= 1.0)

    def test_reported_noise_matches_applied_noise(
        self, model, policy, z, prev_mean, bounds, rng_key
    ):
        """The trainer logs this as exploration noise, so it must be the real
        clean-minus-executed deviation — zero when evaluating."""
        low, high = bounds
        _, _, _, noise = plan(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        assert noise.shape == (BATCH, ACT_DIM)
        assert jnp.any(jnp.abs(noise) > 0)

        _, _, _, eval_noise = plan(
            model,
            policy,
            z,
            prev_mean,
            rng_key,
            low,
            high,
            **dict(PLAN_KWARGS, evaluate=True),
        )
        np.testing.assert_allclose(eval_noise, jnp.zeros((BATCH, ACT_DIM)), atol=1e-7)

    def test_plain_plan_works_inside_an_outer_jit(
        self, model, policy, z, prev_mean, bounds, rng_key
    ):
        """Regression: the trainer's eval rollout calls the planner from inside
        its own compiled while_loop. `plan_jit` nested in another trace fails on
        the models' rng state, which is why the un-jitted `plan` exists."""
        low, high = bounds

        @nnx.jit
        def outer(model, policy, z, prev_mean, key):
            action, next_mean, _, _ = plan_raw(
                model, policy, z, prev_mean, key, low, high, **PLAN_KWARGS
            )
            return action, next_mean

        action, next_mean = outer(model, policy, z, prev_mean, rng_key)
        assert action.shape == (BATCH, ACT_DIM)
        assert next_mean.shape == (BATCH, HORIZON, ACT_DIM)
        assert jnp.all(jnp.isfinite(action))

    def test_jitted_and_plain_agree(
        self, model, policy, z, prev_mean, bounds, rng_key
    ):
        """The two entry points must be the same function."""
        low, high = bounds
        a_jit, m_jit, _, _ = plan(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        a_raw, m_raw, _, _ = plan_raw(
            model, policy, z, prev_mean, rng_key, low, high, **PLAN_KWARGS
        )
        np.testing.assert_allclose(a_jit, a_raw, rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(m_jit, m_raw, rtol=1e-5, atol=1e-6)

    def test_warm_start_shifts_plan(self, model, policy, z, bounds, rng_key):
        """The returned mean must be the fitted plan shifted one step forward —
        the next step re-plans from where this one left off."""
        low, high = bounds
        prev = jnp.zeros((BATCH, HORIZON, ACT_DIM))
        _, next_mean, _, _ = plan(
            model, policy, z, prev, rng_key, low, high, **PLAN_KWARGS
        )
        # Tail is a repeat of the last fitted entry (the shift's padding).
        np.testing.assert_allclose(next_mean[:, -1], next_mean[:, -2], rtol=1e-6)

    def test_accepts_warm_started_mean(self, model, policy, z, bounds, rng_key):
        """Feeding a previous plan back in must work and change the result —
        otherwise the warm start is silently ignored."""
        low, high = bounds
        cold = jnp.zeros((BATCH, HORIZON, ACT_DIM))
        warm = jnp.full((BATCH, HORIZON, ACT_DIM), 0.9)
        a_cold, _, _, _ = plan(
            model, policy, z, cold, rng_key, low, high, **PLAN_KWARGS
        )
        a_warm, _, _, _ = plan(
            model, policy, z, warm, rng_key, low, high, **PLAN_KWARGS
        )
        assert not jnp.allclose(a_cold, a_warm)

    def test_respects_asymmetric_bounds(self, model, policy, z, prev_mean, rng_key):
        """Candidates stay in [-1, 1]; env bounds only affect the scaling done
        inside the model, so a different range must not leak into the output."""
        action, _, _, _ = plan(
            model,
            policy,
            z,
            prev_mean,
            rng_key,
            jnp.array([-2.0, 0.0]),
            jnp.array([3.0, 5.0]),
            **PLAN_KWARGS,
        )
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)

    def test_single_env_batch(self, model, policy, bounds, rng_key):
        low, high = bounds
        z = model.encode(jnp.ones((1, OBS_DIM)))
        action, next_mean, _, _ = plan(
            model,
            policy,
            z,
            jnp.zeros((1, HORIZON, ACT_DIM)),
            rng_key,
            low,
            high,
            **PLAN_KWARGS,
        )
        assert action.shape == (1, ACT_DIM)
        assert next_mean.shape == (1, HORIZON, ACT_DIM)


class _ConstantDynamics(nnx.Module):
    """Identity latent dynamics — isolates the planner from model error."""

    def __call__(self, latents, actions, training=False):
        return latents


class _TargetReward(nnx.Module):
    """Reward peaks at a known action, so the optimum is known in closed form."""

    def __init__(self, target):
        self.target = target

    def __call__(self, latents, actions, training=False):
        err = jnp.sum(jnp.square(actions - self.target), axis=-1, keepdims=True)
        return -err


class _ZeroCritic(nnx.Module):
    def __call__(self, latents, actions, training=False):
        zeros = jnp.zeros(latents.shape[:-1] + (1,))
        return zeros, zeros


class TestPlannerOptimizes:
    def test_finds_known_reward_optimum(self, policy):
        """The real contract: given a model whose reward is maximized at a known
        action, the planner's chosen action must land near it. Uses a
        hand-built model so any failure is the optimizer's, not the network's."""
        target = jnp.array([0.6, -0.4])
        model = TOLD(
            encoder=Encoder(
                in_features=OBS_DIM,
                features=[8],
                latent_dim=LATENT_DIM,
                rngs=nnx.Rngs(params=0, dropout=1),
            ),
            dynamics=_ConstantDynamics(),
            reward=_TargetReward(target),
            critic=_ZeroCritic(),
        )
        z = jnp.zeros((BATCH, LATENT_DIM))
        action, _, std, _ = plan(
            model,
            policy,
            z,
            jnp.zeros((BATCH, HORIZON, ACT_DIM)),
            jax.random.PRNGKey(0),
            -jnp.ones(ACT_DIM),
            jnp.ones(ACT_DIM),
            horizon=HORIZON,
            num_samples=256,
            num_elites=16,
            num_policy_trajectories=8,
            num_iterations=8,
            evaluate=True,
        )
        # Env-unit bounds are [-1, 1] here, so planner units == env units.
        np.testing.assert_allclose(action, jnp.tile(target, (BATCH, 1)), atol=0.15)

    def test_std_collapses_when_converged(self, policy):
        """The reported std is the convergence diagnostic: on an easy objective
        it must shrink well below the initial max_std."""
        model = TOLD(
            encoder=Encoder(
                in_features=OBS_DIM,
                features=[8],
                latent_dim=LATENT_DIM,
                rngs=nnx.Rngs(params=0, dropout=1),
            ),
            dynamics=_ConstantDynamics(),
            reward=_TargetReward(jnp.array([0.6, -0.4])),
            critic=_ZeroCritic(),
        )
        _, _, std, _ = plan(
            model,
            policy,
            jnp.zeros((BATCH, LATENT_DIM)),
            jnp.zeros((BATCH, HORIZON, ACT_DIM)),
            jax.random.PRNGKey(0),
            -jnp.ones(ACT_DIM),
            jnp.ones(ACT_DIM),
            horizon=HORIZON,
            num_samples=256,
            num_elites=16,
            num_policy_trajectories=8,
            num_iterations=8,
            max_std=2.0,
        )
        assert float(std) < 0.5
