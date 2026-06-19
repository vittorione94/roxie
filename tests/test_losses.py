import copy

import jax
import jax.numpy as jnp
import pytest
from flax import nnx

# Import agents first to avoid circular import
import roxie.agents  # noqa: F401
from roxie.agents.agent import Agent
from roxie.agents.sac import LogAlpha
from roxie.losses.actor_losses import ddpg_actor_loss_fn, ppo_loss_fn, sac_actor_loss_fn, sac_alpha_loss_fn
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
            0.99,
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
            0.99,
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

    def test_actor_loss_scalar(self, stoch_actor, ppo_data):
        obs, actions, log_probs, values, advantages = ppo_data
        loss = ppo_loss_fn(
            stoch_actor,
            obs,
            actions,
            log_probs,
            jnp.full(ACT_DIM, -1.0),
            jnp.full(ACT_DIM, 1.0),
            advantages,
            clip_epsilon=0.2,
            entropy_coef=0.01,
            key=jax.random.PRNGKey(0),
        )
        assert loss.shape == ()
        assert jnp.isfinite(loss)

    def test_critic_loss_scalar(self, stoch_critic, ppo_data):
        obs, actions, log_probs, values, advantages = ppo_data
        loss = ppo_critic_loss_fn(stoch_critic, obs, values, advantages)
        assert loss.shape == ()
        assert jnp.isfinite(loss)
        assert loss >= 0.0


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
            0.99,
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
