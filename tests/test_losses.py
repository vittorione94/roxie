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
        # Per-sample bootstrap coefficient (gamma^b, 0 at terminals) — the
        # DDPG/TD3 critic losses consume this instead of gamma + terminal
        # flags. SAC still reads `terminals`; both keys coexist here.
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
