import types

import jax
import jax.numpy as jnp
import pytest
from omegaconf import OmegaConf

import roxie.agents  # noqa: F401  (avoid circular import)
from roxie.agents.ppo import PPO

NUM_ENVS = 4
OBS_DIM = 6
ACT_DIM = 2
ROLLOUT = 8


def _make_agent(**kwargs):
    actor_config = OmegaConf.create(
        {"_target_": "roxie.models.actors.StochasticActor", "features": [16, 16]}
    )
    critic_config = OmegaConf.create(
        {"_target_": "roxie.models.critics.VCritic", "features": [16, 16]}
    )
    memory_config = OmegaConf.create(
        {
            "_target_": "flashbax.buffers.make_trajectory_queue",
            "max_length_time_axis": 64,
            "add_batch_size": NUM_ENVS,
            "add_sequence_length": 1,
            "sample_sequence_length": ROLLOUT,
        }
    )
    params = dict(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=jnp.full((ACT_DIM,), -1.0),
        action_high=jnp.full((ACT_DIM,), 1.0),
        actor_config=actor_config,
        critic_config=critic_config,
        memory_config=memory_config,
        # One pass over one minibatch, so the single reported approx_kl IS the
        # first pass's -- measured at the behaviour parameters themselves.
        learning_steps=1,
        num_minibatches=1,
        target_kl=None,
    )
    params.update(kwargs)
    return PPO(**params)


def _env_state(key, scale):
    return types.SimpleNamespace(
        obs=jax.random.normal(key, (NUM_ENVS, OBS_DIM)) * scale + scale,
        reward=jax.random.normal(key, (NUM_ENVS,)),
        info={
            "termination": jnp.zeros((NUM_ENVS,), jnp.bool_),
            "truncation": jnp.zeros((NUM_ENVS,), jnp.bool_),
        },
    )


def _collect_and_update(agent, drift=2.0, unfreeze_norm=False):
    """Roll out under a deliberately drifting observation distribution."""
    key = jax.random.PRNGKey(0)
    key, k0 = jax.random.split(key)
    state = _env_state(k0, 1.0)

    for t in range(ROLLOUT):
        key, act_key, step_key = jax.random.split(key, 3)
        agent.step(state.obs, evaluate=False, key=act_key)
        nxt = _env_state(step_key, 1.0 + drift * t)
        agent.add(state, nxt)
        state = nxt

    if unfreeze_norm:
        # Reproduce the pre-fix behaviour: throw away the acting snapshot so the
        # update re-derives mean/std from the (heavily drifted) running stats.
        agent._obs_norm = None

    key, update_key = jax.random.split(key)
    agent.update(steps=ROLLOUT * NUM_ENVS, agent_rng=update_key)
    return agent.pop_diagnostics()


class TestPPOObsNormFreeze:
    def test_first_pass_ratio_is_one_despite_stat_drift(self):
        """The behaviour log-probs must be reproducible at the same parameters.

        `norm_obs` is rebuilt at update time while the running obs statistics
        moved throughout the rollout. Unless the normalizer is pinned to the
        snapshot that `step()` acted under, re-evaluating the rollout gives a
        ratio != 1 before any gradient step, so the clip and the KL early stop
        trip on normalization drift rather than policy drift.
        """
        diag = _collect_and_update(_make_agent(normalize_observations=True))
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)
        assert diag["ppo/clip_frac"] == pytest.approx(0.0, abs=1e-6)

    def test_drifting_stats_would_break_the_ratio(self):
        """Guards the test above against silently passing for the wrong reason."""
        diag = _collect_and_update(
            _make_agent(normalize_observations=True), unfreeze_norm=True
        )
        assert diag["ppo/approx_kl"] > 1e-3

    def test_snapshot_refreshes_between_rollouts(self):
        agent = _make_agent(normalize_observations=True)
        _collect_and_update(agent)
        first = agent._frozen_obs_norm()
        # More data at a different scale, then another rollout boundary.
        _collect_and_update(agent, drift=5.0)
        second = agent._frozen_obs_norm()
        assert not jnp.allclose(first[0], second[0])

    def test_unnormalized_agent_leaves_obs_untouched(self):
        """With normalization off the stats stay empty; obs must not be rescaled.

        `_prepare_rollout` used to normalize unconditionally, which divided by a
        zero-variance std and pinned every feature to the clip bound.
        """
        agent = _make_agent(normalize_observations=False)
        diag = _collect_and_update(agent)
        assert float(agent.state.obs_stats.count) == 0.0
        assert diag["ppo/approx_kl"] == pytest.approx(0.0, abs=1e-5)
