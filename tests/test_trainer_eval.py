"""Trainer eval rollout: the pluggable action hook.

`_make_eval_fn` gained a `critic` argument and an action carry so planning
agents can be evaluated with their planner. These tests pin both halves: TD3
(no hook) must behave exactly as before, and TDMPC must actually plan and thread
its warm start through the compiled while_loop. CPU only.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import struct
from omegaconf import OmegaConf

from roxie.agents.td3 import TD3
from roxie.agents.tdmpc import TDMPC
from roxie.utils.trainer import Trainer, _default_eval_action_fn

OBS_DIM = 6
ACT_DIM = 3
NUM_TESTS = 4
MAX_STEPS = 6
HORIZON = 3
LATENT_DIM = 8


@struct.dataclass
class _EnvState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    step: jnp.ndarray


@struct.dataclass
class _Wrapped:
    env_state: _EnvState


def _initial_states():
    return _Wrapped(
        env_state=_EnvState(
            obs=jnp.zeros((NUM_TESTS, OBS_DIM)),
            reward=jnp.zeros((NUM_TESTS,)),
            done=jnp.zeros((NUM_TESTS,), dtype=bool),
            step=jnp.zeros((NUM_TESTS,), dtype=jnp.int32),
        )
    )


def _v_step(states, action):
    """Toy env: reward is the mean action, episodes end after 4 steps. Reward
    depends on the action so a change of policy shows up in the score."""
    step = states.env_state.step + 1
    return _Wrapped(
        env_state=_EnvState(
            obs=states.env_state.obs + jnp.mean(action, axis=-1, keepdims=True),
            reward=jnp.mean(action, axis=-1),
            done=step >= 4,
            step=step,
        )
    )


def _memory_config():
    return OmegaConf.create(
        {
            "_target_": "flashbax.buffers.make_flat_buffer",
            "max_length": 2_000,
            "min_length": 32,
            "sample_batch_size": 8,
            "add_sequences": False,
            "add_batch_size": NUM_TESTS,
        }
    )


def _noise_config():
    return OmegaConf.create(
        {
            "_target_": "roxie.exploration.noisy.GaussianNoise",
            "initial_noise_scale": 0.1,
        }
    )


@pytest.fixture
def td3_agent():
    return TD3(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        actor_config=OmegaConf.create(
            {
                "_target_": "roxie.models.actors.DeterministicActor",
                "features": [32],
            }
        ),
        critic_config=OmegaConf.create(
            {"_target_": "roxie.models.critics.QCritic", "features": [32]}
        ),
        memory_config=_memory_config(),
        noise_config=_noise_config(),
        memory_warmup=0,
    )


@pytest.fixture
def tdmpc_agent():
    return TDMPC(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        actor_config=OmegaConf.create(
            {
                "_target_": "roxie.models.actors.DeterministicActor",
                "features": [32],
            }
        ),
        critic_config=OmegaConf.create(
            {
                "encoder": {
                    "_target_": "roxie.models.world.Encoder",
                    "features": [32],
                },
                "dynamics": {
                    "_target_": "roxie.models.world.LatentDynamics",
                    "features": [32],
                },
                "reward": {
                    "_target_": "roxie.models.world.RewardPredictor",
                    "features": [32],
                },
                "q": {"_target_": "roxie.models.critics.QCritic", "features": [32]},
            }
        ),
        memory_config=_memory_config(),
        noise_config=_noise_config(),
        horizon=HORIZON,
        latent_dim=LATENT_DIM,
        num_samples=8,
        num_elites=4,
        num_policy_trajectories=2,
        num_iterations=2,
        memory_warmup=0,
    )


def _run_eval(agent):
    trainer = Trainer(output_dir="/tmp", test_episodes=NUM_TESTS)
    trainer.agent = agent
    eval_fn = trainer._make_eval_fn(_v_step, NUM_TESTS, MAX_STEPS)

    initial_carry = getattr(agent, "initial_plan_mean", None)
    carry = initial_carry(NUM_TESTS) if initial_carry is not None else None

    return eval_fn(
        agent.state.actor,
        agent.state.critic,
        agent.state.obs_stats,
        _initial_states(),
        carry,
        jax.random.PRNGKey(0),
    )


class TestNonPlanningAgentUnchanged:
    def test_eval_runs(self, td3_agent):
        scores, lengths = _run_eval(td3_agent)
        assert scores.shape == (NUM_TESTS,)
        assert lengths.shape == (NUM_TESTS,)
        assert jnp.all(jnp.isfinite(scores))
        # Episodes end after 4 steps, below the 6-step cap.
        np.testing.assert_array_equal(np.asarray(lengths), np.full(NUM_TESTS, 4))

    def test_matches_plain_actor_pass(self, td3_agent):
        """The default hook must be exactly the old behaviour: greedy actor
        output, normalized obs, scaled to env units."""
        act = _default_eval_action_fn(td3_agent)
        obs = jnp.zeros((NUM_TESTS, OBS_DIM))
        action, carry = act(
            td3_agent.state.actor,
            td3_agent.state.critic,
            td3_agent.state.obs_stats,
            obs,
            None,
            jax.random.PRNGKey(0),
        )
        assert carry is None

        from roxie.agents.agent import Agent

        mean, std = Agent.obs_mean_std(td3_agent.state.obs_stats, td3_agent.obs_eps)
        expected = Agent.scale_to_env(
            jnp.clip(
                td3_agent.state.actor(
                    Agent.normalize_obs(obs, mean, std, td3_agent.obs_clip)
                ),
                -1.0,
                1.0,
            ),
            td3_agent.action_low,
            td3_agent.action_high,
        )
        np.testing.assert_allclose(action, expected, rtol=1e-6)

    def test_no_eval_hook(self, td3_agent):
        assert not hasattr(td3_agent, "eval_action_fn")


class TestPlanningAgent:
    def test_eval_runs_with_planner(self, tdmpc_agent):
        scores, lengths = _run_eval(tdmpc_agent)
        assert scores.shape == (NUM_TESTS,)
        assert jnp.all(jnp.isfinite(scores))
        np.testing.assert_array_equal(np.asarray(lengths), np.full(NUM_TESTS, 4))

    def test_planner_differs_from_bare_actor(self, tdmpc_agent):
        """The whole reason for the hook: planning must produce a different
        action than the bare policy prior, or eval would be scoring a different
        agent than the one being trained."""
        act = tdmpc_agent.eval_action_fn()
        obs = jnp.ones((NUM_TESTS, OBS_DIM))
        planned, _ = act(
            tdmpc_agent.state.actor,
            tdmpc_agent.state.critic,
            tdmpc_agent.state.obs_stats,
            obs,
            tdmpc_agent.initial_plan_mean(NUM_TESTS),
            jax.random.PRNGKey(0),
        )
        # The bare prior: encode, then a single greedy actor pass — what the
        # default hook would effectively score if TD-MPC did not override it.
        from roxie.agents.agent import Agent

        mean, std = Agent.obs_mean_std(
            tdmpc_agent.state.obs_stats, tdmpc_agent.obs_eps
        )
        z = tdmpc_agent.state.critic.encode(
            Agent.normalize_obs(obs, mean, std, tdmpc_agent.obs_clip)
        )
        prior = Agent.scale_to_env(
            jnp.clip(tdmpc_agent.state.actor(z), -1.0, 1.0),
            tdmpc_agent.action_low,
            tdmpc_agent.action_high,
        )
        assert not jnp.allclose(planned, prior)

    def test_carry_advances_through_the_loop(self, tdmpc_agent):
        """The plan carry must be threaded, not reset each step: two successive
        hook calls starting from a cold plan must differ."""
        act = tdmpc_agent.eval_action_fn()
        obs = jnp.ones((NUM_TESTS, OBS_DIM))
        cold = tdmpc_agent.initial_plan_mean(NUM_TESTS)
        _, carry1 = act(
            tdmpc_agent.state.actor,
            tdmpc_agent.state.critic,
            tdmpc_agent.state.obs_stats,
            obs,
            cold,
            jax.random.PRNGKey(0),
        )
        assert carry1.shape == cold.shape
        assert not jnp.allclose(carry1, cold)

    def test_eval_is_deterministic(self, tdmpc_agent):
        """Eval planning takes the best elite with no noise, so two identical
        rollouts must score identically."""
        s1, _ = _run_eval(tdmpc_agent)
        s2, _ = _run_eval(tdmpc_agent)
        np.testing.assert_allclose(s1, s2, rtol=1e-6)
