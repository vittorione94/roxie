"""End-to-end TDMPC agent: construction, acting, buffering, learning.

Runs the real code path — hydra-instantiated nets from the shipped yaml, a real
flashbax trajectory buffer, and the fused jitted gradient step — on a tiny
config. CPU only.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from omegaconf import OmegaConf

from roxie.agents import agents
from roxie.agents.tdmpc import TDMPC
from roxie.models.world import TOLD

OBS_DIM = 6
ACT_DIM = 3
NUM_ENVS = 4
HORIZON = 3
LATENT_DIM = 8


def _configs():
    actor = OmegaConf.create(
        {
            "_target_": "roxie.models.actors.DeterministicActor",
            "features": [32, 32],
            "use_layer_norm": True,
        }
    )
    critic = OmegaConf.create(
        {
            "encoder": {
                "_target_": "roxie.models.world.Encoder",
                "features": [32],
                "use_layer_norm": True,
            },
            "dynamics": {
                "_target_": "roxie.models.world.LatentDynamics",
                "features": [32],
                "use_layer_norm": True,
            },
            "reward": {
                "_target_": "roxie.models.world.RewardPredictor",
                "features": [32],
                "use_layer_norm": True,
            },
            "q": {
                "_target_": "roxie.models.critics.QCritic",
                "features": [32],
                "use_layer_norm": True,
            },
        }
    )
    memory = OmegaConf.create(
        {
            "_target_": "flashbax.buffers.make_flat_buffer",
            "max_length": 4_000,
            "min_length": 64,
            "sample_batch_size": 16,
            "add_sequences": False,
            "add_batch_size": NUM_ENVS,
        }
    )
    # TD-MPC explores through the planner's own sampling, not the noise module,
    # but DDPG.__init__ requires one.
    noise = OmegaConf.create(
        {
            "_target_": "roxie.exploration.noisy.GaussianNoise",
            "initial_noise_scale": 0.1,
        }
    )
    return actor, critic, memory, noise


@pytest.fixture
def agent():
    actor_config, critic_config, memory_config, noise_config = _configs()
    return TDMPC(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        actor_config=actor_config,
        critic_config=critic_config,
        memory_config=memory_config,
        noise_config=noise_config,
        horizon=HORIZON,
        latent_dim=LATENT_DIM,
        num_samples=16,
        num_elites=4,
        num_policy_trajectories=2,
        num_iterations=2,
        steps_before_learning=0,
        steps_between_updates=1,
        learning_steps=2,
        memory_warmup=0,
    )


def _fill_buffer(agent, key, steps=40):
    """Push `steps` env-step batches of random transitions through the agent's
    own add path, so the buffer ends up in exactly the layout learning reads."""
    for i in range(steps):
        key, k1, k2 = jax.random.split(key, 3)
        prev_obs = jax.random.normal(k1, (NUM_ENVS, OBS_DIM))
        next_obs = jax.random.normal(k2, (NUM_ENVS, OBS_DIM))
        action = jnp.zeros((NUM_ENVS, ACT_DIM))
        reward = jnp.ones((NUM_ENVS,))
        # Terminate one env periodically so the masks see real boundaries.
        term = jnp.zeros((NUM_ENVS,), dtype=bool).at[0].set(i % 11 == 10)
        trunc = jnp.zeros((NUM_ENVS,), dtype=bool).at[1].set(i % 13 == 12)
        agent.add_transitions(prev_obs, action, reward, term, trunc, next_obs)
    return key


class TestConstruction:
    def test_registered(self):
        assert agents["tdmpc"] is TDMPC

    def test_builds_told_in_critic_slot(self, agent):
        """The whole integration rests on this: the world model occupies the
        critic slot, so save/load and target updates need no special casing."""
        assert isinstance(agent.state.critic, TOLD)
        assert isinstance(agent.state.target_critic, TOLD)

    def test_policy_prior_is_over_latents(self, agent):
        z = jnp.ones((2, LATENT_DIM))
        assert agent.state.actor(z).shape == (2, ACT_DIM)

    def test_horizon_drives_replay_window(self, agent):
        """horizon + 1 items per sampled window — the sequence the loss needs."""
        assert agent.n_step == HORIZON
        sample = agent.replay.sample(agent.state.buffer_state, jax.random.PRNGKey(0))
        assert sample.experience.observation.shape[1] == HORIZON + 1

    def test_exports_hyperparams(self, agent):
        hp = agent._export_hyperparams()
        assert hp["horizon"] == HORIZON
        assert hp["latent_dim"] == LATENT_DIM
        assert hp["num_samples"] == 16


class TestActing:
    def test_step_shape_and_bounds(self, agent, rng_key):
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        action = agent.step(obs, evaluate=False, key=rng_key)
        assert action.shape == (NUM_ENVS, ACT_DIM)
        assert jnp.all(action >= -1.0) and jnp.all(action <= 1.0)
        assert jnp.all(jnp.isfinite(action))

    def test_plan_is_carried_between_steps(self, agent, rng_key):
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        assert agent._plan_mean is None
        agent.step(obs, evaluate=False, key=rng_key)
        first = agent._plan_mean
        assert first.shape == (NUM_ENVS, HORIZON, ACT_DIM)
        agent.step(obs, evaluate=False, key=jax.random.PRNGKey(7))
        assert not jnp.allclose(first, agent._plan_mean)

    def test_eval_plan_is_separate_from_training_plan(self, agent, rng_key):
        """Eval runs interleave with training steps; they must not clobber the
        acting plan."""
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        agent.step(obs, evaluate=False, key=rng_key)
        train_plan = agent._plan_mean
        agent.step(obs, evaluate=True, key=rng_key)
        np.testing.assert_allclose(train_plan, agent._plan_mean)
        assert agent._eval_plan_mean is not None

    def test_done_envs_drop_their_plan(self, agent, rng_key):
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        agent.step(obs, evaluate=False, key=rng_key)
        assert jnp.any(jnp.abs(agent._plan_mean) > 0)

        term = jnp.zeros((NUM_ENVS,), dtype=bool).at[0].set(True)
        trunc = jnp.zeros((NUM_ENVS,), dtype=bool).at[2].set(True)
        agent.add_transitions(
            obs, agent.last_action, jnp.ones((NUM_ENVS,)), term, trunc, obs
        )
        np.testing.assert_allclose(agent._plan_mean[0], 0.0)
        np.testing.assert_allclose(agent._plan_mean[2], 0.0)
        assert jnp.any(jnp.abs(agent._plan_mean[1]) > 0)

    def test_reset_plan(self, agent, rng_key):
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        agent.step(obs, evaluate=True, key=rng_key)
        assert agent._eval_plan_mean is not None
        agent.reset_plan(evaluate=True)
        assert agent._eval_plan_mean is None

    def test_eval_action_fn_matches_planner(self, agent, rng_key):
        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        act = agent.eval_action_fn()
        action, next_mean = act(
            agent.state.actor,
            agent.state.critic,
            agent.state.obs_stats,
            obs,
            agent.initial_plan_mean(NUM_ENVS),
            rng_key,
        )
        assert action.shape == (NUM_ENVS, ACT_DIM)
        assert next_mean.shape == (NUM_ENVS, HORIZON, ACT_DIM)
        assert jnp.all(jnp.isfinite(action))


class TestLearning:
    def test_update_runs_and_reports_losses(self, agent, rng_key):
        key = _fill_buffer(agent, rng_key)
        steps, policy_loss, model_loss = agent.update(steps=0, agent_rng=key)
        assert steps == 2
        assert jnp.isfinite(policy_loss) and jnp.isfinite(model_loss)

    def test_update_changes_world_model(self, agent, rng_key):
        key = _fill_buffer(agent, rng_key)
        # np.asarray, not jnp: _grad_steps donates the train state, so a device
        # view of a pre-update weight is freed by the update itself.
        before = np.asarray(agent.state.critic.encoder.output_layer.kernel.value)
        agent.update(steps=0, agent_rng=key)
        after = np.asarray(agent.state.critic.encoder.output_layer.kernel.value)
        assert not np.allclose(before, after)

    def test_update_changes_policy_prior(self, agent, rng_key):
        key = _fill_buffer(agent, rng_key)
        before = np.asarray(agent.state.actor.output_layer.kernel.value)
        agent.update(steps=0, agent_rng=key)
        after = np.asarray(agent.state.actor.output_layer.kernel.value)
        assert not np.allclose(before, after)

    def test_target_model_tracks_online_model(self, agent, rng_key):
        """The target must move toward the online model but stay behind it —
        that is the whole point of the EMA."""
        key = _fill_buffer(agent, rng_key)
        before = np.asarray(
            agent.state.target_critic.encoder.output_layer.kernel.value
        )
        agent.update(steps=0, agent_rng=key)
        online = np.asarray(agent.state.critic.encoder.output_layer.kernel.value)
        target = np.asarray(
            agent.state.target_critic.encoder.output_layer.kernel.value
        )
        assert not np.allclose(before, target), "target did not update"
        assert not np.allclose(online, target), "target jumped to the online weights"

    def test_learning_is_stable_over_several_updates(self, agent, rng_key):
        key = _fill_buffer(agent, rng_key)
        for _ in range(5):
            key, update_key = jax.random.split(key)
            _, policy_loss, model_loss = agent.update(steps=0, agent_rng=update_key)
            assert jnp.isfinite(policy_loss), "policy loss diverged"
            assert jnp.isfinite(model_loss), "model loss diverged"

    def test_acting_still_works_after_learning(self, agent, rng_key):
        key = _fill_buffer(agent, rng_key)
        agent.update(steps=0, agent_rng=key)
        obs = jax.random.normal(key, (NUM_ENVS, OBS_DIM))
        action = agent.step(obs, evaluate=False, key=key)
        assert jnp.all(jnp.isfinite(action))


class TestCheckpointing:
    def test_save_and_reload_preserves_planning(self, agent, rng_key, tmp_path):
        """TOLD sits in the critic slot precisely so the generic save/load path
        works; a reloaded agent must plan identically."""
        key = _fill_buffer(agent, rng_key)
        agent.update(steps=0, agent_rng=key)

        obs = jax.random.normal(rng_key, (NUM_ENVS, OBS_DIM))
        agent.reset_plan()
        before = agent.step(obs, evaluate=True, key=rng_key)

        path = tmp_path / "ckpt"
        agent.save(path)

        actor_config, critic_config, memory_config, noise_config = _configs()
        restored = TDMPC.load(
            path,
            OBS_DIM,
            ACT_DIM,
            actor_config,
            critic_config,
            memory_config,
            noise_config,
        )
        restored.reset_plan()
        after = restored.step(obs, evaluate=True, key=rng_key)
        np.testing.assert_allclose(before, after, rtol=1e-5, atol=1e-5)
