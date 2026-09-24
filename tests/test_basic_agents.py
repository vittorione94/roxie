import numpy as np

from roxie.agents.basic import NormalRandom


class TestNormalRandom:
    def test_step_shape(self):
        agent = NormalRandom(
            env_obs_size=4,
            env_action_size=3,
            action_low=-1.0,
            action_high=1.0,
            seed=42,
        )
        obs = np.zeros(4)
        action = agent.select_action(obs)[0]
        assert action.shape == (3,)

    def test_reproducibility(self):
        agent1 = NormalRandom(
            env_obs_size=4, env_action_size=3, action_low=-1, action_high=1, seed=42
        )
        agent2 = NormalRandom(
            env_obs_size=4, env_action_size=3, action_low=-1, action_high=1, seed=42
        )
        obs = np.zeros(4)
        a1 = agent1.select_action(obs)[0]
        a2 = agent2.select_action(obs)[0]
        np.testing.assert_array_equal(a1, a2)

    def test_different_seeds_differ(self):
        agent1 = NormalRandom(
            env_obs_size=4, env_action_size=3, action_low=-1, action_high=1, seed=0
        )
        agent2 = NormalRandom(
            env_obs_size=4, env_action_size=3, action_low=-1, action_high=1, seed=1
        )
        obs = np.zeros(4)
        a1 = agent1.select_action(obs)[0]
        a2 = agent2.select_action(obs)[0]
        assert not np.allclose(a1, a2)

    def test_custom_loc_scale(self):
        agent = NormalRandom(
            env_obs_size=4,
            env_action_size=2,
            action_low=-1,
            action_high=1,
            seed=0,
            loc=5.0,
            scale=0.001,
        )
        obs = np.zeros(4)
        actions = [agent.select_action(obs)[0] for _ in range(100)]
        mean_action = np.mean(actions)
        assert abs(mean_action - 5.0) < 0.5

    def test_export_hyperparams(self):
        agent = NormalRandom(
            env_obs_size=4, env_action_size=3, action_low=-1, action_high=1
        )
        assert agent._export_hyperparams() == {}
