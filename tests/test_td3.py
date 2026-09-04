"""End-to-end guards for the TD3 gradient burst and its `td3/` diagnostics.

The diagnostics are produced inside `nnx.cond` (delayed policy update) nested in
`lax.scan` (fused burst), which is exactly where a shape/pytree mismatch fails
silently at trace time rather than at review time — hence a real burst here
rather than a unit test of the loss functions alone.

What every agent's diagnostics must satisfy lives in `test_diagnostics.py`;
what is here is TD3's own reading of them — the saturation and value-inflation
metrics its tuning actually turns on.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from roxie.agents.td3 import TD3
from roxie.agents.utils import Transition

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8


@pytest.fixture
def agent():
    cfg = lambda d: OmegaConf.create(d)  # noqa: E731
    return TD3(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        actor_config=cfg(
            {
                "_target_": "roxie.models.actors.DeterministicActor",
                "features": [32, 32],
                "use_layer_norm": True,
                "output_init_scale": 0.01,
            }
        ),
        critic_config=cfg(
            {
                "_target_": "roxie.models.critics.QCritic",
                "features": [32, 32],
                "use_layer_norm": True,
            }
        ),
        memory_config=cfg(
            {
                "_target_": "flashbax.buffers.make_flat_buffer",
                "max_length": 2048,
                "min_length": 64,
                "sample_batch_size": 32,
                "add_sequences": False,
                "add_batch_size": NUM_ENVS,
            }
        ),
        noise_config=cfg(
            {
                "_target_": "roxie.exploration.noisy.GaussianNoise",
                "initial_noise_scale": 0.1,
            }
        ),
        policy_delay=2,
        learning_steps=4,
        pre_activation_coef=1e-1,
    )


def _fill_buffer(agent, n_batches=40, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        agent.add_transitions(
            jnp.asarray(rng.standard_normal((NUM_ENVS, OBS_DIM)), jnp.float32),
            jnp.asarray(rng.standard_normal((NUM_ENVS, ACT_DIM)), jnp.float32),
            jnp.asarray(rng.standard_normal(NUM_ENVS), jnp.float32),
            jnp.zeros(NUM_ENVS, jnp.bool_),
            jnp.zeros(NUM_ENVS, jnp.bool_),
            jnp.asarray(rng.standard_normal((NUM_ENVS, OBS_DIM)), jnp.float32),
        )


EXPECTED_KEYS = {
    "td3/pre_act_abs",
    "td3/pre_act_max",
    "td3/pre_act_penalty",
    "td3/sat_frac",
    "td3/tanh_grad",
    "td3/actor_q",
    "td3/q_buffer",
    "td3/q_target",
    "td3/td_abs",
    "td3/twin_gap",
    "td3/target_smooth_clip_frac",
    "td3/target_act_rail_frac",
}


class TestTD3Diagnostics:
    def test_no_burst_reports_nothing(self, agent):
        """A misleading zero is worse than a gap in the curve."""
        assert agent.pop_diagnostics() == {}

    def test_burst_emits_every_metric_as_a_finite_float(self, agent):
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        diag = agent.pop_diagnostics()

        assert EXPECTED_KEYS <= set(diag)
        for key, value in diag.items():
            assert isinstance(value, float), key
            assert np.isfinite(value), key

    def test_drains_on_read(self, agent):
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        assert agent.pop_diagnostics() != {}
        # Second read in the same epoch must not re-report the same burst.
        assert agent.pop_diagnostics() == {}

    def test_bounded_metrics_stay_in_range(self, agent):
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        diag = agent.pop_diagnostics()
        for key in (
            "td3/sat_frac",
            "td3/tanh_grad",
            "td3/target_smooth_clip_frac",
            "td3/target_act_rail_frac",
        ):
            assert 0.0 <= diag[key] <= 1.0, key
        assert diag["td3/pre_act_max"] >= diag["td3/pre_act_abs"]

    def test_fresh_actor_reports_a_live_tanh_gradient(self, agent):
        """At init the small output layer keeps the logits in the linear region;
        if this ever starts near 0 the actor is born saturated."""
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        diag = agent.pop_diagnostics()
        assert diag["td3/tanh_grad"] > 0.5
        assert diag["td3/sat_frac"] < 0.1

    def test_replay_ratio_reported_only_once_env_steps_are_known(self, agent):
        """`updates_per_env_step` needs the trainer's step count, which reaches
        the agent as the `pop_diagnostics` argument — so it is reported on the
        async path too, where the learner calls `learn` and `update` never runs.
        """
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        assert "td3/updates_per_env_step" not in agent.pop_diagnostics()

        agent.learn(jax.random.PRNGKey(1))
        diag = agent.pop_diagnostics(env_steps=1000)
        assert diag["td3/updates_per_env_step"] > 0.0
        assert 0.0 <= diag["td3/buffer_frac"] <= 1.0


class TestTD3GradientBurst:
    @staticmethod
    def _critic_params(agent):
        # `nnx.Param` only: the raw module tree also carries rng keys, which are
        # not concatenable.
        leaves = jax.tree.leaves(nnx.state(agent.state.critic, nnx.Param))
        return jnp.concatenate([jnp.asarray(p).ravel() for p in leaves])

    def test_burst_trains_and_returns_finite_losses(self, agent):
        _fill_buffer(agent)
        before = self._critic_params(agent)
        actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(0))
        after = self._critic_params(agent)

        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)
        assert not jnp.allclose(before, after), "critic did not move"

    def test_policy_delay_does_not_leak_zeros_into_the_actor_metrics(self, agent):
        """The skipped `nnx.cond` branch returns zeros to keep the pytree shape;
        `_grad_steps` must divide by the number of ACTOR updates, not by
        `n_steps`, or every actor metric reads `1/policy_delay` of its true
        value."""
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0), n_steps=2)  # 1 actor update of 2 steps
        delayed = agent.pop_diagnostics()["td3/tanh_grad"]

        agent2_seed_matched = agent  # same weights, same buffer
        agent2_seed_matched.learn(jax.random.PRNGKey(0), n_steps=1)  # 1 of 1
        undelayed = agent2_seed_matched.pop_diagnostics()["td3/tanh_grad"]

        # Both are means over exactly one actor update on a near-identical
        # actor: within a hair of each other, and nowhere near a 2x dilution.
        assert delayed == pytest.approx(undelayed, rel=0.15)
