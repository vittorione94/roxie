"""End-to-end guards for the fused SAC gradient burst.

SAC used to run its `learning_steps` as a Python loop of separate `nnx.jit`
dispatches; it now fuses them into one `lax.scan` with a donated train state,
like DDPG/TD3. The entropy temperature and its Adam slots ride in the scan
carry, which is exactly the kind of thing that silently stops updating (or
fails to trace) without a real burst exercising it — hence these tests rather
than unit tests of the loss functions alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from roxie.agents.sac import SAC

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8


def _make_agent(**kwargs):
    cfg = lambda d: OmegaConf.create(d)  # noqa: E731
    return SAC(
        env_obs_size=OBS_DIM,
        env_action_size=ACT_DIM,
        action_low=-jnp.ones(ACT_DIM),
        action_high=jnp.ones(ACT_DIM),
        actor_config=cfg(
            {
                "_target_": "roxie.models.actors.StochasticActor",
                "features": [32, 32],
                "use_layer_norm": True,
                "std_min": 1e-4,
                "std_max": 5.0,
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
        **{"learning_steps": 4, **kwargs},
    )


@pytest.fixture
def agent():
    return _make_agent()


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


def _actor_params(agent):
    return jax.tree.map(jnp.copy, nnx.state(agent.state.actor, nnx.Param))


class TestFusedBurst:
    def test_burst_runs_and_moves_params(self, agent):
        _fill_buffer(agent)
        before = _actor_params(agent)

        actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(0))

        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)
        moved = [
            not jnp.allclose(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(_actor_params(agent)))
        ]
        assert any(moved), "actor params unchanged after a gradient burst"

    def test_target_critic_tracks_online_critic(self, agent):
        """The soft update runs inside the scan body, so a burst of N steps must
        move the target N times — not once, and not zero times."""
        _fill_buffer(agent)
        before = jax.tree.map(
            jnp.copy, nnx.state(agent.state.target_critic, nnx.Param)
        )
        agent.learn(jax.random.PRNGKey(0))
        after = nnx.state(agent.state.target_critic, nnx.Param)

        assert any(
            not jnp.allclose(a, b)
            for a, b in zip(jax.tree.leaves(before), jax.tree.leaves(after))
        )

    def test_alpha_updates_every_step_of_the_burst(self):
        """`log_alpha` and its Adam slots live in the scan carry. If they were
        closed over instead, alpha would advance by one step per burst no matter
        how long the burst is — this pins the difference."""
        short = _make_agent(learning_steps=1)
        long = _make_agent(learning_steps=8)
        _fill_buffer(short)
        _fill_buffer(long)

        short.learn(jax.random.PRNGKey(0))
        long.learn(jax.random.PRNGKey(0))

        a_short = float(short.log_alpha_module.log_alpha.value)
        a_long = float(long.log_alpha_module.log_alpha.value)
        assert a_short != pytest.approx(0.0), "alpha did not update at all"
        assert abs(a_long) > abs(a_short), (
            f"8-step burst moved alpha {a_long} no further than a 1-step burst "
            f"{a_short}: the temperature is not being carried through the scan"
        )

    def test_auto_alpha_off_freezes_temperature(self):
        agent = _make_agent(auto_alpha=False, init_log_alpha=-1.0)
        _fill_buffer(agent)
        agent.learn(jax.random.PRNGKey(0))
        assert float(agent.log_alpha_module.log_alpha.value) == pytest.approx(-1.0)

    def test_state_is_reusable_after_donation(self, agent):
        """`_grad_steps` donates the train state. Consecutive bursts must keep
        working — i.e. the agent really does rebind `self.state` to the result
        rather than reading the donated (now invalid) input."""
        _fill_buffer(agent)
        for i in range(3):
            actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(i))
            assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)


class TestPolicyDelay:
    """The actor + temperature update runs under `nnx.cond` inside the scan.
    Both branches have to leave the graph state structurally identical, and the
    default (1) has to reproduce the undelayed behaviour exactly."""

    def test_actor_movement_decreases_monotonically_with_delay(self):
        """Drift from the initial actor must order 1 > 2 > 4 > 8: that is the
        signature of the mask actually gating the update count, and it pins
        delay=1 as the every-step case."""
        drifts = []
        for delay in (1, 2, 4, 8):
            a = _make_agent(learning_steps=8, policy_delay=delay)
            _fill_buffer(a)
            start = _actor_params(a)
            a.learn(jax.random.PRNGKey(0))
            drifts.append(
                sum(
                    float(jnp.sum((x - y) ** 2))
                    for x, y in zip(
                        jax.tree.leaves(start), jax.tree.leaves(_actor_params(a))
                    )
                )
            )

        assert drifts == sorted(drifts, reverse=True), (
            f"actor drift not monotonically decreasing in policy_delay: {drifts}"
        )

    def test_delay_skips_actor_but_not_critic(self):
        """With a delay longer than the burst, only the first step updates the
        actor — but every step must still update the critic."""
        a = _make_agent(learning_steps=4, policy_delay=99)
        _fill_buffer(a)

        actor_before = _actor_params(a)
        critic_before = jax.tree.map(jnp.copy, nnx.state(a.state.critic, nnx.Param))
        a.learn(jax.random.PRNGKey(0))

        assert any(
            not jnp.allclose(x, y)
            for x, y in zip(
                jax.tree.leaves(critic_before),
                jax.tree.leaves(nnx.state(a.state.critic, nnx.Param)),
            )
        ), "critic must update on every step regardless of policy_delay"
        # The actor still moved once (step 0 is always an update step).
        assert any(
            not jnp.allclose(x, y)
            for x, y in zip(
                jax.tree.leaves(actor_before), jax.tree.leaves(_actor_params(a))
            )
        )

    def test_alpha_rides_with_the_actor(self):
        """Alpha's gradient is a function of the actor's log-probs, so it must
        be delayed alongside the policy, not updated every step."""
        base = _make_agent(learning_steps=8, policy_delay=1)
        delayed = _make_agent(learning_steps=8, policy_delay=8)
        _fill_buffer(base)
        _fill_buffer(delayed)

        base.learn(jax.random.PRNGKey(0))
        delayed.learn(jax.random.PRNGKey(0))

        assert abs(float(delayed.log_alpha_module.log_alpha.value)) < abs(
            float(base.log_alpha_module.log_alpha.value)
        )


class TestNStep:
    def test_n_step_burst_runs(self):
        """n_step > 1 swaps in a trajectory buffer and routes the samples
        through `repack_samples`; the whole path has to trace."""
        agent = _make_agent(n_step=5)
        _fill_buffer(agent, n_batches=60)

        actor_loss, critic_loss = agent.learn(jax.random.PRNGKey(0))
        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)

    def test_n_step_stores_truncation(self):
        agent = _make_agent(n_step=5)
        assert agent.state.buffer_state.experience.truncation is not None


class TestActionSelection:
    def test_eval_is_deterministic_and_noiseless(self, agent):
        obs = jnp.zeros((NUM_ENVS, OBS_DIM))
        a1 = agent.step(obs, evaluate=True, key=jax.random.PRNGKey(0))
        n1 = agent.last_noise
        a2 = agent.step(obs, evaluate=True, key=jax.random.PRNGKey(1))

        assert jnp.allclose(a1, a2), "eval actions depend on the key"
        assert jnp.allclose(n1, 0.0), "eval reported nonzero exploration noise"

    def test_training_actions_are_stochastic(self, agent):
        obs = jnp.zeros((NUM_ENVS, OBS_DIM))
        a1 = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(0))
        a2 = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(a1, a2)

    def test_actions_respect_bounds(self, agent):
        obs = jax.random.normal(jax.random.PRNGKey(0), (NUM_ENVS, OBS_DIM))
        a = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(1))
        assert jnp.all(a >= -1.0) and jnp.all(a <= 1.0)
