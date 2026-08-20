"""End-to-end guards for the fused MPO gradient burst.

MPO used to run its `learning_steps` as a Python loop of separate `nnx.jit`
dispatches; it now fuses them into one `lax.scan` with a donated train state,
like SAC/TD3/DDPG. The Lagrange duals (temperature + the two decoupled KL
multipliers) and their Adam slots ride in the scan carry, which is exactly the
kind of thing that silently stops updating (or fails to trace) without a real
burst exercising it — hence these tests rather than unit tests of the loss
functions alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from roxie.agents.mpo import MPO
from roxie.agents.utils import Transition

OBS_DIM, ACT_DIM, NUM_ENVS = 6, 3, 8


def _make_agent(**kwargs):
    cfg = lambda d: OmegaConf.create(d)  # noqa: E731
    return MPO(
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
        **{"learning_steps": 4, "num_action_samples": 5, **kwargs},
    )


@pytest.fixture
def agent():
    return _make_agent()


def _fill_buffer(agent, n_batches=40, seed=0):
    rng = np.random.default_rng(seed)
    for _ in range(n_batches):
        experience = Transition(
            observation=jnp.asarray(
                rng.standard_normal((NUM_ENVS, OBS_DIM)), jnp.float32
            ),
            action=jnp.asarray(rng.standard_normal((NUM_ENVS, ACT_DIM)), jnp.float32),
            reward=jnp.asarray(rng.standard_normal(NUM_ENVS), jnp.float32),
            terminal=jnp.zeros(NUM_ENVS, jnp.bool_),
        )
        agent.state.buffer_state = agent.replay.add(
            agent.state.buffer_state, experience
        )


def _actor_params(agent):
    return jax.tree.map(jnp.copy, nnx.state(agent.state.actor, nnx.Param))


class TestFusedBurst:
    def test_burst_runs_and_moves_params(self, agent):
        _fill_buffer(agent)
        before = _actor_params(agent)

        actor_loss, critic_loss = agent._learn(jax.random.PRNGKey(0))

        assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)
        moved = [
            not jnp.allclose(a, b)
            for a, b in zip(
                jax.tree.leaves(before), jax.tree.leaves(_actor_params(agent))
            )
        ]
        assert any(moved), "actor params unchanged after a gradient burst"

    def test_target_networks_track_online_networks(self, agent):
        """The soft updates run inside the scan body, so a burst of N steps must
        move both targets N times — not once, and not zero times."""
        _fill_buffer(agent)
        before = {
            name: jax.tree.map(jnp.copy, nnx.state(getattr(agent.state, name), nnx.Param))
            for name in ("target_actor", "target_critic")
        }
        agent._learn(jax.random.PRNGKey(0))

        for name, old in before.items():
            after = nnx.state(getattr(agent.state, name), nnx.Param)
            assert any(
                not jnp.allclose(a, b)
                for a, b in zip(jax.tree.leaves(old), jax.tree.leaves(after))
            ), f"{name} unchanged after a gradient burst"

    def test_duals_update_every_step_of_the_burst(self):
        """The temperature and KL multipliers live in the scan carry. If they
        were closed over instead, they would advance by one step per burst no
        matter how long the burst is — this pins the difference."""
        short = _make_agent(learning_steps=1)
        long = _make_agent(learning_steps=8)
        _fill_buffer(short)
        _fill_buffer(long)

        short._learn(jax.random.PRNGKey(0))
        long._learn(jax.random.PRNGKey(0))

        init = float(_make_agent().dual_params.log_temperature.value)
        t_short = float(short.dual_params.log_temperature.value)
        t_long = float(long.dual_params.log_temperature.value)

        assert t_short != pytest.approx(init), "temperature did not update at all"
        assert abs(t_long - init) > abs(t_short - init), (
            f"8-step burst moved the temperature to {t_long}, no further than a "
            f"1-step burst {t_short}: the duals are not carried through the scan"
        )

    def test_dual_optimizer_state_advances(self, agent):
        """The dual Adam slots ride in the carry too; if they were dropped the
        step count would stay at one per burst."""
        _fill_buffer(agent)
        agent._learn(jax.random.PRNGKey(0))

        counts = [
            int(leaf)
            for leaf in jax.tree.leaves(nnx.state(agent.dual_optimizer))
            if jnp.asarray(leaf).dtype == jnp.int32 and jnp.asarray(leaf).ndim == 0
        ]
        assert agent.learning_steps in counts, (
            f"dual optimizer step count {counts} does not reflect a "
            f"{agent.learning_steps}-step burst"
        )

    def test_state_is_reusable_after_donation(self, agent):
        """`_mpo_grad_steps` donates the train state. Consecutive bursts must
        keep working — i.e. the agent really does rebind `self.state` to the
        result rather than reading the donated (now invalid) input."""
        _fill_buffer(agent)
        for i in range(3):
            actor_loss, critic_loss = agent._learn(jax.random.PRNGKey(i))
            assert jnp.isfinite(actor_loss) and jnp.isfinite(critic_loss)

    def test_no_public_learn_hook(self, agent):
        """`Trainer._run` routes any agent exposing a public `learn` through the
        async learner, which also calls `select_action` / `add_transitions`. MPO
        implements neither, so the burst entry point stays private — otherwise
        `async_learner=True` would crash instead of falling back to sync."""
        assert not hasattr(agent, "learn"), (
            "MPO exposes `learn` but not the rest of the async-learner contract"
        )
        assert not hasattr(agent, "select_action")

    def test_update_gate_respects_schedule(self, agent):
        """One burst per `steps_between_updates` of ELAPSED env steps.

        This used to assert that `update(105)` fires nothing because 105 is not
        exactly a boundary. That is the behaviour that broke the v1 release grid:
        the trainer advances `steps` in strides of `parallel_envs` and only ever
        lands exactly on a boundary when the stride divides the offset, so an
        unaligned `steps_before_learning` disarmed learning completely. The gate
        now serves a boundary on the first call at or past it — see
        tests/test_update_schedule.py.
        """
        _fill_buffer(agent)
        agent.steps_before_learning = 100
        agent.steps_between_updates = 10

        # Before warmup: nothing, regardless of where the stride lands.
        assert agent.update(99, jax.random.PRNGKey(0))[0] == 0
        # First call past warmup serves the boundary at 100 even though the
        # stride overshot it by 5.
        assert agent.update(105, jax.random.PRNGKey(0))[0] == agent.learning_steps
        # Still inside the same window — no second burst.
        assert agent.update(108, jax.random.PRNGKey(0))[0] == 0
        # Next window opens at 110.
        assert agent.update(110, jax.random.PRNGKey(0))[0] == agent.learning_steps


class TestActionSelection:
    def test_eval_is_deterministic(self, agent):
        obs = jnp.zeros((NUM_ENVS, OBS_DIM))
        a1 = agent.step(obs, evaluate=True, key=jax.random.PRNGKey(0))
        a2 = agent.step(obs, evaluate=True, key=jax.random.PRNGKey(1))
        assert jnp.allclose(a1, a2), "eval actions depend on the key"

    def test_training_actions_are_stochastic(self, agent):
        obs = jnp.zeros((NUM_ENVS, OBS_DIM))
        a1 = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(0))
        a2 = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(1))
        assert not jnp.allclose(a1, a2)

    def test_actions_respect_bounds(self, agent):
        obs = jax.random.normal(jax.random.PRNGKey(0), (NUM_ENVS, OBS_DIM))
        a = agent.step(obs, evaluate=False, key=jax.random.PRNGKey(1))
        assert jnp.all(a >= -1.0) and jnp.all(a <= 1.0)
