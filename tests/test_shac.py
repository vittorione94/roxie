"""SHAC: a deterministic actor trained by backpropagating through a world model.

Built with its world through the harness, from the shipped yaml shrunk to CPU
size. What these pin is the contract between the two: the pass trains the
agent and leaves the world untouched, the actor's gradient really does arrive
through the model, and the pair refuses to be split.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from roxie.losses.actor_losses import bptt_actor_loss_fn
from roxie.utils.math import scale_to_env
from roxie.utils.trainer import Trainer
from tests import harness
from tests.harness import ACT, ENVS, OBS


@pytest.fixture(scope="module")
def trained():
    """One SHAC agent and its world after a fit and a learning pass each.

    Returns `(agent, world, before, steps, diagnostics)`, `before` holding
    parameter snapshots from just ahead of the agent's pass.
    """
    agent, world, steps = harness.warmed_with_world()
    world.learn(jax.random.PRNGKey(0))
    world.pop_diagnostics()
    before = {
        "actor": harness.snapshot(agent.state.actor),
        "critic": harness.snapshot(agent.state.critic),
        "dynamics": harness.snapshot(world.state.dynamics),
    }
    buffer = np.asarray(world.state.buffer_state.experience.observation)
    gradient_steps, actor_loss, critic_loss = agent.learn(
        jax.random.PRNGKey(1), world=world,
    )
    assert gradient_steps == agent.hp.learning_steps
    assert np.isfinite(float(actor_loss)) and np.isfinite(float(critic_loss))
    before["buffer"] = buffer
    return agent, world, before, steps, agent.pop_diagnostics(steps)


def test_a_pass_trains_the_agent_and_leaves_the_world_alone(trained):
    agent, world, before, _, _ = trained
    assert harness.moved(before["actor"], agent.state.actor)
    assert harness.moved(before["critic"], agent.state.critic)
    assert not harness.moved(before["dynamics"], world.state.dynamics)
    # Read, not donated: the world's buffer survives the agent's pass intact.
    np.testing.assert_array_equal(
        before["buffer"],
        np.asarray(world.state.buffer_state.experience.observation),
    )


def test_the_agent_holds_no_buffer_of_its_own(trained):
    agent, _, _, _, _ = trained
    assert agent.replay is None and agent.state.buffer_state is None
    assert agent.checkpoint_payload(include_buffer=True)["trainstate_state"].get(
        "buffer_state"
    ) is None


def test_diagnostics_arrive_namespaced_and_finite(trained):
    agent, _, _, steps, out = trained
    assert out and all(k.startswith("shac/") for k in out)
    assert all(np.isfinite(v) for v in out.values()), out
    for key in ("imagined_return", "survival_H", "value_H", "value_ev"):
        assert f"shac/{key}" in out
    assert agent.pop_diagnostics(steps) == {}


def _actor_grad(agent, world, dynamics):
    """d(imagined loss)/d(actor) from a fixed batch of starts."""
    starts = jax.random.normal(jax.random.PRNGKey(7), (ENVS, OBS))
    mean, std = world.normalizer(world.state.obs_stats)

    def policy_fn(actor, obs):
        return scale_to_env(actor(obs), agent.action_low, agent.action_high)

    def model_fn(dynamics, obs, action):
        return world.step(dynamics, mean, std, obs, action)

    def value_fn(critic, obs):
        return critic(obs)[..., 0]

    (_, _), grads = nnx.value_and_grad(
        functools.partial(
            bptt_actor_loss_fn,
            policy_fn=policy_fn, model_fn=model_fn, value_fn=value_fn,
            gamma=0.99, horizon=4,
        ),
        has_aux=True,
    )(agent.state.actor, agent.state.critic, dynamics, starts)
    return jax.tree.leaves(grads)


def test_the_actor_gradient_arrives_through_the_model(trained):
    """Blind the model to the action — zero the trunk's action-input weights —
    and nothing the rollout produces depends on the actor any more: not the
    rewards, not the states, not the bootstrap. The gradient must vanish
    exactly, and must not vanish with the model's sight restored."""
    agent, world, _, _, _ = trained
    dynamics = world.state.dynamics

    seeing = _actor_grad(agent, world, dynamics)
    assert any(float(jnp.max(jnp.abs(g))) > 0 for g in seeing)

    graphdef, state = nnx.split(dynamics)
    blind = nnx.merge(graphdef, state)
    kernel = blind.trunk.hidden_layers[0].kernel
    kernel[...] = kernel[...].at[OBS:].set(0.0)
    for grad in _actor_grad(agent, world, blind):
        np.testing.assert_array_equal(np.asarray(grad), 0.0)


def test_acting_is_bounded_and_deterministic_under_evaluate(trained):
    agent, _, _, _, _ = trained
    obs = jax.random.normal(jax.random.PRNGKey(2), (ENVS, OBS))
    noisy, _, _ = agent.select_action(obs, jax.random.PRNGKey(3))
    assert noisy.shape == (ENVS, ACT)
    assert float(jnp.max(jnp.abs(noisy))) <= 1.0
    a, _, _ = agent.select_action(obs, jax.random.PRNGKey(4), evaluate=True)
    b, _, _ = agent.select_action(obs, jax.random.PRNGKey(5), evaluate=True)
    np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


# --- the pairing -----------------------------------------------------------


def test_a_world_agent_refuses_to_learn_without_its_world(trained):
    agent, _, _, _, _ = trained
    with pytest.raises(ValueError, match="world model"):
        agent.learn(jax.random.PRNGKey(0))


def test_a_model_free_agent_refuses_a_world():
    agent, _steps = harness.warmed("td3")
    world = harness.build_world_model()
    with pytest.raises(ValueError, match="model-free"):
        agent.learn(jax.random.PRNGKey(0), world=world)


def test_the_trainer_refuses_a_mismatched_pair(trained):
    agent, world, _, _, _ = trained
    trainer = Trainer(output_dir="/nonexistent")
    with pytest.raises(ValueError, match="world model"):
        trainer.initialize(agent, environment=None)
    td3, _steps = harness.warmed("td3")
    with pytest.raises(ValueError, match="model-free"):
        trainer.initialize(td3, environment=None, world=world)
    trainer.initialize(agent, environment=None, world=world)
    assert trainer.world is world


# --- the checkpoint ----------------------------------------------------------


def test_a_resume_restores_the_agent_and_its_world(tmp_path, trained):
    """Through `Trainer._save`, the path a run takes: the world rides in the
    agent's payload under its own key, and both halves come back from it."""
    from roxie.utils.checkpoint import find_checkpoint

    agent, world, _, _, _ = trained
    trainer = Trainer(output_dir=str(tmp_path), save_buffer=True)
    trainer.initialize(agent, environment=None, world=world)
    trainer.steps = 1234
    trainer._save(agent, epochs=1, episodes=2, gradient_steps=3)
    trainer._checkpoint_manager.close()

    path = find_checkpoint(tmp_path)
    fresh_agent = harness.build("shac")
    fresh_world = harness.build_world_model()
    metadata = fresh_agent.restore(path)
    assert metadata["steps"] == 1234
    assert fresh_world.restore(path) is True

    for name in ("actor", "critic"):
        assert not harness.moved(
            harness.snapshot(getattr(agent.state, name)),
            getattr(fresh_agent.state, name),
        ), f"{name} did not come back"
    assert not harness.moved(
        harness.snapshot(world.state.dynamics), fresh_world.state.dynamics,
    )
    np.testing.assert_array_equal(
        np.asarray(world.state.buffer_state.experience.observation),
        np.asarray(fresh_world.state.buffer_state.experience.observation),
    )
