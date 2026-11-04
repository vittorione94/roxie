"""The world model: what it fits, what it masks, what it hands an agent.

Built through the shipped yaml via the harness, shrunk to CPU size, so the
fitting pass and the checkpoint are exercised exactly as `train.py` builds
them.
"""

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from roxie.agents.utils import Transition
from roxie.losses.world_losses import window_masks, world_rollout_loss_fn
from roxie.worlds import build_world
from tests import harness
from tests.harness import ACT, ENVS, OBS


def _linear_block(key, steps):
    """A `(T, B, ...)` block of a known linear system, `x' = A x + B u`.

    Small, stable dynamics with a reward that is a fixed function of the state,
    so a model that fits has nothing to explain but what is here.
    """
    a_key, b_key, x_key, u_key = jax.random.split(key, 4)
    # A contraction (0.9 times a rotation), so the state stays O(1) over the
    # block: an unstable `A` makes the reward term blow up and own the clip.
    q, _ = jnp.linalg.qr(jax.random.normal(a_key, (OBS, OBS)))
    a = 0.9 * q
    b = 0.3 * jax.random.normal(b_key, (ACT, OBS))
    x = jax.random.normal(x_key, (ENVS, OBS))
    actions = jax.random.uniform(u_key, (steps, ENVS, ACT), minval=-1, maxval=1)

    obs, rewards = [x], []
    for t in range(steps):
        rewards.append(-jnp.mean(jnp.square(x), axis=-1))
        x = x @ a.T + actions[t] @ b
        obs.append(x)
    obs = jnp.stack(obs)
    false = jnp.zeros((steps, ENVS), jnp.bool_)
    return Transition(
        observation=obs[:-1], action=actions, reward=jnp.stack(rewards),
        terminal=false, truncation=false,
    ), obs[1:]


def test_the_fit_learns_a_linear_system():
    # Wider than the harness default: a 16-unit trunk underfits even this.
    cfg = harness.world_config(learning_steps=50, learning_rate=3e-3)
    cfg.dynamics_config.features = [64, 64]
    world = build_world(cfg, env_obs_size=OBS, env_action_size=ACT)
    world.fill_buffer(*_linear_block(jax.random.PRNGKey(0), steps=48))

    world.learn(jax.random.PRNGKey(1))
    first = world.pop_diagnostics()
    for i in range(8):
        world.learn(jax.random.PRNGKey(2 + i))
    last = world.pop_diagnostics()

    # Measured ~0.05x (state) and ~0.1x (reward) of the first pass.
    for key in ("world/state_loss", "world/reward_loss"):
        assert last[key] < 0.3 * first[key], (
            f"{key}: {first[key]:.4f} -> {last[key]:.4f} — the model is not "
            f"fitting a linear system"
        )


def _window(dones_at=()):
    """A hand-built `(B=2, k+1=4)` window; row 0 ends at each step in `dones_at`."""
    key = jax.random.PRNGKey(3)
    obs_key, act_key, rew_key = jax.random.split(key, 3)
    terminal = np.zeros((2, 4), bool)
    for j in dones_at:
        terminal[0, j] = True
    return Transition(
        observation=jax.random.normal(obs_key, (2, 4, OBS)),
        action=jax.random.uniform(act_key, (2, 4, ACT), minval=-1, maxval=1),
        reward=jax.random.normal(rew_key, (2, 4)),
        terminal=jnp.asarray(terminal),
        truncation=jnp.zeros((2, 4), jnp.bool_),
    )


def _loss(world, window):
    loss, _ = world_rollout_loss_fn(
        world.state.dynamics, window, jnp.zeros(OBS), jnp.ones(OBS), 1.0, 1.0,
    )
    return float(loss)


def test_the_masks_follow_the_episode_the_window_started_in():
    terminal = jnp.asarray([[False, True, False], [False, False, False]])
    step_valid, state_valid = window_masks(terminal, jnp.zeros_like(terminal))
    # Step 1 ends row 0: its own reward and label still count, the state it
    # "lands on" is the next episode's first, and nothing after it counts.
    np.testing.assert_array_equal(step_valid, [[1, 1, 0], [1, 1, 1]])
    np.testing.assert_array_equal(state_valid, [[1, 0, 0], [1, 1, 1]])


def test_nothing_after_an_episode_ends_reaches_the_loss():
    """The pool auto-resets, so the observations after a done belong to a new
    episode. Overwriting them with garbage must not move the loss — while
    touching the step BEFORE the done must, or the check is vacuous."""
    world = harness.build_world_model()
    window = _window(dones_at=(1,))
    base = _loss(world, window)

    garbage = window.replace(
        observation=window.observation.at[0, 2:].set(1e3),
        reward=window.reward.at[0, 2:].set(-1e3),
    )
    assert _loss(world, garbage) == pytest.approx(base, rel=1e-6)

    valid_target = window.replace(
        observation=window.observation.at[0, 1].add(5.0)
    )
    assert _loss(world, valid_target) != pytest.approx(base, rel=1e-3)


def test_step_is_differentiable_in_the_state_and_the_action():
    world = harness.build_world_model()
    mean, std = world.normalizer(world.state.obs_stats)
    obs = jax.random.normal(jax.random.PRNGKey(4), (ENVS, OBS))
    action = jnp.zeros((ENVS, ACT))

    def scalar(obs, action):
        next_obs, reward, term = world.step(
            world.state.dynamics, mean, std, obs, action,
        )
        return jnp.sum(next_obs) + jnp.sum(reward) + jnp.sum(term)

    d_obs, d_action = jax.grad(scalar, argnums=(0, 1))(obs, action)
    for name, grad in (("obs", d_obs), ("action", d_action)):
        assert np.all(np.isfinite(grad)), f"d/d{name} is not finite"
        assert np.any(grad != 0), f"d/d{name} is identically zero"


def test_an_untrained_model_predicts_nothing_moves_and_nothing_ends():
    world = harness.build_world_model()
    mean, std = world.normalizer(world.state.obs_stats)
    obs = jax.random.normal(jax.random.PRNGKey(5), (ENVS, OBS))
    next_obs, _, term = world.step(
        world.state.dynamics, mean, std, obs, jnp.zeros((ENVS, ACT)),
    )
    assert float(jnp.max(jnp.abs(next_obs - obs))) < 0.5
    assert float(jnp.max(term)) < 0.02


def test_consecutive_passes_survive_the_donated_state():
    agent, world, _ = harness.warmed_with_world()
    del agent
    for i in range(2):
        steps, loss = world.learn(jax.random.PRNGKey(i))
        assert steps == world.hp.learning_steps and np.isfinite(float(loss))
    # Still readable: a donated-and-dropped buffer would raise here.
    assert np.isfinite(np.asarray(world.state.buffer_state.experience.reward)).all()


def test_diagnostics_arrive_namespaced_and_finite():
    _agent, world, steps = harness.warmed_with_world()
    world.learn(jax.random.PRNGKey(0))
    out = world.pop_diagnostics(steps)
    assert out and all(k.startswith("world/") for k in out)
    assert all(np.isfinite(v) for v in out.values())
    assert world.pop_diagnostics(steps) == {}


# --- checkpoints -----------------------------------------------------------


def _save(world, path, *, include_buffer):
    """One self-contained checkpoint carrying only the world's half."""
    payload = {"world": world.checkpoint_payload(include_buffer=include_buffer)}
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save(Path(path).resolve(), payload)


def _params(world):
    return harness.snapshot(world.state.dynamics)


def test_a_round_trip_restores_the_model_the_stats_and_the_buffer(tmp_path):
    _agent, world, _ = harness.warmed_with_world()
    world.learn(jax.random.PRNGKey(0))
    _save(world, tmp_path / "ckpt", include_buffer=True)

    fresh = harness.build_world_model()
    assert harness.moved(_params(world), fresh.state.dynamics)
    assert fresh.restore(tmp_path / "ckpt") is True

    assert not harness.moved(_params(world), fresh.state.dynamics)
    np.testing.assert_array_equal(
        np.asarray(world.state.obs_stats.sum),
        np.asarray(fresh.state.obs_stats.sum),
    )
    np.testing.assert_array_equal(
        np.asarray(world.state.buffer_state.experience.observation),
        np.asarray(fresh.state.buffer_state.experience.observation),
    )


def test_without_the_buffer_the_trainer_is_told_to_refill(tmp_path):
    _agent, world, _ = harness.warmed_with_world()
    _save(world, tmp_path / "ckpt", include_buffer=False)
    assert harness.build_world_model().restore(tmp_path / "ckpt") is False


def test_a_checkpoint_without_a_world_is_refused(tmp_path):
    agent, _steps = harness.warmed("td3")
    payload = agent.checkpoint_payload()
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save((tmp_path / "ckpt").resolve(), payload)
    with pytest.raises(ValueError, match="carries no world model"):
        harness.build_world_model().restore(tmp_path / "ckpt")


def test_a_world_trained_on_another_width_is_refused(tmp_path):
    _agent, world, _ = harness.warmed_with_world()
    payload = {"world": world.checkpoint_payload()}
    payload["world"]["hyperparams"]["env_obs_size"] = OBS + 1
    with ocp.StandardCheckpointer() as checkpointer:
        checkpointer.save((tmp_path / "ckpt").resolve(), payload)
    with pytest.raises(ValueError, match="dimensional"):
        harness.build_world_model().restore(tmp_path / "ckpt")
