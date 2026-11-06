"""`impl=batch`: upstream playground tasks with physics from `mujoco.batch`."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from roxie.environment import suites
from roxie.environment.loader import build_playground_env, load_playground_env
from roxie.environment.vector import BatchVectorEnv

NUM_ENVS = 2


def _actions(env, seed=0):
    size = env.single_action_space.shape[0]
    return jnp.asarray(
        np.random.default_rng(seed).uniform(-1, 1, (NUM_ENVS, size)), jnp.float32,
    )


@pytest.mark.parametrize("task", suites.DMC_TASKS)
def test_one_step_matches_mjx(task):
    """Same reset, action and task code, so only the physics engine differs.

    A field a task reads that the batch does not return stays at its reset value,
    and shows up here as a wrong observation or reward.
    """
    mjx = build_playground_env(task, num_envs=NUM_ENVS, impl="jax").env
    native = build_playground_env(
        task, num_envs=NUM_ENVS, impl="batch", num_threads=1,
    ).env
    key = jax.random.PRNGKey(0)
    state, _ = mjx.reset(key)
    actions = _actions(mjx)

    _, want = mjx.step(state, actions, key)
    _, got = native.step(state, actions, key)

    np.testing.assert_allclose(got.obs, want.obs, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(got.reward, want.reward, atol=1e-3)


def _stepped_walker():
    """A walker one step in, with the carry the next step would be handed."""
    env = build_playground_env(
        "WalkerWalk", num_envs=NUM_ENVS, impl="batch", num_threads=1,
    ).env
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)
    state, _ = env.step(state, _actions(env), key)
    carried = {name: getattr(state.env_state.data, name) for name in env._last}
    return env, carried


def test_an_unchanged_world_keeps_its_float64_state():
    """Reloading the carried float32 every step would discard the low bits."""
    env, carried = _stepped_walker()
    reference, _ = _stepped_walker()
    env._rows["qvel"][:] += 1e-12

    action = np.zeros((NUM_ENVS, 6))
    env._step_physics(carried, action)
    reference._step_physics(carried, action)

    assert not np.array_equal(env._rows["qpos"], reference._rows["qpos"])


def test_only_the_replaced_world_is_reloaded():
    """An auto-reset edits one world of the carry; the others must keep running."""
    env, carried = _stepped_walker()
    reference, _ = _stepped_walker()
    moved = {**carried, "qpos": carried["qpos"].at[1, 2].add(0.5)}

    action = np.zeros((NUM_ENVS, 6))
    env._step_physics(moved, action)
    reference._step_physics(carried, action)

    np.testing.assert_array_equal(env._rows["qpos"][0], reference._rows["qpos"][0])
    assert not np.allclose(env._rows["qpos"][1], reference._rows["qpos"][1])


def test_a_task_outside_the_suite_is_refused():
    with pytest.raises(ValueError, match="dm_control suite"):
        build_playground_env("Go1JoystickFlatTerrain", impl="batch")


def test_a_task_that_transforms_its_action_is_refused():
    """The batch steps before the task runs, so it would have applied the wrong
    controls without a word."""
    func_env, _ = load_playground_env("WalkerWalk", impl="batch")
    task = func_env.env

    class Scaling:
        def __getattr__(self, name):
            return getattr(task, name)

        def step(self, state, action):
            return task.step(state, 0.5 * action)

    func_env.env = Scaling()
    env = BatchVectorEnv(func_env, NUM_ENVS, num_threads=1)
    key = jax.random.PRNGKey(0)
    state, _ = env.reset(key)
    with pytest.raises(ValueError, match="unchanged"):
        env.step(state, _actions(env), key)


def test_a_batch_run_replays_on_mjx():
    from omegaconf import OmegaConf

    from roxie.utils.playback import resolve_single_world

    cfg = OmegaConf.create({"env": {"impl": "batch", "num_threads": 10}})
    env = resolve_single_world(cfg).env
    assert env.impl == "jax"
    assert "num_threads" not in env
