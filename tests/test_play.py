"""The steps `roxie/play.py` runs before it opens a window."""

import jax
import jax.numpy as jnp
import pytest


@pytest.fixture
def play(monkeypatch):
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("XLA_FLAGS", "")
    from roxie import play

    return play


class _Agent:
    def __init__(self):
        self.calls = 0

    def select_action(self, obs, key, evaluate):
        self.calls += 1
        return jnp.zeros((1, 2), dtype=jnp.float32), None, None


class _State:
    obs = jnp.zeros(3, dtype=jnp.float32)


def test_warming_up_runs_a_full_step_of_the_world(play):
    stepped = []
    world = play._World(
        observe=lambda obs: obs, ghost=None, model=None, data=None, step_dt=0.01,
        reset=lambda key: _State(),
        step=lambda state, action: stepped.append(action) or state,
        env_nq=0, env_nbody=0,
    )
    agent = _Agent()
    play._warm_up(world, agent, jax.random.PRNGKey(0))
    assert agent.calls == 1
    assert stepped[0].shape == (2,)


def test_up_and_down_cycle_the_tasks_and_flag_the_change(play):
    controls = play._Controls(("walk", "run", "stand"), 0)
    controls.key_callback(play._KEY_UP)
    assert controls.task == 2 and controls.task_changed
    controls.task_changed = False
    controls.key_callback(play._KEY_DOWN)
    assert controls.task == 0 and controls.task_changed


def test_a_single_task_ignores_the_task_keys(play):
    controls = play._Controls((None,), 0)
    controls.key_callback(play._KEY_DOWN)
    assert controls.task == 0 and not controls.task_changed


def test_space_pauses_and_delete_asks_for_one_reset(play):
    controls = play._Controls((None,), 0)
    controls.key_callback(play._KEY_SPACE)
    controls.key_callback(play._KEY_DELETE)
    assert controls.paused
    assert controls.take_reset_request()
    assert not controls.take_reset_request()


def _model(bodies):
    import mujoco

    worldbody = "".join(
        f'<body pos="{i} 0 1"><freejoint/><geom size=".1"/></body>'
        for i in range(bodies)
    )
    return mujoco.MjModel.from_xml_string(
        f"<mujoco><worldbody>{worldbody}</worldbody></mujoco>"
    )


def test_the_first_window_follows_the_root_body(play):
    import mujoco

    camera = mujoco.MjvCamera()
    play._aim_camera(camera, _model(1), None)
    assert camera.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert camera.trackbodyid == 1


def test_the_next_window_keeps_the_view_or_frees_a_missing_body(play):
    import mujoco

    previous = mujoco.MjvCamera()
    previous.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    previous.trackbodyid = 2
    previous.distance = 7.0

    kept = mujoco.MjvCamera()
    play._aim_camera(kept, _model(2), previous)
    assert kept.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert kept.trackbodyid == 2 and kept.distance == 7.0

    freed = mujoco.MjvCamera()
    play._aim_camera(freed, _model(1), previous)
    assert freed.type == mujoco.mjtCamera.mjCAMERA_FREE
    assert freed.distance == 7.0
