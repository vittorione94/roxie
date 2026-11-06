"""The replay window, in a hidden window, driven through its input callbacks."""

import ctypes

import glfw
import mujoco
import numpy as np
import pytest

from roxie.utils import mjui
from roxie.utils.viewer import Viewer

ONE_BODY = """
<mujoco><worldbody>
  <body pos="0 0 1"><freejoint/><geom size=".5"/></body>
</worldbody></mujoco>"""
THREE_BODIES = """
<mujoco><worldbody>
  <body pos="0 0 1"><freejoint/><geom size=".5"/>
    <body pos="1 0 0"><joint/><geom size=".1"/></body>
  </body>
  <body pos="0 2 1"><freejoint/><geom size=".1"/></body>
</worldbody></mujoco>"""
TASKS = ("stand", "walk", "run")


def _world(xml):
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture
def viewer():
    if not glfw.init():
        pytest.skip("no display for GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    try:
        window = Viewer(*_world(THREE_BODIES), tasks=TASKS)
    except RuntimeError as error:
        pytest.skip(str(error))
    window.draw()
    yield window
    window.close()


def _move(viewer, x, y_from_top):
    viewer._on_cursor(None, x, y_from_top)


def _press(viewer, button=glfw.MOUSE_BUTTON_LEFT, mods=0):
    viewer._on_button(None, button, glfw.PRESS, mods)


def _release(viewer, button=glfw.MOUSE_BUTTON_LEFT):
    viewer._on_button(None, button, glfw.RELEASE, 0)


def _key(viewer, key):
    viewer._on_key(None, key, 0, glfw.PRESS, 0)


def _item(viewer, section, index):
    return viewer._ui.sect[section].item[index]


def _click_item(viewer, item, element=0, elements=1):
    """Clicks a widget at the window position the panel draws it."""
    rect = item.rect
    # Widget rects count upward from the panel's current height.
    inside = rect.bottom + rect.height * (1.0 - (element + 0.5) / elements)
    _move(viewer, rect.left + rect.width / 2, viewer._ui.height - inside)
    _press(viewer)
    _release(viewer)


def _click_task(viewer, element):
    _click_item(viewer, _item(viewer, 0, 0), element, len(TASKS))


def _scene_centre(viewer):
    rect = viewer._state.rect[2]
    return rect.left + rect.width / 2, rect.bottom + rect.height / 2


def test_the_camera_starts_on_the_root_body_and_esc_frees_it(viewer):
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.cam.trackbodyid == 1
    _key(viewer, glfw.KEY_ESCAPE)
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE


def test_clicking_a_task_reports_it_once(viewer):
    _click_task(viewer, 1)
    assert viewer.take_clicked_task() == "walk"
    assert viewer.take_clicked_task() is None
    _click_task(viewer, 2)
    assert viewer.take_clicked_task() == "run"


def test_clicking_the_chosen_task_reports_it_again(viewer):
    _click_task(viewer, 0)
    assert viewer.take_clicked_task() == "stand"
    _click_task(viewer, 0)
    assert viewer.take_clicked_task() == "stand"


def test_with_one_task_there_is_no_tasks_section():
    if not glfw.init():
        pytest.skip("no display for GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    with Viewer(*_world(ONE_BODY), tasks=(None,)) as window:
        assert window._ui.sect[0].name == b"Simulation"


def test_the_simulation_section_pauses_resumes_and_resets(viewer):
    run_radio, reset = _item(viewer, 1, 0), _item(viewer, 1, 1)
    assert not viewer.paused
    _click_item(viewer, run_radio, 0, 2)
    assert viewer.paused
    _click_item(viewer, run_radio, 1, 2)
    assert not viewer.paused
    assert not viewer.take_reset_request()
    _click_item(viewer, reset)
    assert viewer.take_reset_request()
    assert not viewer.take_reset_request()


def test_keys_pause_reset_and_hide_the_panel(viewer):
    _key(viewer, glfw.KEY_SPACE)
    assert viewer.paused
    viewer.resume()
    assert not viewer.paused
    _key(viewer, glfw.KEY_DELETE)
    assert viewer.take_reset_request()
    assert viewer._state.rect[1].width > 0
    _key(viewer, glfw.KEY_TAB)
    assert viewer._state.rect[1].width == 0
    assert viewer._state.rect[2].left == 0


def test_a_flag_shortcut_reaches_the_option_it_names(viewer):
    index = next(
        i for i, row in enumerate(mujoco.mjVISSTRING) if row[0] == "Contact Point"
    )
    before = int(viewer.opt.flags[index])
    _key(viewer, ord(mujoco.mjVISSTRING[index][2]))
    assert int(viewer.opt.flags[index]) != before


def test_a_mouse_drag_in_the_scene_moves_the_camera(viewer):
    x, y = _scene_centre(viewer)
    _move(viewer, x, y)
    azimuth = viewer.cam.azimuth
    _press(viewer)
    _move(viewer, x + 100, y)
    _release(viewer)
    assert viewer.cam.azimuth != azimuth
    azimuth = viewer.cam.azimuth
    _move(viewer, x + 300, y)
    assert viewer.cam.azimuth == azimuth


def test_a_drag_starting_on_the_panel_leaves_the_camera_alone(viewer):
    _move(viewer, 5, 5)
    azimuth = viewer.cam.azimuth
    _press(viewer)
    x, y = _scene_centre(viewer)
    _move(viewer, x, y)
    _release(viewer)
    assert viewer.cam.azimuth == azimuth


def test_scrolling_zooms_only_over_the_scene(viewer):
    _move(viewer, 5, 5)
    distance = viewer.cam.distance
    viewer._on_scroll(None, 0.0, 1.0)
    assert viewer.cam.distance == distance
    x, y = _scene_centre(viewer)
    _move(viewer, x, y)
    viewer._on_scroll(None, 0.0, 1.0)
    assert viewer.cam.distance != distance


def test_double_clicking_a_body_selects_it_and_ctrl_drag_pushes_it(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.lookat = (0.0, 0.0, 1.0)
    x, y = _scene_centre(viewer)
    _move(viewer, x, y)
    viewer.draw()
    _press(viewer)
    _release(viewer)
    _press(viewer)
    _release(viewer)
    assert viewer.perturb.select == 1

    viewer._state.buttontime = -10.0
    _press(viewer, glfw.MOUSE_BUTTON_RIGHT, glfw.MOD_CONTROL)
    assert viewer.perturb.active == mujoco.mjtPertBit.mjPERT_TRANSLATE
    azimuth = viewer.cam.azimuth
    _move(viewer, x + 50, y)
    assert viewer.cam.azimuth == azimuth
    _release(viewer, glfw.MOUSE_BUTTON_RIGHT)
    assert viewer.perturb.active == 0


def test_a_prepared_model_is_shown_without_building_anything(viewer, monkeypatch):
    model, data = _world(ONE_BODY)
    viewer.prepare(model)

    def refuse(*args):
        raise AssertionError("built a render context on a task switch")

    monkeypatch.setattr(mujoco, "MjrContext", refuse)
    viewer.show(model, data)
    viewer.draw()


def test_going_back_to_a_model_reuses_its_renderer(viewer):
    first_context = viewer._context
    first_model, first_data = viewer._model, viewer._data
    viewer.show(*_world(ONE_BODY))
    assert viewer._context is not first_context
    viewer.show(first_model, first_data)
    assert viewer._context is first_context
    viewer.draw()


def test_a_new_model_keeps_the_camera_and_the_tracked_body_it_has(viewer):
    viewer.cam.distance = 7.5
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 2
    viewer.perturb.select = 2
    viewer.show(*_world(THREE_BODIES))
    viewer.draw()
    assert viewer.cam.distance == 7.5
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_TRACKING
    assert viewer.perturb.select == 0


def test_a_new_model_without_the_tracked_body_frees_the_camera(viewer):
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    viewer.cam.trackbodyid = 2
    viewer.show(*_world(ONE_BODY))
    # Aborts the process if the id reaches mjv_updateCamera out of range.
    viewer.draw()
    assert viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE


def test_the_panel_flags_stay_bound_across_a_new_model(viewer):
    index = next(
        i for i, row in enumerate(mujoco.mjRNDSTRING) if row[0] == "Shadow"
    )
    item = viewer._ui.sect[2].item[index]
    assert item.pdata == viewer._render_flags.ctypes.data + index
    viewer.show(*_world(ONE_BODY))
    viewer._render_flags[index] = 0
    viewer.draw()
    assert viewer._scene.flags[index] == 0


def test_closing_twice_is_harmless():
    if not glfw.init():
        pytest.skip("no display for GLFW")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = Viewer(*_world(ONE_BODY))
    window.close()
    window.close()


def test_a_mujoco_with_another_mjui_h_is_refused(monkeypatch):
    monkeypatch.setattr(mjui, "_MJUI_H_SHA256", "not the hash")
    with pytest.raises(RuntimeError, match="different mjui.h"):
        mjui.load()


def test_the_ctypes_structs_are_the_sizes_the_header_gives():
    """Measured with gcc against MuJoCo 3.9's own mjui.h."""
    assert ctypes.sizeof(mjui.Item) == 1504
    assert ctypes.sizeof(mjui.UI) == 3009832
