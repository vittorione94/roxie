"""The window `roxie/play.py` replays into, with MuJoCo's own widgets for panels.

`mujoco.viewer` cannot be given another button: its panels are compiled into
`_simulate`, with no access from Python. This window draws the same widgets
through `roxie/utils/mjui.py`, and its camera, picking and pushing follow what
`simulate` does with the same mouse gestures.

Everything runs on the calling thread, between the steps of the replay. There is
no render thread to race at shutdown, and on macOS no need for `mjpython`.

Two kinds of render context share the window because `mujoco.MjrContext` hides
its pointer from `mjui_*`. Each model's scene draws with one of its own, kept for
as long as the window. The panels draw with one made through `ctypes`, which only
holds fonts and so has nothing to do with the model on show.
"""

import ctypes

import glfw
import mujoco
import numpy as np

from roxie.utils import mjui

_RECT_WINDOW = 0
_RECT_PANEL = 1
_RECT_SCENE = 2

_AUX_BUFFER = 0
_THEME = 0
_RADIO_COLUMNS = 1
_MAX_GEOM = 10_000
_DOUBLE_CLICK_SECONDS = 0.25
_ZOOM_PER_SCROLL = 0.05
_FALLBACK_FONT_SCALE = 150

# Not vsync: a replay step takes as long as a frame, and a blocking swap would
# hold the world below real time. So frames are paced here and swaps never wait.
_SECONDS_PER_FRAME = 1.0 / 60.0

_MOUSE_BUTTONS = {
    glfw.MOUSE_BUTTON_LEFT: mjui.BUTTON_LEFT,
    glfw.MOUSE_BUTTON_RIGHT: mjui.BUTTON_RIGHT,
    glfw.MOUSE_BUTTON_MIDDLE: mjui.BUTTON_MIDDLE,
}


def _flag_rows(flags: np.ndarray, strings: tuple) -> list[tuple]:
    """Check boxes bound to a flag array, with the shortcuts MuJoCo names."""
    start = flags.ctypes.data
    return [
        (mjui.ITEM_CHECKBYTE, name, 1, start + index, f" {shortcut}" if shortcut else "")
        for index, (name, _default, shortcut) in enumerate(strings)
    ]


class Viewer:
    """An open window drawing one model, with a panel of widgets beside it.

    Attributes:
        cam: the camera the mouse moves.
        opt: the visualization options the panel's check boxes write.
        perturb: the body picked and the push on it, for the replay to apply.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        tasks: tuple[str | None, ...] = (),
    ):
        """Opens the window.

        Args:
            model: the model to draw first.
            data: the buffer to draw it from, kept up to date by the caller.
            tasks: the names to offer in a Tasks section, the first as chosen.
              With fewer than two the section is left out.

        Raises:
            RuntimeError: if no window can be opened.
        """
        self._lib = mjui.load()
        if not glfw.init():
            raise RuntimeError("GLFW could not initialize; is there a display?")
        mode = glfw.get_video_mode(glfw.get_primary_monitor())
        glfw.window_hint(glfw.SAMPLES, 4)
        self._window = glfw.create_window(
            2 * mode.size.width // 3, 2 * mode.size.height // 3, "roxie", None, None,
        )
        if not self._window:
            glfw.terminate()
            raise RuntimeError("GLFW could not open a window.")
        glfw.make_context_current(self._window)
        glfw.swap_interval(0)

        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.perturb = mujoco.MjvPerturb()
        mujoco.mjv_defaultFreeCamera(model, self.cam)
        # A locomotion task leaves a fixed view within seconds. Esc frees it.
        if model.nbody > 1:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = 1
        mujoco.mjv_defaultOption(self.opt)
        mujoco.mjv_defaultPerturb(self.perturb)

        self._fontscale = self._font_scale()
        self._model = None
        self._data = None
        self._scene = None
        self._context = None
        self._renderers = {}
        self._next_frame = 0.0
        # Scene flags get their own array: the panel's check boxes hold a pointer
        # to it, and a scene does not survive a change of model.
        self._render_flags = np.zeros(mujoco.mjtRndFlag.mjNRNDFLAG, dtype=np.uint8)
        self.show(model, data)
        self._render_flags[:] = self._scene.flags

        self._tasks = tuple(tasks)
        self._chosen = ctypes.c_int(0)
        self._running = ctypes.c_int(1)
        self._clicked_task = None
        self._reset_requested = False
        self._panel_shown = True

        self._state = mjui.State()
        self._ui = mjui.UI()
        self._ui_ptr = ctypes.byref(self._ui)
        self._state_ptr = ctypes.byref(self._state)
        # Fonts are all the panels need from a context, and these come from no
        # model in particular, so this one stays valid across `show`.
        self._empty_model = mujoco.MjModel.from_xml_string("<mujoco/>")
        self._widgets = mjui.make_context(self._lib, self._empty_model, self._fontscale)
        self._build_panel()

        glfw.set_cursor_pos_callback(self._window, self._on_cursor)
        glfw.set_mouse_button_callback(self._window, self._on_button)
        glfw.set_scroll_callback(self._window, self._on_scroll)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_framebuffer_size_callback(self._window, self._on_resize)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()

    def close(self) -> None:
        """Releases the render contexts and the window."""
        if self._window is None:
            return
        for _model, _scene, context in self._renderers.values():
            context.free()
        self._lib.mjr_freeContext(self._widgets)
        glfw.destroy_window(self._window)
        glfw.terminate()
        self._window = None

    def is_running(self) -> bool:
        """Whether the window is still open."""
        return not glfw.window_should_close(self._window)

    @property
    def paused(self) -> bool:
        """Whether the panel's Simulation section is set to Pause."""
        return not self._running.value

    def take_clicked_task(self) -> str | None:
        """The task clicked since the last call, even if already the chosen one."""
        clicked, self._clicked_task = self._clicked_task, None
        return clicked

    def take_reset_request(self) -> bool:
        """Whether a reset was asked for since the last call."""
        requested, self._reset_requested = self._reset_requested, False
        return requested

    def resume(self) -> None:
        """Sets the Simulation section to Run."""
        self._running.value = 1
        self._refresh_panel()

    def prepare(self, model: mujoco.MjModel) -> None:
        """Builds a model's scene and render context ahead of showing it."""
        # 40 ms on a mesh-heavy model, nearly all of it uploading meshes and
        # textures. That is felt on every click if left to `show`, so a replay
        # pays it once per task before the first frame.
        if id(model) not in self._renderers:
            self._renderers[id(model)] = (
                model,
                mujoco.MjvScene(model, _MAX_GEOM),
                mujoco.MjrContext(model, self._fontscale),
            )

    def show(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Points the window at another model and its buffer; the camera stays."""
        self.prepare(model)
        self._model = model
        self._data = data
        _, self._scene, self._context = self._renderers[id(model)]
        self._next_frame = 0.0
        self.perturb.active = 0
        self.perturb.select = 0
        # Tracking survives when the body exists on the new model. When it does
        # not, mjv_updateCamera aborts the whole process.
        if self.cam.trackbodyid >= model.nbody or self.cam.fixedcamid >= model.ncam:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE

    def sync(self) -> None:
        """Handles what the mouse and keyboard did, and draws if a frame is due."""
        glfw.poll_events()
        now = glfw.get_time()
        if now >= self._next_frame:
            self._next_frame = now + _SECONDS_PER_FRAME
            self.draw()

    def draw(self) -> None:
        """Renders the scene and the panel from the data as it stands."""
        scene_rect = self._state.rect[_RECT_SCENE]
        self._scene.flags[:] = self._render_flags
        mujoco.mjv_updateScene(
            self._model, self._data, self.opt, self.perturb, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self._scene,
        )
        mujoco.mjr_render(
            mujoco.MjrRect(
                scene_rect.left, scene_rect.bottom, scene_rect.width, scene_rect.height,
            ),
            self._scene,
            self._context,
        )
        if self._panel_shown:
            self._lib.mjui_render(self._ui_ptr, self._state_ptr, self._widgets)
        glfw.swap_buffers(self._window)

    def _font_scale(self) -> int:
        """Estimates a font scale for the display, as `simulate` does."""
        buffer_width, _ = glfw.get_framebuffer_size(self._window)
        window_width, _ = glfw.get_window_size(self._window)
        ratio = buffer_width / window_width
        monitor = glfw.get_primary_monitor()
        width_mm, _ = glfw.get_monitor_physical_size(monitor)
        pixels = glfw.get_video_mode(monitor).size.width
        ppi = ratio * 25.4 * pixels / width_mm if width_mm > 0 else 0.0
        if buffer_width > window_width:
            scale = ratio * 100
        elif 50 < ppi < 350:
            scale = ppi
        else:
            scale = _FALLBACK_FONT_SCALE
        return min(300, max(100, 50 * round(scale * 0.02)))

    def _build_panel(self) -> None:
        ui = self._ui
        ui.spacing = self._lib.mjui_themeSpacing(_THEME)
        ui.color = self._lib.mjui_themeColor(_THEME)
        ui.rectid = _RECT_PANEL
        ui.auxid = _AUX_BUFFER
        ui.radiocol = _RADIO_COLUMNS

        rows = []
        if len(self._tasks) > 1:
            rows += [
                (mjui.ITEM_SECTION, "Tasks", mjui.SECT_OPEN, None, "AT"),
                (
                    mjui.ITEM_RADIO, "", 1, ctypes.addressof(self._chosen),
                    "\n".join(self._tasks),
                ),
            ]
        rows += [
            (mjui.ITEM_SECTION, "Simulation", mjui.SECT_OPEN, None, "AS"),
            (
                mjui.ITEM_RADIO, "", 1, ctypes.addressof(self._running), "Pause\nRun",
            ),
            (mjui.ITEM_BUTTON, "Reset", 1, None, ""),
            (mjui.ITEM_SECTION, "Rendering", 0, None, "AR"),
            *_flag_rows(self._render_flags, mujoco.mjRNDSTRING),
            (mjui.ITEM_SECTION, "Visualization", 0, None, "AV"),
            *_flag_rows(self.opt.flags, mujoco.mjVISSTRING),
        ]
        mjui.add_table(self._lib, ui, rows)
        self._lib.mjui_resize(self._ui_ptr, self._widgets)
        self._lib.mjr_addAux(
            ui.auxid, ui.width, ui.maxheight, ui.spacing.samples, self._widgets,
        )
        self._layout()

    def _layout(self) -> None:
        state = self._state
        width, height = glfw.get_framebuffer_size(self._window)
        window_width, _ = glfw.get_window_size(self._window)
        self._buffer_per_window = width / max(window_width, 1)
        panel = self._ui.width if self._panel_shown else 0
        state.nrect = 3
        state.rect[_RECT_WINDOW] = mjui.Rect(0, 0, width, height)
        state.rect[_RECT_PANEL] = mjui.Rect(0, 0, panel, height)
        state.rect[_RECT_SCENE] = mjui.Rect(panel, 0, max(width - panel, 0), height)
        self._refresh_panel()

    def _refresh_panel(self) -> None:
        self._lib.mjui_update(-1, -1, self._ui_ptr, self._state_ptr, self._widgets)

    def _rect_under_mouse(self) -> int:
        state = self._state
        for index in (_RECT_PANEL, _RECT_SCENE):
            rect = state.rect[index]
            if (
                rect.left <= state.x < rect.left + rect.width
                and rect.bottom <= state.y < rect.bottom + rect.height
            ):
                return index
        return _RECT_WINDOW

    def _read_modifiers(self, mods: int | None = None) -> None:
        state = self._state
        if mods is not None:
            state.control = bool(mods & glfw.MOD_CONTROL)
            state.shift = bool(mods & glfw.MOD_SHIFT)
            state.alt = bool(mods & glfw.MOD_ALT)
            return

        def down(*keys):
            return any(glfw.get_key(self._window, key) == glfw.PRESS for key in keys)

        state.control = down(glfw.KEY_LEFT_CONTROL, glfw.KEY_RIGHT_CONTROL)
        state.shift = down(glfw.KEY_LEFT_SHIFT, glfw.KEY_RIGHT_SHIFT)
        state.alt = down(glfw.KEY_LEFT_ALT, glfw.KEY_RIGHT_ALT)

    def _on_resize(self, window, width, height) -> None:
        self._layout()

    def _on_cursor(self, window, x, y) -> None:
        state = self._state
        # The widgets count from the framebuffer's bottom-left corner.
        x *= self._buffer_per_window
        y = state.rect[_RECT_WINDOW].height - y * self._buffer_per_window
        state.dx, state.dy = x - state.x, y - state.y
        state.x, state.y = x, y
        state.type = mjui.EVENT_MOVE
        state.mouserect = self._rect_under_mouse()
        self._read_modifiers()
        self._dispatch()

    def _on_button(self, window, button, action, mods) -> None:
        index = _MOUSE_BUTTONS.get(button)
        if index is None:
            return
        state = self._state
        pressed = action == glfw.PRESS
        if index == mjui.BUTTON_LEFT:
            state.left = pressed
        elif index == mjui.BUTTON_RIGHT:
            state.right = pressed
        else:
            state.middle = pressed
        self._read_modifiers(mods)
        state.mouserect = self._rect_under_mouse()
        if pressed:
            now = glfw.get_time()
            state.doubleclick = (
                state.button == index
                and now - state.buttontime < _DOUBLE_CLICK_SECONDS
            )
            state.buttontime = now
            state.button = index
            state.dragbutton = index
            state.dragrect = state.mouserect
            state.type = mjui.EVENT_PRESS
        else:
            state.type = mjui.EVENT_RELEASE
        self._dispatch()
        if not pressed:
            state.dragrect = 0
            state.dragbutton = mjui.BUTTON_NONE

    def _on_scroll(self, window, x_offset, y_offset) -> None:
        state = self._state
        state.sx, state.sy = x_offset, y_offset
        state.type = mjui.EVENT_SCROLL
        state.mouserect = self._rect_under_mouse()
        self._read_modifiers()
        self._dispatch()

    def _on_key(self, window, key, scancode, action, mods) -> None:
        if action == glfw.RELEASE:
            return
        state = self._state
        state.key = key
        state.keytime = glfw.get_time()
        state.type = mjui.EVENT_KEY
        self._read_modifiers(mods)
        self._dispatch()

    def _dispatch(self) -> None:
        state = self._state
        over_panel = state.dragrect == _RECT_PANEL or (
            state.dragrect == 0 and state.mouserect == _RECT_PANEL
        )
        if self._panel_shown and (over_panel or state.type == mjui.EVENT_KEY):
            item = self._lib.mjui_event(self._ui_ptr, self._state_ptr, self._widgets)
            if item:
                self._on_item(item.contents)
                return
            if state.type == mjui.EVENT_KEY and state.key == 0:
                return
        if state.type == mjui.EVENT_KEY:
            self._on_shortcut(state.key)
        elif state.mouserect == _RECT_SCENE or state.dragrect == _RECT_SCENE:
            self._on_scene_mouse()

    def _on_item(self, item: mjui.Item) -> None:
        if item.pdata == ctypes.addressof(self._chosen):
            self._clicked_task = self._tasks[self._chosen.value]
        elif item.pdata == ctypes.addressof(self._running):
            print("[viewer] paused" if self.paused else "[viewer] running", flush=True)
        elif item.type == mjui.ITEM_BUTTON:
            self._reset_requested = True

    def _on_shortcut(self, key: int) -> None:
        if key == glfw.KEY_SPACE:
            self._running.value = int(self.paused)
            self._refresh_panel()
            print("[viewer] paused" if self.paused else "[viewer] running", flush=True)
        elif key in (glfw.KEY_DELETE, glfw.KEY_BACKSPACE):
            self._reset_requested = True
        elif key == glfw.KEY_ESCAPE:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        elif key == glfw.KEY_TAB:
            self._panel_shown = not self._panel_shown
            self._layout()

    def _on_scene_mouse(self) -> None:
        state = self._state
        rect = state.rect[_RECT_SCENE]
        if state.type == mjui.EVENT_SCROLL and state.mouserect == _RECT_SCENE:
            mujoco.mjv_moveCamera(
                self._model, mujoco.mjtMouse.mjMOUSE_ZOOM,
                0.0, -_ZOOM_PER_SCROLL * state.sy, self.cam,
            )
        elif state.type == mjui.EVENT_PRESS and state.mouserect == _RECT_SCENE:
            self._on_scene_press(rect)
        elif state.type == mjui.EVENT_RELEASE and state.dragrect == _RECT_SCENE:
            self.perturb.active = 0
        elif state.type == mjui.EVENT_MOVE and state.dragrect == _RECT_SCENE:
            self._on_scene_drag(rect)

    def _on_scene_press(self, rect: mjui.Rect) -> None:
        state = self._state
        pert = self.perturb
        active = 0
        if state.control and pert.select > 0:
            if state.right:
                active = mujoco.mjtPertBit.mjPERT_TRANSLATE
            elif state.left:
                active = mujoco.mjtPertBit.mjPERT_ROTATE
            if active and not pert.active:
                mujoco.mjv_initPerturb(self._model, self._data, self._scene, pert)
        pert.active = active
        if state.doubleclick:
            self._on_scene_double_click(rect)
            pert.active = 0

    def _on_scene_double_click(self, rect: mjui.Rect) -> None:
        state = self._state
        pert = self.perturb
        point = np.zeros(3, dtype=np.float64)
        geom = np.zeros(1, dtype=np.int32)
        flex = np.zeros(1, dtype=np.int32)
        skin = np.zeros(1, dtype=np.int32)
        body = mujoco.mjv_select(
            self._model, self._data, self.opt, rect.width / rect.height,
            (state.x - rect.left) / rect.width, (state.y - rect.bottom) / rect.height,
            self._scene, point, geom, flex, skin,
        )
        if state.button == mjui.BUTTON_LEFT:
            pert.select = max(body, 0)
            pert.flexselect = int(flex[0]) if body >= 0 else -1
            pert.skinselect = int(skin[0]) if body >= 0 else -1
            if body >= 0:
                frame = self._data.xmat[body].reshape(3, 3)
                pert.localpos = frame.T @ (point - self._data.xpos[body])
            return
        if body >= 0:
            self.cam.lookat = point
        if state.control and body > 0:
            self.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self.cam.trackbodyid = body
            self.cam.fixedcamid = -1

    def _on_scene_drag(self, rect: mjui.Rect) -> None:
        state = self._state
        if state.right:
            action = (
                mujoco.mjtMouse.mjMOUSE_MOVE_H if state.shift
                else mujoco.mjtMouse.mjMOUSE_MOVE_V
            )
        elif state.left:
            action = (
                mujoco.mjtMouse.mjMOUSE_ROTATE_H if state.shift
                else mujoco.mjtMouse.mjMOUSE_ROTATE_V
            )
        else:
            action = mujoco.mjtMouse.mjMOUSE_ZOOM
        # `state.y` counts upward; the camera and the push expect screen direction.
        dx, dy = state.dx / rect.height, -state.dy / rect.height
        if self.perturb.active:
            mujoco.mjv_movePerturb(
                self._model, self._data, action, dx, dy, self._scene, self.perturb,
            )
        else:
            mujoco.mjv_moveCamera(self._model, action, dx, dy, self.cam)
