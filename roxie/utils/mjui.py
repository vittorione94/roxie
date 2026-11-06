"""`ctypes` bindings for MuJoCo's widget toolkit, which its Python package omits.

`mjui_*` is exported by `libmujoco` and its structs are public in `mjui.h`, but
no binding reaches them. A wrong struct layout here does not raise, it corrupts
memory, so the layout is pinned to the header it was written from: `load` refuses
any MuJoCo whose `mjui.h` differs.
"""

import ctypes
import hashlib
from ctypes import POINTER, c_char, c_double, c_float, c_int, c_void_p
from pathlib import Path

import mujoco

_MJUI_H_SHA256 = "797532f5e00744561ba08c938fe30deba6f7c595f7c6e9834e6bb4f74a77fedc"

MAX_NAME = 40
MAX_MULTI = 35
_MAX_UISECT = 10
_MAX_UIITEM = 200
_MAX_UITEXT = 300
_MAX_UIEDIT = 7
_MAX_UIRECT = 25

# `mjrContext` is only ever handed back to the library, never read here, so it
# needs no layout. It only needs room, and in 3.9 it takes well under a tenth of
# this.
_CONTEXT_BYTES = 1 << 20

ITEM_END = -2
ITEM_SECTION = -1
ITEM_BUTTON = 2
ITEM_CHECKBYTE = 4
ITEM_RADIO = 5

SECT_OPEN = 1

EVENT_MOVE = 1
EVENT_PRESS = 2
EVENT_RELEASE = 3
EVENT_SCROLL = 4
EVENT_KEY = 5

BUTTON_NONE = 0
BUTTON_LEFT = 1
BUTTON_RIGHT = 2
BUTTON_MIDDLE = 3


class Rect(ctypes.Structure):
    """`mjrRect`."""

    _fields_ = [
        ("left", c_int), ("bottom", c_int), ("width", c_int), ("height", c_int),
    ]


class State(ctypes.Structure):
    """`mjuiState`, the mouse and keyboard as the widgets see them."""

    _fields_ = [
        ("nrect", c_int),
        ("rect", Rect * _MAX_UIRECT),
        ("userdata", c_void_p),
        ("type", c_int),
        ("left", c_int),
        ("right", c_int),
        ("middle", c_int),
        ("doubleclick", c_int),
        ("button", c_int),
        ("buttontime", c_double),
        ("x", c_double),
        ("y", c_double),
        ("dx", c_double),
        ("dy", c_double),
        ("sx", c_double),
        ("sy", c_double),
        ("control", c_int),
        ("shift", c_int),
        ("alt", c_int),
        ("key", c_int),
        ("keytime", c_double),
        ("mouserect", c_int),
        ("dragrect", c_int),
        ("dragbutton", c_int),
        ("dropcount", c_int),
        ("droppaths", c_void_p),
    ]


class ThemeSpacing(ctypes.Structure):
    """`mjuiThemeSpacing`."""

    _fields_ = [
        (name, c_int)
        for name in (
            "total", "scroll", "label", "section", "cornersect", "cornersep",
            "itemside", "itemmid", "itemver", "texthor", "textver",
            "linescroll", "samples",
        )
    ]


class ThemeColor(ctypes.Structure):
    """`mjuiThemeColor`."""

    _fields_ = [
        (name, c_float * 3)
        for name in (
            "master", "thumb", "secttitle", "secttitle2", "secttitleuncheck",
            "secttitleuncheck2", "secttitlecheck", "secttitlecheck2", "sectfont",
            "sectsymbol", "sectpane", "separator", "separator2", "shortcut",
            "fontactive", "fontinactive", "decorinactive", "decorinactive2",
            "button", "check", "radio", "select", "select2", "slider", "slider2",
            "edit", "edit2", "cursor",
        )
    ]


class _ItemSingle(ctypes.Structure):
    _fields_ = [("modifier", c_int), ("shortcut", c_int)]


class _ItemMulti(ctypes.Structure):
    _fields_ = [("nelem", c_int), ("name", (c_char * MAX_NAME) * MAX_MULTI)]


class _ItemSlider(ctypes.Structure):
    _fields_ = [("range", c_double * 2), ("divisions", c_double)]


class _ItemEdit(ctypes.Structure):
    _fields_ = [("nelem", c_int), ("range", (c_double * 2) * _MAX_UIEDIT)]


class _ItemData(ctypes.Union):
    _fields_ = [
        ("single", _ItemSingle), ("multi", _ItemMulti),
        ("slider", _ItemSlider), ("edit", _ItemEdit),
    ]


class Item(ctypes.Structure):
    """`mjuiItem`, one widget."""

    _anonymous_ = ("data",)
    _fields_ = [
        ("type", c_int),
        ("name", c_char * MAX_NAME),
        ("state", c_int),
        ("pdata", c_void_p),
        ("sectionid", c_int),
        ("itemid", c_int),
        ("userid", c_int),
        ("data", _ItemData),
        ("rect", Rect),
        ("skip", c_int),
    ]


class Section(ctypes.Structure):
    """`mjuiSection`."""

    _fields_ = [
        ("name", c_char * MAX_NAME),
        ("state", c_int),
        ("modifier", c_int),
        ("shortcut", c_int),
        ("checkbox", c_int),
        ("nitem", c_int),
        ("item", Item * _MAX_UIITEM),
        ("rtitle", Rect),
        ("rcontent", Rect),
        ("lastclick", c_int),
    ]


class UI(ctypes.Structure):
    """`mjUI`, one panel of widgets."""

    _fields_ = [
        ("spacing", ThemeSpacing),
        ("color", ThemeColor),
        ("predicate", c_void_p),
        ("userdata", c_void_p),
        ("rectid", c_int),
        ("auxid", c_int),
        ("radiocol", c_int),
        ("width", c_int),
        ("height", c_int),
        ("maxheight", c_int),
        ("scroll", c_int),
        ("mousesect", c_int),
        ("mouseitem", c_int),
        ("mousehelp", c_int),
        ("mouseclicks", c_int),
        ("mousesectcheck", c_int),
        ("editsect", c_int),
        ("edititem", c_int),
        ("editcursor", c_int),
        ("editscroll", c_int),
        ("edittext", c_char * _MAX_UITEXT),
        ("editchanged", POINTER(Item)),
        ("nsect", c_int),
        ("sect", Section * _MAX_UISECT),
    ]


class Def(ctypes.Structure):
    """`mjuiDef`, one row of the table `mjui_add` reads."""

    _fields_ = [
        ("type", c_int),
        ("name", c_char * MAX_NAME),
        ("state", c_int),
        ("pdata", c_void_p),
        ("other", c_char * _MAX_UITEXT),
        ("otherint", c_int),
    ]


def _library_path() -> Path:
    package = Path(mujoco.__file__).parent
    return next(path for path in package.rglob("libmujoco*") if path.is_file())


def _check_layout() -> None:
    header = Path(mujoco.__file__).parent / "include" / "mujoco" / "mjui.h"
    found = hashlib.sha256(header.read_bytes()).hexdigest()
    if found != _MJUI_H_SHA256:
        raise RuntimeError(
            f"MuJoCo {mujoco.__version__} ships a different mjui.h than "
            f"roxie/utils/mjui.py was written from. Compare the structs against "
            f"it, then update _MJUI_H_SHA256 to {found}."
        )


def load() -> ctypes.CDLL:
    """Returns `libmujoco` with the signatures of the calls the viewer makes.

    Raises:
        RuntimeError: if this MuJoCo's `mjui.h` is not the one written against.
    """
    _check_layout()
    lib = ctypes.CDLL(str(_library_path()))
    lib.mjui_themeSpacing.restype = ThemeSpacing
    lib.mjui_themeSpacing.argtypes = [c_int]
    lib.mjui_themeColor.restype = ThemeColor
    lib.mjui_themeColor.argtypes = [c_int]
    lib.mjui_add.restype = None
    lib.mjui_add.argtypes = [POINTER(UI), POINTER(Def)]
    lib.mjui_resize.restype = None
    lib.mjui_resize.argtypes = [POINTER(UI), c_void_p]
    lib.mjui_update.restype = None
    lib.mjui_update.argtypes = [
        c_int, c_int, POINTER(UI), POINTER(State), c_void_p,
    ]
    lib.mjui_event.restype = POINTER(Item)
    lib.mjui_event.argtypes = [POINTER(UI), POINTER(State), c_void_p]
    lib.mjui_render.restype = None
    lib.mjui_render.argtypes = [POINTER(UI), POINTER(State), c_void_p]
    lib.mjr_defaultContext.restype = None
    lib.mjr_defaultContext.argtypes = [c_void_p]
    lib.mjr_makeContext.restype = None
    lib.mjr_makeContext.argtypes = [c_void_p, c_void_p, c_int]
    lib.mjr_freeContext.restype = None
    lib.mjr_freeContext.argtypes = [c_void_p]
    lib.mjr_addAux.restype = None
    lib.mjr_addAux.argtypes = [c_int, c_int, c_int, c_int, c_void_p]
    return lib


def make_context(lib: ctypes.CDLL, model: mujoco.MjModel, fontscale: int):
    """Builds an `mjrContext` the widgets draw with, on the current GL context."""
    context = ctypes.create_string_buffer(_CONTEXT_BYTES)
    lib.mjr_defaultContext(context)
    lib.mjr_makeContext(model._address, context, fontscale)
    return context


def add_table(lib: ctypes.CDLL, ui: UI, rows: list[tuple]) -> None:
    """Adds `(type, name, state, pdata, other)` rows to a panel."""
    table = (Def * (len(rows) + 1))()
    for row, (kind, name, state, pdata, other) in zip(table, rows):
        row.type = kind
        row.name = name.encode()
        row.state = state
        row.pdata = pdata
        row.other = other.encode()
    table[len(rows)].type = ITEM_END
    lib.mjui_add(ctypes.byref(ui), table)
