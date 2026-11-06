"""The memory guard kills a run below its reserve and leaves others alone."""

import pathlib
import signal
import subprocess
import sys
import time

import pytest

GUARD = pathlib.Path(__file__).parent.parent / "roxie" / "utils" / "memory_guard.py"

pytestmark = pytest.mark.skipif(
    not pathlib.Path("/proc/meminfo").exists(), reason="needs /proc/meminfo",
)

_LOAD_GUARD = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("memory_guard", {str(GUARD)!r})
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)
"""
_UNREACHABLE_RESERVE_GB = 1e9
_TIMEOUT_SECONDS = 30


def _run(body: str, *args, **kwargs):
    return subprocess.run(
        [sys.executable, "-c", _LOAD_GUARD + body, *args],
        capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, **kwargs,
    )


def test_kills_a_run_below_its_reserve_and_says_why():
    out = _run(f"guard.start({_UNREACHABLE_RESERVE_GB})\nimport time; time.sleep(60)")
    assert out.returncode == -signal.SIGKILL
    assert "MEMORY GUARD" in out.stderr
    assert "env.parallel_envs" in out.stderr


def test_kills_a_run_that_is_holding_the_gil():
    # ctypes.PyDLL keeps the GIL across the call, as native code can.
    out = _run(
        f"guard.start({_UNREACHABLE_RESERVE_GB})\n"
        "import ctypes; ctypes.PyDLL(None).sleep(60)"
    )
    assert out.returncode == -signal.SIGKILL


def test_leaves_a_run_above_its_reserve_alone():
    out = _run("guard.start(0)\nimport time; time.sleep(0.5)\nprint('finished')")
    assert out.returncode == 0
    assert "finished" in out.stdout
    assert "MEMORY GUARD" not in out.stderr


def test_watcher_exits_with_the_run():
    out = _run("guard.start(0)\nprint(guard._watcher.pid)\nimport time; time.sleep(0.3)")
    watcher = pathlib.Path(f"/proc/{int(out.stdout.split()[-1])}")
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while watcher.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not watcher.exists()


def test_start_is_idempotent():
    out = _run(
        "guard.start(0)\nfirst = guard._watcher\nguard.start(0)\n"
        "print(first is guard._watcher)"
    )
    assert out.stdout.split()[-1] == "True"
