"""Which cores a run gets, and the guard that kills it below its memory reserve."""

import pathlib
import signal
import subprocess
import sys
import time

import pytest

from roxie.utils.system import _l3_domains, _parse_cpu_list, select_cpu_cores

SYSTEM = pathlib.Path(__file__).parent.parent / "roxie" / "utils" / "system.py"

needs_meminfo = pytest.mark.skipif(
    not pathlib.Path("/proc/meminfo").exists(), reason="needs /proc/meminfo",
)

_LOAD_SYSTEM = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("system", {str(SYSTEM)!r})
system = importlib.util.module_from_spec(spec)
spec.loader.exec_module(system)
"""
_UNREACHABLE_RESERVE_GB = 1e9
_TIMEOUT_SECONDS = 30


def _run(body: str, *args, **kwargs):
    return subprocess.run(
        [sys.executable, "-c", _LOAD_SYSTEM + body, *args],
        capture_output=True, text=True, timeout=_TIMEOUT_SECONDS, **kwargs,
    )


# ---- CPU core selection -----------------------------------------------------
#
# Which cpus the run gets, not just how many: a learning pass slows down once
# the mask spans two L3 domains, so it has to stay inside one while it fits.


def test_cpu_list_parsing_handles_ranges_and_singletons():
    assert _parse_cpu_list("0-5,12-17") == [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    assert _parse_cpu_list("3") == [3]
    assert _parse_cpu_list("") == []


def test_core_selection_packs_into_one_l3_domain(monkeypatch):
    """A request that fits in one domain must not straddle two."""
    ccd0 = [0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17]
    ccd1 = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
    monkeypatch.setattr(
        "roxie.utils.system._l3_domains", lambda available: [ccd0, ccd1]
    )
    # The naive `available[:cores]` would return 0..11 here -- six cpus from
    # each CCD, which is the case this exists to avoid.
    assert select_cpu_cores(sorted(ccd0 + ccd1), 12) == sorted(ccd0)
    assert set(select_cpu_cores(sorted(ccd0 + ccd1), 6)) <= set(ccd0)


def test_core_selection_spills_to_a_second_domain_only_when_it_must():
    picked = select_cpu_cores(list(range(24)), 24)
    assert picked == list(range(24)), "everything available must still be usable"


def test_core_selection_stays_within_the_cpus_it_was_given():
    """An outer `taskset` chooses which cores a run gets; this only narrows."""
    available = [6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23]
    for n in (1, 6, 12):
        assert set(select_cpu_cores(available, n)) <= set(available)
        assert len(select_cpu_cores(available, n)) == n


def test_core_selection_falls_back_when_topology_is_unreadable(monkeypatch):
    """No sysfs (container, non-Linux) must degrade to the flat behaviour, not
    raise."""
    def boom(*args, **kwargs):
        raise OSError("no sysfs here")

    monkeypatch.setattr("builtins.open", boom)
    assert _l3_domains([0, 1, 2, 3]) == [[0, 1, 2, 3]]
    assert select_cpu_cores([0, 1, 2, 3], 2) == [0, 1]


# ---- memory guard -----------------------------------------------------------


@needs_meminfo
def test_kills_a_run_below_its_reserve_and_says_why():
    out = _run(
        f"system.start_memory_guard({_UNREACHABLE_RESERVE_GB})\n"
        "import time; time.sleep(60)"
    )
    assert out.returncode == -signal.SIGKILL
    assert "MEMORY GUARD" in out.stderr
    assert "env.parallel_envs" in out.stderr


@needs_meminfo
def test_kills_a_run_that_is_holding_the_gil():
    # ctypes.PyDLL keeps the GIL across the call, as native code can.
    out = _run(
        f"system.start_memory_guard({_UNREACHABLE_RESERVE_GB})\n"
        "import ctypes; ctypes.PyDLL(None).sleep(60)"
    )
    assert out.returncode == -signal.SIGKILL


@needs_meminfo
def test_leaves_a_run_above_its_reserve_alone():
    out = _run(
        "system.start_memory_guard(0)\nimport time; time.sleep(0.5)\nprint('finished')"
    )
    assert out.returncode == 0
    assert "finished" in out.stdout
    assert "MEMORY GUARD" not in out.stderr


@needs_meminfo
def test_watcher_exits_with_the_run():
    out = _run(
        "system.start_memory_guard(0)\nprint(system._watcher.pid)\n"
        "import time; time.sleep(0.3)"
    )
    watcher = pathlib.Path(f"/proc/{int(out.stdout.split()[-1])}")
    deadline = time.monotonic() + _TIMEOUT_SECONDS
    while watcher.exists() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not watcher.exists()


@needs_meminfo
def test_start_is_idempotent():
    out = _run(
        "system.start_memory_guard(0)\nfirst = system._watcher\n"
        "system.start_memory_guard(0)\nprint(first is system._watcher)"
    )
    assert out.stdout.split()[-1] == "True"
