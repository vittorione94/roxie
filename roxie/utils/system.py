"""What a run asks of the machine: its CPU cores, and a guard on its memory.

`pin_cpu_cores` narrows the process to a subset of its cores, packed into as few
L3 domains as it can: a learning pass slows down once the mask spans two.

`start_memory_guard` kills the run before its memory use can take the machine
down. A watcher process polls the machine's `MemAvailable` and SIGKILLs the run
when it drops below a reserve. It is a process rather than a thread because
native code can hold the GIL for as long as a large allocation takes. `RLIMIT_AS`
would not do either: a run's address space is far larger than its resident
memory, so the limit refuses runs that fit. It covers host RAM only, not GPU
memory.

The watcher runs this file as a script under `-I -S`, so it imports only the
standard library.
"""

import os
import signal
import subprocess
import sys
import time
from typing import Any

_MEMINFO = "/proc/meminfo"
_POLL_SECONDS = 0.1
_DEFAULT_RESERVE_FRACTION = 0.1

_watcher = None


def _parse_cpu_list(text: str) -> list[int]:
    """Parses a CPU range string into individual core indices."""
    out: list[int] = []
    for part in text.strip().split(","):
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            out.extend(range(int(lo), int(hi) + 1))
        else:
            out.append(int(part))
    return out


def _l3_domains(available: list[int]) -> list[list[int]]:
    """Groups available CPUs by their shared L3 cache domain."""
    groups: dict[tuple[int, ...], list[int]] = {}
    for cpu in available:
        path = f"/sys/devices/system/cpu/cpu{cpu}/cache/index3/shared_cpu_list"
        try:
            with open(path) as fh:
                key = tuple(sorted(_parse_cpu_list(fh.read())))
        except (OSError, ValueError):
            return [list(available)]
        groups.setdefault(key, []).append(cpu)
    return sorted(groups.values(), key=lambda g: (-len(g), g[0]))


def select_cpu_cores(available: list[int], cores: int) -> list[int]:
    """Selects core indices packed to minimize L3 cache domain boundaries."""
    picked: list[int] = []
    for group in _l3_domains(available):
        if len(picked) >= cores:
            break
        picked.extend(group[: cores - len(picked)])
    return sorted(picked)


def pin_cpu_cores(cores: Any, platform: str | None) -> int | None:
    """Pins CPU affinity for the current process to a subset of available cores."""
    if not cores or str(platform) != "cpu" or not hasattr(os, "sched_setaffinity"):
        return None
    available = sorted(os.sched_getaffinity(0))
    cores = int(cores)
    if cores >= len(available):
        return None
    os.sched_setaffinity(0, set(select_cpu_cores(available, cores)))
    return cores


def _meminfo_bytes(field: str) -> int:
    with open(_MEMINFO) as fh:
        for line in fh:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def _rss_bytes(pid: int) -> int:
    with open(f"/proc/{pid}/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def start_memory_guard(reserve_gb: float | None = None) -> None:
    """Starts the watcher over this process, if none is running yet.

    Args:
        reserve_gb: the available memory to leave the rest of the machine; 10%
          of its RAM when None.
    """
    global _watcher
    if _watcher is not None:
        return
    if not os.path.exists(_MEMINFO):
        print(f"Memory guard: OFF, this platform has no {_MEMINFO}.", flush=True)
        return
    reserve = (_meminfo_bytes("MemTotal") * _DEFAULT_RESERVE_FRACTION
               if reserve_gb is None else float(reserve_gb) * 1e9)
    # -I keeps roxie/utils/ off the watcher's sys.path, where math.py would
    # shadow the stdlib's. A session of its own keeps Ctrl-C away from it.
    _watcher = subprocess.Popen(
        [sys.executable, "-I", "-S", os.path.abspath(__file__),
         str(os.getpid()), str(int(reserve))],
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        start_new_session=True,
    )
    print(f"Memory guard: stops the run below {reserve / 1e9:.1f} GB available.",
          flush=True)


def _watch(pid: int, reserve: int) -> None:
    while os.getppid() == pid:
        available = _meminfo_bytes("MemAvailable")
        if available < reserve:
            print(
                f"\nMEMORY GUARD: killing the run. The machine has "
                f"{available / 1e9:.1f} GB available, below the "
                f"{reserve / 1e9:.1f} GB reserve; the run holds "
                f"{_rss_bytes(pid) / 1e9:.1f} GB. Lower env.parallel_envs or "
                f"the replay's max_length, or set +runtime.memory_reserve_gb=.",
                file=sys.stderr, flush=True,
            )
            os.kill(pid, signal.SIGKILL)
            return
        time.sleep(_POLL_SECONDS)


if __name__ == "__main__":
    _watch(int(sys.argv[1]), int(sys.argv[2]))
