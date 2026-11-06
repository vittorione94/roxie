"""Kills the run before its memory use can take the machine down.

A watcher process polls the machine's `MemAvailable` and SIGKILLs the run when
it drops below a reserve. It is a process rather than a thread because native
code can hold the GIL for as long as a large allocation takes. `RLIMIT_AS` would
not do either: a run's address space is far larger than its resident memory, so
the limit refuses runs that fit.

Covers host RAM only, not GPU memory.
"""

import os
import signal
import subprocess
import sys
import time

_MEMINFO = "/proc/meminfo"
_POLL_SECONDS = 0.1
_DEFAULT_RESERVE_FRACTION = 0.1

_watcher = None


def _meminfo_bytes(field: str) -> int:
    with open(_MEMINFO) as fh:
        for line in fh:
            if line.startswith(f"{field}:"):
                return int(line.split()[1]) * 1024
    raise KeyError(field)


def _rss_bytes(pid: int) -> int:
    with open(f"/proc/{pid}/statm") as fh:
        return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def start(reserve_gb: float | None = None) -> None:
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
