"""Extend Hydra's config search path to the repo's out-of-package config dirs.

The launchable experiment configs live in top-level ``experiments/`` (grouped by
env into ``dmc/`` and so on) — that tree isn't inside the installed ``roxie``
package, so Hydra can't see it by default. The primary ``config_path`` stays at
``roxie/configs`` (the shared ``agent`` / ``noise`` groups); this plugin appends
``experiments/`` so configs there compose against those groups (e.g.
``--config-name dmc/bench_td3``).

TWO directories are searched, because a task can live outside this repo. The
first is derived from this file's location, so roxie's own experiments are found
whatever the working directory. The second is ``./experiments`` — that is what
lets a downstream repo (roxie-mocap, say) keep its launchables next to its env
and still run them through ``roxie.train``. When the two are the same directory,
as they are for a run started from a roxie checkout, only one entry is added.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

# roxie/utils/hydra_searchpath.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]


def _search_dirs() -> dict[str, Path]:
    """The experiment dirs to append, most specific last.

    Resolved on every call rather than at import: this is cheap, and a test (or
    a caller that chdirs before composing) then sees the directory it is
    actually sitting in.
    """
    dirs = {"roxie-experiments": REPO_ROOT / "experiments"}
    cwd = Path.cwd().resolve() / "experiments"
    if cwd != dirs["roxie-experiments"] and cwd.is_dir():
        dirs["roxie-local-experiments"] = cwd
    return dirs


class RoxieSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for provider, path in _search_dirs().items():
            search_path.append(provider=provider, path=f"file://{path}")


def register() -> None:
    """Register the search-path plugin. Idempotent; call before Hydra composes."""
    Plugins.instance().register(RoxieSearchPathPlugin)
