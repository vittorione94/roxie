"""Extend Hydra's config search path to the repo's out-of-package config dirs.

Launchable experiment configs live in top-level ``experiments/``, outside the
installed ``roxie`` package and so invisible to Hydra by default. The primary
``config_path`` stays at ``roxie/configs`` (the shared ``agent`` / ``noise``
groups); this plugin appends ``experiments/`` so configs there compose against
them (``--config-name dmc/bench_td3``).

Two directories are searched, because a task can live outside this repo: one
derived from this file's location, so roxie's own experiments are found whatever
the working directory, and ``./experiments``, which lets a downstream repo keep
its launchables next to its env. When the two coincide only one entry is added.
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
