"""Extend Hydra's config search path to the repo's out-of-package config dirs.

The launchable experiment configs live in top-level ``experiments/`` (grouped
by env into ``ant/``, ``walker/``, ``mocap/`` subfolders) — that tree isn't
inside the installed ``roxie`` package, so Hydra can't see it by default. The
primary ``config_path`` stays at ``roxie/configs`` (the shared ``agent`` /
``noise`` groups); this plugin appends ``experiments/`` so configs there compose
against those groups (e.g. ``--config-name walker/walker_ddpg``). The path is
derived from this file's location, so it holds regardless of the working
directory Hydra runs in.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

# roxie/utils/hydra_searchpath.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

_EXTRA_SEARCH_DIRS = {
    "roxie-experiments": REPO_ROOT / "experiments",
}


class RoxieSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for provider, path in _EXTRA_SEARCH_DIRS.items():
            search_path.append(provider=provider, path=f"file://{path}")


def register() -> None:
    """Register the search-path plugin. Idempotent; call before Hydra composes."""
    Plugins.instance().register(RoxieSearchPathPlugin)
