"""Extend Hydra's config search path to the repo's out-of-package config dirs.

The launchable experiment configs live in top-level ``experiments/`` and the
mocap example keeps its own configs under ``examples/mocap/configs/`` — neither
is inside the installed ``roxie`` package, so Hydra can't see them by default.
The primary ``config_path`` stays at ``roxie/configs`` (the shared ``agent`` /
``noise`` groups); this plugin appends the two external dirs so experiment
configs there compose against those groups. Paths are derived from this file's
location, so they hold regardless of the working directory Hydra runs in.
"""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

# roxie/utils/hydra_searchpath.py -> repo root
REPO_ROOT = Path(__file__).resolve().parents[2]

_EXTRA_SEARCH_DIRS = {
    "roxie-experiments": REPO_ROOT / "experiments",
    "roxie-examples-mocap": REPO_ROOT / "examples" / "mocap" / "configs",
}


class RoxieSearchPathPlugin(SearchPathPlugin):
    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        for provider, path in _EXTRA_SEARCH_DIRS.items():
            search_path.append(provider=provider, path=f"file://{path}")


def register() -> None:
    """Register the search-path plugin. Idempotent; call before Hydra composes."""
    Plugins.instance().register(RoxieSearchPathPlugin)
