"""Hydra search path plugin to include experiment configuration directories."""

from pathlib import Path

from hydra.core.config_search_path import ConfigSearchPath
from hydra.core.plugins import Plugins
from hydra.plugins.search_path_plugin import SearchPathPlugin

REPO_ROOT = Path(__file__).resolve().parents[2]


def _search_dirs() -> dict[str, Path]:
    """Returns experiment search directories, prioritizing local project paths."""
    dirs = {"roxie-experiments": REPO_ROOT / "experiments"}
    cwd = Path.cwd().resolve() / "experiments"
    if cwd != dirs["roxie-experiments"] and cwd.is_dir():
        dirs["roxie-local-experiments"] = cwd
    return dirs


class RoxieSearchPathPlugin(SearchPathPlugin):
    """Plugin appending repo-root and local experiment paths to Hydra's search path."""

    def manipulate_search_path(self, search_path: ConfigSearchPath) -> None:
        """Appends resolved experiment paths to Hydra's ConfigSearchPath."""
        for provider, path in _search_dirs().items():
            search_path.append(provider=provider, path=f"file://{path}")


def register() -> None:
    """Registers the Hydra search-path plugin. Safe to call multiple times."""
    Plugins.instance().register(RoxieSearchPathPlugin)