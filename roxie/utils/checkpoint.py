"""Locating a run's checkpoints on disk.

A run writes `<output_dir>/checkpoints/<N>/`, one directory per save, where `N`
is the env-step count. To resume, `train.py` needs the exact step directory —
but the path a user has at hand is usually the run directory they copied out of
the console, or the `checkpoints/` dir. This module reduces any of those three
to the one directory `Agent.restore` reads, so `resume=` accepts all of them.

`checkpoints/` is an `orbax.CheckpointManager` directory: it owns the step
naming and the retention policy (`Trainer.replace_checkpoint` -> `max_to_keep`),
and it nests the payload one level further down, under an item subdirectory.
Everything here works on the step directory itself, which is the unit a user
names, that `resume=` accepts, and that `play.py` resolves its run config
relative to.

Nothing here imports JAX: it is pure path arithmetic, so it stays usable (and
testable) without a device.
"""

import re
from pathlib import Path

CHECKPOINTS_DIRNAME = "checkpoints"

# A CheckpointManager step directory holds one subdirectory per item; the
# trainer writes a single unnamed item, which orbax files under "default".
CHECKPOINT_ITEM = "default"

_STEP_RE = re.compile(r"^(\d+)$")


def checkpoint_steps(path: str | Path) -> int | None:
    """Env-step count encoded in a step directory's name, or None.

    The authoritative step count is the one the trainer wrote into the
    checkpoint's metadata; this is the fallback for a checkpoint whose metadata
    did not survive.
    """
    match = _STEP_RE.match(Path(path).name)
    return int(match.group(1)) if match else None


def find_checkpoint(path: str | Path) -> Path:
    """Resolve `path` to the step directory to restore from.

    Accepts, in order of preference:

    - a `<N>` step directory              → itself
    - a directory containing `<N>` dirs   → the one with the HIGHEST N
    - a run output directory              → the same, under `checkpoints/`

    The highest N wins rather than the newest mtime: a run configured with
    `trainer.replace_checkpoint: false` keeps every save, and "latest" must mean
    furthest along in training, which a re-written file's timestamp does not.

    Raises FileNotFoundError if nothing resolves — resuming from the wrong path
    must fail at launch, not silently start a fresh run.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"No such checkpoint path: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Checkpoint path is not a directory: {path}")

    if checkpoint_steps(path) is not None:
        return path.resolve()

    for candidate in (path, path / CHECKPOINTS_DIRNAME):
        if not candidate.is_dir():
            continue
        steps = [
            (checkpoint_steps(child), child)
            for child in candidate.iterdir()
            if child.is_dir() and checkpoint_steps(child) is not None
        ]
        if steps:
            return max(steps, key=lambda item: item[0])[1].resolve()

    raise FileNotFoundError(
        f"No numbered checkpoint directory found under {path}. Pass a run "
        f"output dir, its `{CHECKPOINTS_DIRNAME}/` dir, or one `<N>` step dir."
    )
