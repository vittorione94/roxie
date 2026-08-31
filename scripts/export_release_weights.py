"""Package trained policies out of the release grid into publishable bundles.

The benchmark leaves ~10 checkpoints per run buried in a timestamped output tree
alongside the wandb dir, the console log and a 100-row log.csv, one such tree
per (task, cell, agent). None of that is what you attach to a release. This
script picks ONE checkpoint per (task, agent) — the best-scoring one, not the
last — and copies it into a self-contained bundle `roxie/play.py` can open on
its own:

    weights/CheetahRun/td3.warp_gpu/
      .hydra/config.yaml        the run's resolved config (play.py reads this)
      .hydra/overrides.yaml     the CLI condition, for reproducibility
      checkpoints/<N>/          the orbax checkpoint itself
      metadata.json             what it scored, where it came from, which commit

The layout is not cosmetic: play.py resolves its config as
`<checkpoint>/../../.hydra/config.yaml`, so the bundle has to mirror a run dir's
shape for the checkpoint path to stay openable. That is why the checkpoint keeps
its `checkpoints/<N>/` nesting instead of being flattened.

    uv run python scripts/export_release_weights.py --dry-run
    uv run python scripts/export_release_weights.py
    uv run python scripts/export_release_weights.py --verify --archive

WHICH TASKS. Every task the manifest has an `ok` run for on the chosen cell, so
a full grid publishes 25 x 7 policies and a partial one publishes what it has.
`--tasks` narrows it.

WHICH RUN. The manifest (`outputs/release_v1/manifest.tsv`) is the source of
truth, and only rows recorded `ok` are eligible — a run that crashed at 40% has
checkpoints on disk that look perfectly loadable and are not a release result.
Among those, only runs at the LONGEST completed budget for the task/cell, which
is what keeps a `--smoke` run out: smoke records itself `ok` in the same ledger
at 100k steps, and "the newest ok run" would publish that the moment anyone
validated the grid after training it. `--steps` pins a specific budget instead;
`--any-budget` opts out and takes the newest run whatever it is.

WHICH CHECKPOINT. The best `test/score` in the run's own log.csv, among the
steps that actually have a checkpoint. Taking the last checkpoint instead would
publish whatever the policy happened to be doing when the budget ran out, which
on a saturating arm is measurably worse than its own peak.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roxie.environment.suites import DMC_TASKS  # noqa: E402
from roxie.utils.checkpoint import (  # noqa: E402
    CHECKPOINTS_DIRNAME,
    checkpoint_steps,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "release_v1"
DEFAULT_DEST = REPO_ROOT / "weights"

# The headline cell. The other cell runs the same algorithm on the same task, so
# publishing it too would ship near-duplicate policies under different names —
# and a policy trained against playground's observation layout is not loadable
# against EnvPool's anyway.
DEFAULT_CELL = "warp_gpu"

# Selection metric, plus the columns copied into metadata.json alongside it.
# `test/score` is the dm_control episode return, 0-1000 across the suite.
# `test/score_per_step` and `test/length` come along because on tasks that can
# terminate early the sum alone cannot separate "acts well" from "survives long".
DEFAULT_METRIC = "test/score"
REPORTED_COLUMNS = (
    "test/score", "test/score/std", "test/score_per_step",
    "test/length", "test/length/std", "test/distinct_starts",
    "train/score", "train/gradient_steps", "sys/sps", "sys/time/total_s",
)


# --------------------------------------------------------------- reading -----


def read_manifest(manifest: Path) -> list[dict]:
    """The benchmark's run ledger, as dicts. Missing file -> no runs."""
    if not manifest.is_file():
        return []
    with manifest.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_log(run_dir: Path) -> list[dict]:
    """A run's log.csv as dicts, or [] if it never got far enough to write one."""
    log = run_dir / "log.csv"
    if not log.is_file():
        return []
    with log.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: dict, column: str) -> float | None:
    """A logged value, or None. The CSV backend writes the string "None" for a
    metric an epoch did not produce (a gap, not a zero), so that spelling has to
    survive the round-trip rather than becoming 0.0."""
    raw = (row.get(column) or "").strip()
    if not raw or raw == "None":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def find_checkpoints(run_dir: Path) -> dict[int, Path]:
    """Every `<N>` step checkpoint in a run, keyed by N."""
    root = run_dir / CHECKPOINTS_DIRNAME
    if not root.is_dir():
        return {}
    found = {}
    for child in sorted(root.iterdir()):
        steps = checkpoint_steps(child) if child.is_dir() else None
        if steps is not None:
            found[steps] = child
    return found


# ------------------------------------------------------------- selecting -----


def pick_checkpoint(run_dir: Path, metric: str) -> tuple[int, Path, dict] | None:
    """The best-scoring checkpoint of a run: `(steps, path, epoch_row)`.

    Only checkpointed steps are candidates — the eval runs every `epoch_steps`
    and saves happen every `save_steps`, so most evaluated epochs have no
    checkpoint to publish and scoring them would pick a policy that no longer
    exists on disk. An epoch row matches a checkpoint when their step counts are
    within one epoch's worth of each other; both cadences are multiples of the
    per-iteration step increment, so in practice this is an exact match, and the
    tolerance only covers a run whose cadences were retuned mid-flight.

    Returns None when the run has no checkpoint, or has checkpoints but never
    logged the metric (a run killed before its first epoch boundary).
    """
    checkpoints = find_checkpoints(run_dir)
    if not checkpoints:
        return None

    rows = read_log(run_dir)
    # Half an epoch, derived from the log rather than assumed, so this holds
    # whatever cadence a run was launched with.
    tolerance = _epoch_tolerance(rows)
    scored: list[tuple[float, int, dict]] = []
    for row in rows:
        value = _as_float(row, metric)
        steps = _as_float(row, "steps")
        if value is None or steps is None:
            continue
        steps = int(steps)
        nearest = min(checkpoints, key=lambda n: abs(n - steps))
        if abs(nearest - steps) <= tolerance:
            scored.append((value, nearest, row))

    if not scored:
        return None
    # A tie goes to the later checkpoint, which has more training behind it.
    value, steps, row = max(scored, key=lambda item: (item[0], item[1]))
    return steps, checkpoints[steps], row


def _epoch_tolerance(rows: list[dict]) -> float:
    """Half the run's epoch size in env steps, from consecutive `steps` values."""
    steps = [_as_float(row, "steps") for row in rows]
    steps = [s for s in steps if s is not None]
    if len(steps) < 2:
        return 0.0
    return max(b - a for a, b in zip(steps, steps[1:])) / 2


def budget_for(manifest_rows: list[dict], task: str, cell: str) -> int | None:
    """The largest completed step budget for a (task, cell), or None.

    This is the default `--steps`, and it exists because `--smoke` records its
    tiny runs as `ok` in the same ledger. Publishing "the newest ok run" would
    then hand out a 500k-step smoke policy the moment anyone validated the grid
    after training it — a bundle that loads perfectly and is worthless. The
    longest budget present is the release run by construction.
    """
    budgets = [
        int(row["steps"])
        for row in manifest_rows
        if row.get("status") == "ok" and row.get("task") == task
        and row.get("cell") == cell and (row.get("steps") or "").isdigit()
    ]
    return max(budgets) if budgets else None


def latest_ok_runs(manifest_rows: list[dict], task: str, cell: str,
                   steps: int | None) -> dict[str, dict]:
    """The newest `ok` run per agent for one (task, cell), as {agent: row}.

    Newest by manifest ORDER, not by parsing the timestamp out of the path: the
    ledger is append-only, so a later line is a later run by construction, and
    that stays true if the run-dir naming ever changes.

    `steps=None` accepts any budget; callers should pass `budget_for(...)`
    instead unless they mean to mix budgets in one export.
    """
    chosen: dict[str, dict] = {}
    for row in manifest_rows:
        if row.get("status") != "ok":
            continue
        if row.get("task") != task or row.get("cell") != cell:
            continue
        if steps is not None and row.get("steps") != str(steps):
            continue
        chosen[row["agent"]] = row
    return chosen


# -------------------------------------------------------------- writing ------


def git_commit() -> str | None:
    """The commit the export was cut from. None outside a git checkout."""
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def display_path(path: Path) -> str:
    """Repo-relative when it can be, absolute otherwise. Paths go into
    metadata.json and into the README's copy-pasteable play command, and
    `--dest` may point anywhere on the filesystem — a bare `relative_to` raises
    on a destination outside the checkout."""
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) \
        else str(path)


def export_one(run_dir: Path, checkpoint: Path, row: dict, dest: Path, *,
               task: str, cell: str, agent: str, metric: str,
               manifest_row: dict, commit: str | None) -> dict:
    """Write one bundle and return its metadata dict."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # play.py resolves `<checkpoint>/../../.hydra/config.yaml`, so both the
    # nesting and the dotted dir name have to be preserved verbatim.
    hydra_src = run_dir / ".hydra"
    if hydra_src.is_dir():
        shutil.copytree(hydra_src, dest / ".hydra")
    shutil.copytree(checkpoint, dest / CHECKPOINTS_DIRNAME / checkpoint.name)

    metadata = {
        "task": task,
        "agent": agent,
        "cell": cell,
        "selected_by": metric,
        "checkpoint_steps": checkpoint_steps(checkpoint),
        "budget_steps": int(manifest_row["steps"]),
        "epoch": int(float(row["epoch"])) if row.get("epoch") else None,
        "metrics": {c: _as_float(row, c) for c in REPORTED_COLUMNS
                    if _as_float(row, c) is not None},
        "source_run": display_path(run_dir),
        "wall_clock_s": int(manifest_row["seconds"]),
        "git_commit": commit,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": dir_size(dest / CHECKPOINTS_DIRNAME),
        "play": (
            "uv run python roxie/play.py --checkpoint-path "
            f"{display_path(dest / CHECKPOINTS_DIRNAME / checkpoint.name)}"
        ),
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_index(dest_root: Path, exported: list[dict], *, cell: str,
                metric: str) -> None:
    """A machine-readable index and a human-readable README over the bundles."""
    index = dest_root / "index.tsv"
    columns = ["task", "cell", "agent", "checkpoint_steps", "budget_steps",
               "test/score", "test/score_per_step", "test/length", "bytes",
               "git_commit"]
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for meta in exported:
            writer.writerow([
                meta["task"], meta["cell"], meta["agent"],
                meta["checkpoint_steps"], meta["budget_steps"],
                meta["metrics"].get("test/score", ""),
                meta["metrics"].get("test/score_per_step", ""),
                meta["metrics"].get("test/length", ""),
                meta["bytes"], meta["git_commit"] or "",
            ])

    tasks = sorted({m["task"] for m in exported})
    agents = sorted({m["agent"] for m in exported})
    ranked = sorted(exported, key=lambda m: m["metrics"].get(metric, float("-inf")),
                    reverse=True)
    lines = [
        "# roxie release weights",
        "",
        f"One policy per (task, agent) from the `{cell}` cell of the release "
        f"benchmark — {len(exported)} bundles over {len(tasks)} task(s) and "
        f"{len(agents)} agent(s) — each selected by best `{metric}` among its "
        "run's checkpoints, not by taking the final checkpoint.",
        "",
        "Scores are dm_control episode returns, so 1000 is the ceiling on every "
        "task and the table is comparable down its rows as well as across them.",
        "",
        "| task | " + " | ".join(agents) + " |",
        "|---" * (len(agents) + 1) + "|",
    ]
    by_cell = {(m["task"], m["agent"]): m for m in exported}
    for task in tasks:
        row = [task]
        for agent in agents:
            meta = by_cell.get((task, agent))
            row.append(_fmt(meta["metrics"].get("test/score")) if meta else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "The full record — per-bundle steps, budget, `test/score_per_step`, "
        "`test/length`, size and the commit each was cut from — is in "
        "`index.tsv` and in each bundle's own `metadata.json`.",
        "",
        "## Playing one back",
        "",
        "```bash",
        f"uv run python roxie/play.py --checkpoint-path {_example_path(ranked)}",
        "```",
        "",
        "Playback is forced onto CPU and MJX physics, so a warp-trained bundle "
        "needs no GPU and no warp install to open. Each bundle carries the run's "
        "own `.hydra/config.yaml`, which is what play.py rebuilds the agent and "
        "the env from — keep the directory intact.",
        "",
        "## What is in a bundle",
        "",
        "```",
        f"<Task>/<agent>.{cell}/",
        "  .hydra/config.yaml     resolved run config (play.py reads this)",
        "  .hydra/overrides.yaml  the CLI condition the run was launched with",
        "  checkpoints/<N>/      the orbax checkpoint",
        "  metadata.json          score, provenance, source run, git commit",
        "```",
        "",
        "The checkpoint carries target networks, optimizer slots and the "
        "observation normalizer as well as the policy, so a bundle also works as "
        "a training restart: `resume=<bundle dir>`.",
        "",
        "Regenerate with `uv run python scripts/export_release_weights.py`.",
        "",
    ]
    (dest_root / "README.md").write_text("\n".join(lines))


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _example_path(ranked: list[dict]) -> str:
    """A real path from this export, so the README's command is copy-pasteable."""
    if not ranked:
        return "weights/<Task>/<agent>.<cell>/checkpoints/<N>"
    return ranked[0]["play"].split()[-1]


def make_archives(dest_root: Path, exported: list[dict]) -> None:
    """One .tar.gz per bundle plus SHA256SUMS, for attaching to a release."""
    archive_dir = dest_root / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    digests = []
    for meta in exported:
        name = f"{meta['agent']}.{meta['cell']}"
        bundle = dest_root / meta["task"] / name
        tarball = archive_dir / f"roxie-{meta['task']}-{name}.tar.gz"
        with tarfile.open(tarball, "w:gz") as tar:
            tar.add(bundle, arcname=name)
        digest = hashlib.sha256(tarball.read_bytes()).hexdigest()
        digests.append(f"{digest}  {tarball.name}")
        print(f"    archived {tarball.name}  ({tarball.stat().st_size / 1e6:.0f} MB)")
    (archive_dir / "SHA256SUMS").write_text("\n".join(digests) + "\n")


# ------------------------------------------------------------- verifying -----


def verify(bundle: Path) -> str | None:
    """Read the exported checkpoint back. Returns an error string, or None.

    This is a copy check, not a behavioural one: it confirms the orbax tree is
    complete and self-consistent (the payload's own metadata agrees with the
    directory name it was copied into), which is what a truncated or partial
    copy breaks. Confirming the POLICY is the one that scored requires building
    the env and rolling it out — that is `roxie/play.py`, and the README points
    at it.
    """
    import jax

    jax.config.update("jax_platforms", "cpu")
    from roxie.agents.agent import _read_checkpoint

    checkpoints = find_checkpoints(bundle)
    if not checkpoints:
        return "no checkpoint in bundle"
    steps, path = next(iter(checkpoints.items()))
    try:
        payload = _read_checkpoint(path)
    except Exception as error:  # orbax raises a wide variety here
        return f"unreadable ({type(error).__name__}: {error})"
    for key in ("trainstate_state", "trainstate_graphdef", "hyperparams"):
        if key not in payload:
            return f"payload missing {key!r}"
    recorded = (payload.get("metadata") or {}).get("steps")
    if recorded is not None and int(recorded) != steps:
        return f"metadata says step {int(recorded)}, directory says {steps}"
    if not (bundle / ".hydra" / "config.yaml").is_file():
        return "no .hydra/config.yaml — play.py could not rebuild the agent"
    return None


# ------------------------------------------------------------------ main -----


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="benchmark output tree (holds manifest.tsv)")
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                        help="where to write the bundles")
    parser.add_argument("--tasks", default="",
                        help="comma-separated subset; default is every task the "
                             "manifest has an ok run for on this cell")
    parser.add_argument("--cell", default=DEFAULT_CELL)
    parser.add_argument("--agents", default="",
                        help="comma-separated subset; default is every agent found")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="log.csv column the best checkpoint is chosen by")
    parser.add_argument("--steps", type=int, default=None,
                        help="only publish runs that finished at this budget "
                             "(default: the longest budget completed for the "
                             "task/cell, so a --smoke run cannot be published)")
    parser.add_argument("--any-budget", action="store_true",
                        help="publish the newest ok run per agent whatever its "
                             "budget, mixing budgets in one export")
    parser.add_argument("--archive", action="store_true",
                        help="also write one .tar.gz per bundle + SHA256SUMS")
    parser.add_argument("--verify", action="store_true",
                        help="read each exported checkpoint back off disk")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="print what would be published, write nothing")
    args = parser.parse_args()

    rows = read_manifest(args.out_root / "manifest.tsv")
    if not rows:
        print(f"No manifest at {args.out_root / 'manifest.tsv'} — nothing to "
              f"export. Run scripts/run_release_benchmark.sh first.",
              file=sys.stderr)
        return 1

    # Read off the ledger rather than DMC_TASKS, which is what makes a partial
    # grid exportable: 4 tasks done, 4 published, no failures for the rest.
    available = {
        row["task"] for row in rows
        if row.get("status") == "ok" and row.get("cell") == args.cell
        and row.get("task")
    }
    if args.tasks:
        wanted_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in wanted_tasks if t not in available]
        if missing:
            print(f"No `ok` {args.cell} runs for: {', '.join(missing)}",
                  file=sys.stderr)
        tasks = [t for t in wanted_tasks if t in available]
    else:
        tasks = [t for t in DMC_TASKS if t in available]
        tasks += sorted(available - set(DMC_TASKS))
    if not tasks:
        print(f"No `ok` runs on cell {args.cell} in the manifest.",
              file=sys.stderr)
        return 1

    wanted_agents = (
        {a.strip() for a in args.agents.split(",") if a.strip()}
        if args.agents else None
    )

    commit = git_commit()
    dest_root = args.dest
    exported: list[dict] = []
    skipped: list[tuple[str, str]] = []

    print(f"cell {args.cell} — {len(tasks)} task(s), selecting by {args.metric}\n")
    for task in tasks:
        # Per task, not once for the whole export: a partial grid can have one
        # task finished at 50M and another piloted at 5M, and a single global
        # budget would silently drop the piloted one.
        budget = None if args.any_budget else (
            args.steps if args.steps is not None
            else budget_for(rows, task, args.cell)
        )
        runs = latest_ok_runs(rows, task, args.cell, budget)
        if wanted_agents is not None:
            runs = {a: r for a, r in runs.items() if a in wanted_agents}
        if not runs:
            continue

        print(f"{task}  ({len(runs)} run(s) at "
              f"{f'{budget:,} steps' if budget else 'any budget'})")
        for agent in sorted(runs):
            manifest_row = runs[agent]
            label = f"{task}/{agent}"
            run_dir = Path(manifest_row["run_dir"])
            if not run_dir.is_absolute():
                run_dir = REPO_ROOT / run_dir
            if not run_dir.is_dir():
                skipped.append((label, f"run dir is gone ({run_dir})"))
                continue

            picked = pick_checkpoint(run_dir, args.metric)
            if picked is None:
                skipped.append((label, "no checkpoint with a logged score"))
                continue
            steps, checkpoint, row = picked
            score = _as_float(row, args.metric)
            total = len(find_checkpoints(run_dir))

            print(f"  {agent:5s}  step_{steps:<12,d} {args.metric}={_fmt(score)}"
                  f"   (best of {total} checkpoint(s))")
            if args.dry_run:
                continue

            bundle = dest_root / task / f"{agent}.{args.cell}"
            metadata = export_one(
                run_dir, checkpoint, row, bundle,
                task=task, cell=args.cell, agent=agent, metric=args.metric,
                manifest_row=manifest_row, commit=commit,
            )
            if args.verify:
                problem = verify(bundle)
                print(f"         verify: {problem or 'ok'}")
                if problem:
                    # Left on disk to be looked at, but out of the index and the
                    # archives: a bundle that did not read back is not something
                    # to hand anyone.
                    skipped.append(
                        (label, f"exported but failed verification: {problem}")
                    )
                    continue
            exported.append(metadata)

    if args.dry_run:
        print("\ndry run — nothing written.")
        return 0

    if exported:
        write_index(dest_root, exported, cell=args.cell, metric=args.metric)
        if args.archive:
            make_archives(dest_root, exported)
        total_bytes = sum(m["bytes"] for m in exported)
        print(f"\n{len(exported)} bundle(s) -> {dest_root} "
              f"({total_bytes / 1e6:.0f} MB)")
        print(f"  index:  {dest_root / 'index.tsv'}")
        print(f"  readme: {dest_root / 'README.md'}")

    if skipped:
        print("\nnot published:", file=sys.stderr)
        for agent, why in skipped:
            print(f"  {agent}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
