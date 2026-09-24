"""Package trained policies out of the release grid into publishable bundles."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from roxie.environment.suites import DMC_TASKS  # noqa: E402
from roxie.utils.checkpoint import (  # noqa: E402
    CHECKPOINTS_DIRNAME,
    checkpoint_steps,
    read_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_ROOT = REPO_ROOT / "outputs" / "release_v1"
DEFAULT_DONE_DIR = REPO_ROOT / "logs" / ".done"

# Both release cells. envpool_cpu is dm_control's XMLs through EnvPool and
# warp_gpu is mujoco_playground: different physics behind an observation of the
# same 67 dimensions and an action of the same 21, so `_check_obs_width` cannot
# refuse a swap and the cell in the bundle name is the only thing separating
# them.
DEFAULT_CELLS = ("envpool_cpu", "warp_gpu")

# Where --push-to-hub publishes. The version is in the repo id so a later grid
# gets its own repo instead of overwriting this one's history.
DEFAULT_HUB_REPO = "vittorione/roxie-release-v1"
# Frontmatter for weights/README.md, which on the Hub IS the model card. No
# `model-index`: its structured results would publish one headline number per
# task, and on the arms that peaked and collapsed that number is the peak.
MODEL_CARD_FRONTMATTER = """\
---
license: mit
library_name: roxie
pipeline_tag: reinforcement-learning
tags:
  - reinforcement-learning
  - deep-reinforcement-learning
  - dm-control
  - mujoco
  - jax
---
"""

# Selection metric, plus the columns copied into metadata.json alongside it.
# `score_per_step` and `length` come along because on tasks that can terminate
# early the sum alone cannot separate "acts well" from "survives long".
DEFAULT_METRIC = "test/score"
# Matches the "final return" the release table and figures report.
FINAL_EVALS = 10
# How far under the target budget a run may land and still count as having
# reached it. Agents step the env at different per-iteration increments, so one
# 500M-step budget finishes at 500,000,768 for six of the seven and 500,006,912
# for PPO; an exact match would publish only PPO. A piloted or smoke run is
# short by orders of magnitude, not by this.
BUDGET_TOLERANCE = 0.99
REPORTED_COLUMNS = (
    "test/score", "test/score/std", "test/score_per_step",
    "test/length", "test/length/std", "test/distinct_starts",
    "train/score", "train/gradient_steps", "sys/sps", "sys/time/total_s",
)


def discover_runs(out_root: Path, done_dir: Path) -> list[dict]:
    """Every run under a release tree, one dict per run directory.

    There is no ledger file to read: `run_release_benchmark.sh` records a
    finished run by touching `logs/.done/<task>_<agent>_<cell>`, so the tree is
    the record of what ran. The layout is `<task>/<cell>/<agent>/<stamp>/`, the
    same one `roxie/plot.py` walks, and each run's budget and wall clock come
    off its own `log.csv` rather than a column written beside it.

    The marker is per ARM, not per run, so it cannot by itself say that the
    newest attempt at an arm finished — `release_runs` pairs it with the budget
    the run actually reached.
    """
    runs = []
    for log in sorted(out_root.rglob("log.csv")):
        rel = log.parent.relative_to(out_root).parts
        if len(rel) != 4:
            continue
        task, cell, agent, stamp = rel
        rows = read_log(log.parent)
        steps = [v for v in (_as_float(r, "steps") for r in rows) if v is not None]
        if not steps:
            continue
        seconds = [v for v in (_as_float(r, "sys/time/total_s") for r in rows)
                   if v is not None]
        runs.append({
            "task": task, "cell": cell, "agent": agent, "stamp": stamp,
            "run_dir": log.parent,
            "steps": int(max(steps)),
            "seconds": int(max(seconds)) if seconds else 0,
            "done": (done_dir / f"{task}_{agent}_{cell}").is_file(),
        })
    return runs


def final_mean(run_dir: Path, metric: str, n: int = FINAL_EVALS) -> float | None:
    """Mean of a run's last `n` logged values of `metric` — where it ENDED.

    Published next to the selected checkpoint's own score because the two can be
    far apart: several warp_gpu arms peak near the 1000 ceiling and collapse
    well before the budget runs out, and it is the collapse the release figures
    plot. A bundle reporting only its peak would contradict them.
    """
    values = [v for v in (_as_float(r, metric) for r in read_log(run_dir))
              if v is not None]
    if not values:
        return None
    tail = values[-n:]
    return sum(tail) / len(tail)


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
    # Derived from the log rather than assumed, so this holds whatever cadence
    # a run was launched with.
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


def budget_for(runs: list[dict], task: str, cell: str) -> int | None:
    """The longest budget a finished run reached for a (task, cell), or None.

    This is the default `--steps`, and it exists because a smoke pass touches
    the same `.done` marker as a release run. Publishing "the newest finished
    run" would then hand out a 100k-step policy the moment anyone validated the
    grid after training it — a bundle that loads perfectly and is worthless.
    The longest budget present is the release run by construction.
    """
    budgets = [run["steps"] for run in runs
               if run["done"] and run["task"] == task and run["cell"] == cell]
    return max(budgets) if budgets else None


def release_runs(runs: list[dict], task: str, cell: str,
                 min_steps: int | None) -> dict[str, dict]:
    """The run to publish per agent for one (task, cell), as {agent: run}.

    Eligible runs are the ones the benchmark marked done that also came within
    `BUDGET_TOLERANCE` of `min_steps`; among those the latest stamp wins, which
    is what picks a re-run of an arm over its first pass. `min_steps=None`
    accepts any budget.
    """
    floor = None if min_steps is None else min_steps * BUDGET_TOLERANCE
    eligible = [run for run in runs if run["done"] and run["task"] == task
                and run["cell"] == cell
                and (floor is None or run["steps"] >= floor)]
    chosen: dict[str, dict] = {}
    for run in sorted(eligible, key=lambda r: r["stamp"]):
        chosen[run["agent"]] = run
    return chosen


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


def export_one(checkpoint: Path, row: dict, dest: Path, *, run: dict,
               metric: str, commit: str | None) -> dict:
    """Write one bundle and return its metadata dict."""
    run_dir = run["run_dir"]
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
        "task": run["task"],
        "agent": run["agent"],
        "cell": run["cell"],
        "selected_by": metric,
        "checkpoint_steps": checkpoint_steps(checkpoint),
        "budget_steps": run["steps"],
        "epoch": int(float(row["epoch"])) if row.get("epoch") else None,
        "metrics": {c: _as_float(row, c) for c in REPORTED_COLUMNS
                    if _as_float(row, c) is not None},
        "final_score": final_mean(run_dir, metric),
        "final_evals": FINAL_EVALS,
        "source_run": display_path(run_dir),
        "wall_clock_s": run["seconds"],
        "git_commit": commit,
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "bytes": dir_size(dest / CHECKPOINTS_DIRNAME),
        "checkpoint_path": f"{CHECKPOINTS_DIRNAME}/{checkpoint.name}",
    }
    (dest / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def write_index(dest_root: Path, exported: list[dict], *, cells: list[str],
                metric: str, hub_repo: str) -> None:
    """A machine-readable index and a human-readable README over the bundles."""
    index = dest_root / "index.tsv"
    columns = ["task", "cell", "agent", "checkpoint_steps", "budget_steps",
               "test/score", f"test/score_final{FINAL_EVALS}",
               "test/score_per_step", "test/length", "bytes", "git_commit"]
    with index.open("w", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(columns)
        for meta in exported:
            final = meta.get("final_score")
            writer.writerow([
                meta["task"], meta["cell"], meta["agent"],
                meta["checkpoint_steps"], meta["budget_steps"],
                meta["metrics"].get("test/score", ""),
                "" if final is None else f"{final:.1f}",
                meta["metrics"].get("test/score_per_step", ""),
                meta["metrics"].get("test/length", ""),
                meta["bytes"], meta["git_commit"] or "",
            ])

    tasks = sorted({m["task"] for m in exported})
    agents = sorted({m["agent"] for m in exported})
    present = [c for c in cells if any(m["cell"] == c for m in exported)]
    ranked = sorted(exported, key=lambda m: m["metrics"].get(metric, float("-inf")),
                    reverse=True)
    by_key = {(m["task"], m["cell"], m["agent"]): m for m in exported}
    lines = [
        MODEL_CARD_FRONTMATTER,
        "# roxie release weights",
        "",
        f"One policy per (task, cell, agent) — {len(exported)} bundles over "
        f"{len(tasks)} task(s), {len(present)} cell(s) and {len(agents)} "
        f"agent(s) — each selected by best `{metric}` among its run's "
        "checkpoints, not by taking the final checkpoint.",
        "",
        "Scores are dm_control episode returns, so 1000 is the ceiling on every "
        "task and the table is comparable down its rows as well as across them. "
        "Each entry gives the published checkpoint's own score, then in "
        f"parentheses the mean of its run's last {FINAL_EVALS} evaluations — "
        "where the run ENDED, which is what the release figures plot. The two "
        "diverge on an arm that peaked and then collapsed, and the bundle is "
        "the peak.",
        "",
        "| task | cell | " + " | ".join(agents) + " |",
        "|---" * (len(agents) + 2) + "|",
    ]
    for task in tasks:
        for cell in present:
            row = [task, f"`{cell}`"]
            row += [_index_score(by_key.get((task, cell, agent)))
                    for agent in agents]
            lines.append("| " + " | ".join(row) + " |")
    lines += [
        "",
        "The two cells are two different environments, not one environment on "
        "two machines: `envpool_cpu` is dm_control's XMLs through EnvPool and "
        "`warp_gpu` is `mujoco_playground`. Their observations are the same "
        "width, so loading one against the other raises nothing — go by the "
        "cell in the bundle name.",
        "",
        "The full record — per-bundle steps, budget, `test/score_per_step`, "
        "`test/length`, size and the commit each was cut from — is in "
        "`index.tsv` and in each bundle's own `metadata.json`.",
        "",
        "## Playing one back",
        "",
        "Each bundle is self-contained, so pull just the one you want rather "
        "than the whole grid:",
        "",
        "```python",
        "from huggingface_hub import snapshot_download",
        "",
        "path = snapshot_download(",
        f'    repo_id="{hub_repo}",',
        f'    allow_patterns="{_example_bundle(ranked)}/*",',
        ")",
        "```",
        "",
        "Then point roxie at the bundle you downloaded:",
        "",
        "```bash",
        "git clone https://github.com/vittorione94/roxie && cd roxie",
        "uv sync",
        f"uv run python roxie/play.py --checkpoint-path <path>/{_example_bundle(ranked)}/{CHECKPOINTS_DIRNAME}/<N>",
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
        "<Task>/<agent>.<cell>/",
        "  .hydra/config.yaml     resolved run config (play.py reads this)",
        "  .hydra/overrides.yaml  the CLI condition the run was launched with",
        "  checkpoints/<N>/       the orbax checkpoint",
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


def _index_score(meta: dict | None) -> str:
    """`peak (final)` for one bundle, or an em dash where the arm is absent."""
    if meta is None:
        return "—"
    peak = _fmt(meta["metrics"].get("test/score"))
    final = meta.get("final_score")
    return peak if final is None else f"{peak} ({_fmt(final)})"


def _fmt(value: float | None, digits: int = 1) -> str:
    return "—" if value is None else f"{value:,.{digits}f}"


def _example_bundle(ranked: list[dict]) -> str:
    """A real `<Task>/<agent>.<cell>` from this export, so the model card's
    download snippet is copy-pasteable rather than a placeholder."""
    if not ranked:
        return "<Task>/<agent>.<cell>"
    best = ranked[0]
    return f"{best['task']}/{best['agent']}.{best['cell']}"


def push_to_hub(dest_root: Path, repo_id: str, *, private: bool,
                commit: str | None) -> None:
    """Upload the whole bundle tree to a Hugging Face model repo.

    The staged tree maps onto the repo root unchanged, so its `README.md`
    becomes the model card and each bundle stays the self-contained directory
    play.py opens. The Hub serves every file individually, which is what lets
    someone pull one 4MB bundle rather than the whole grid — so there is nothing
    to tar up first.
    """
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit(
            "Pushing needs huggingface_hub: uv sync --extra hub"
        )

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    message = "roxie release weights"
    if commit:
        message += f" (roxie @ {commit[:12]})"
    api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=str(dest_root),
        commit_message=message,
    )
    print(f"  pushed:  https://huggingface.co/{repo_id}")


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

    # Read on the host: a release checkpoint is a GPU run's, and this only
    # inspects the tree. Must precede the first array op, which initializes the
    # backend — importing jax does not.
    jax.config.update("jax_platforms", "cpu")

    checkpoints = find_checkpoints(bundle)
    if not checkpoints:
        return "no checkpoint in bundle"
    steps, path = next(iter(checkpoints.items()))
    try:
        payload = read_checkpoint(path)
    except Exception as error:  # orbax raises a wide variety here
        return f"unreadable ({type(error).__name__}: {error})"
    # No graphdef: `checkpoint_payload` re-derives it from the live modules.
    for key in ("trainstate_state", "hyperparams"):
        if key not in payload:
            return f"payload missing {key!r}"
    recorded = (payload.get("metadata") or {}).get("steps")
    if recorded is not None and int(recorded) != steps:
        return f"metadata says step {int(recorded)}, directory says {steps}"
    if not (bundle / ".hydra" / "config.yaml").is_file():
        return "no .hydra/config.yaml — play.py could not rebuild the agent"
    return None


def export_all(finished: list[dict], dest_root: Path, *, tasks: list[str],
               cells: list[str], agents: set[str] | None, metric: str,
               steps: int | None, any_budget: bool, do_verify: bool,
               dry_run: bool, commit: str | None) -> tuple[list[dict], list]:
    """Stage one bundle per (task, cell, agent). Returns `(exported, skipped)`."""
    exported: list[dict] = []
    skipped: list[tuple[str, str]] = []
    for task in tasks:
        for cell in cells:
            # Per (task, cell), not once for the whole export: a partial grid can
            # have one arm finished at 500M and another piloted at 5M, and a
            # single global budget would silently drop the piloted one.
            budget = None if any_budget else (
                steps if steps is not None else budget_for(finished, task, cell)
            )
            chosen = release_runs(finished, task, cell, budget)
            if agents is not None:
                chosen = {a: r for a, r in chosen.items() if a in agents}
            if not chosen:
                continue

            print(f"{task} / {cell}  ({len(chosen)} run(s) at "
                  f"{f'{budget:,} steps' if budget else 'any budget'})")
            for agent in sorted(chosen):
                run = chosen[agent]
                label = f"{task}/{cell}/{agent}"
                run_dir = run["run_dir"]
                if not run_dir.is_dir():
                    skipped.append((label, f"run dir is gone ({run_dir})"))
                    continue

                picked = pick_checkpoint(run_dir, metric)
                if picked is None:
                    skipped.append((label, "no checkpoint with a logged score"))
                    continue
                ckpt_steps, checkpoint, row = picked
                print(f"  {agent:5s}  step_{ckpt_steps:<12,d} "
                      f"{metric}={_fmt(_as_float(row, metric))}  "
                      f"final{FINAL_EVALS}={_fmt(final_mean(run_dir, metric))}"
                      f"   (best of {len(find_checkpoints(run_dir))} checkpoint(s))")
                if dry_run:
                    continue

                bundle = dest_root / task / f"{agent}.{cell}"
                metadata = export_one(checkpoint, row, bundle, run=run,
                                      metric=metric, commit=commit)
                if do_verify:
                    problem = verify(bundle)
                    print(f"         verify: {problem or 'ok'}")
                    if problem:
                        # Left staged to be looked at, but out of the index and
                        # the upload: a bundle that did not read back is not
                        # something to hand anyone.
                        skipped.append(
                            (label, f"failed verification: {problem}")
                        )
                        continue
                exported.append(metadata)
    return exported, skipped


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT,
                        help="benchmark output tree, walked as "
                             "<task>/<cell>/<agent>/<stamp>/log.csv")
    parser.add_argument("--done-dir", type=Path, default=DEFAULT_DONE_DIR,
                        help="where run_release_benchmark.sh touches its "
                             "<task>_<agent>_<cell> completion markers")
    parser.add_argument("--tasks", default="",
                        help="comma-separated subset; default is every task the "
                             "tree has a finished run for")
    parser.add_argument("--cells", default=",".join(DEFAULT_CELLS),
                        help="comma-separated; one bundle per (task, cell, agent)")
    parser.add_argument("--agents", default="",
                        help="comma-separated subset; default is every agent found")
    parser.add_argument("--metric", default=DEFAULT_METRIC,
                        help="log.csv column the best checkpoint is chosen by")
    parser.add_argument("--steps", type=int, default=None,
                        help="only publish runs that reached this budget "
                             "(default: the longest budget completed for the "
                             "task/cell, so a smoke run cannot be published)")
    parser.add_argument("--any-budget", action="store_true",
                        help="publish the newest finished run per agent whatever "
                             "its budget, mixing budgets in one export")
    parser.add_argument("--hub-repo", default=DEFAULT_HUB_REPO,
                        help="the Hugging Face model repo to publish to")
    parser.add_argument("--private", action="store_true",
                        help="create the hub repo private rather than public")
    parser.add_argument("--dest", type=Path, default=None,
                        help="stage the bundles here and DO NOT push, for "
                             "inspecting what would be published (default: a "
                             "temp dir, uploaded and then discarded)")
    parser.add_argument("--verify", action="store_true",
                        help="read each staged checkpoint back off disk")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="print what would be published, write nothing")
    args = parser.parse_args()

    if not args.out_root.is_dir():
        print(f"No release tree at {args.out_root}. Run "
              f"scripts/run_release_benchmark.sh first.", file=sys.stderr)
        return 1
    runs = discover_runs(args.out_root, args.done_dir)
    finished = [r for r in runs if r["done"]]
    if not finished:
        print(f"{len(runs)} run(s) under {args.out_root}, none with a marker in "
              f"{args.done_dir} — nothing to export.", file=sys.stderr)
        return 1

    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    # Read off the tree rather than DMC_TASKS, which is what makes a partial
    # grid exportable.
    available = {r["task"] for r in finished if r["cell"] in cells}
    if args.tasks:
        wanted_tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
        missing = [t for t in wanted_tasks if t not in available]
        if missing:
            print(f"No finished runs for: {', '.join(missing)}", file=sys.stderr)
        tasks = [t for t in wanted_tasks if t in available]
    else:
        tasks = [t for t in DMC_TASKS if t in available]
        tasks += sorted(available - set(DMC_TASKS))
    if not tasks:
        print(f"No finished runs on cell(s) {', '.join(cells)} under "
              f"{args.out_root}.", file=sys.stderr)
        return 1

    wanted_agents = (
        {a.strip() for a in args.agents.split(",") if a.strip()}
        if args.agents else None
    )
    commit = git_commit()
    pushing = args.dest is None and not args.dry_run
    where = (f"-> {args.hub_repo}" if pushing else
             f"-> {args.dest}" if args.dest else "(dry run)")
    print(f"{len(tasks)} task(s) x {len(cells)} cell(s), selecting by "
          f"{args.metric}  {where}\n")

    # Staged rather than kept: anyone who trained the grid already has the
    # checkpoints under outputs/, so a second copy in the working tree is only
    # the upload's raw material.
    staging = None if args.dest is not None else tempfile.TemporaryDirectory(
        prefix="roxie-weights-")
    dest_root = args.dest if staging is None else Path(staging.name)
    try:
        exported, skipped = export_all(
            finished, dest_root, tasks=tasks, cells=cells, agents=wanted_agents,
            metric=args.metric, steps=args.steps, any_budget=args.any_budget,
            do_verify=args.verify, dry_run=args.dry_run, commit=commit,
        )

        if args.dry_run:
            print("\ndry run — nothing written.")
            return 0

        if exported:
            write_index(dest_root, exported, cells=cells, metric=args.metric,
                        hub_repo=args.hub_repo)
            total_bytes = sum(m["bytes"] for m in exported)
            print(f"\n{len(exported)} bundle(s), {total_bytes / 1e6:.0f} MB")
            # After the index and the card, so what lands on the Hub is the
            # complete tree — and after `skipped` is collected, so a bundle that
            # failed verification is not what gets published.
            if pushing:
                push_to_hub(dest_root, args.hub_repo, private=args.private,
                            commit=commit)
            else:
                print(f"  staged: {dest_root}  (not pushed)")
    finally:
        if staging is not None:
            staging.cleanup()

    if skipped:
        print("\nnot published:", file=sys.stderr)
        for label, why in skipped:
            print(f"  {label}: {why}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
