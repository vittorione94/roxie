"""Assemble the W&B release report from the benchmark runs.

`scripts/run_release_benchmark.sh` streams every run to a project named after
its ENV — `roxie-CheetahRun`, `roxie-WalkerWalk`, one per task — with the agent
and the backend cell as the run identity inside it (`name = "<agent>.<cell>"`,
`job_type = <cell>`) and `group` naming the grid. This script reads that
structure back across all 25 projects and builds one report out of it: a section
per task, each holding the agent comparison for that task.

    uv run python roxie/report.py                       # build and publish
    uv run python roxie/report.py --dry-run             # print structure only
    uv run python roxie/report.py --tasks CheetahRun,WalkerWalk
    uv run python roxie/report.py --entity my-team

One project per env rather than env tags, because a project is the unit W&B
gives a workspace, a run table and cross-run charts to: its default view is then
already the comparison that means something — same task, same budget, every
agent — instead of 350 runs on incomparable score scales.

The report itself must live in one project, so it is written into whichever
`--report-project` names (by default the first task's) and its runsets reach
across the others.

W&B has no "overwrite report by title" API, so each invocation publishes a NEW
report and prints its URL; delete superseded ones in the UI.

Requires the optional `report` extra (wandb-workspaces):

    uv sync --extra report
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

from roxie.environment.suites import DMC_TASKS

# Projects the grid writes to are exactly f"{PROJECT_PREFIX}{task}".
DEFAULT_PROJECT_PREFIX = "roxie-"
# `group` on every run of the v1 grid (release.grid in experiments/dmc/bench).
DEFAULT_GRID = "release-v1"

# Backend cells in display order, keyed by job_type.
CELL_INFO = {
    "warp_gpu": "GPU physics (mujoco_warp) + GPU learner",
    "envpool_cpu": "CPU physics (native MuJoCo pool) + CPU learner — fully GPU-free",
    "mjx_gpu": "GPU physics (MJX) + GPU learner",
    "mjx_cpu": "CPU physics (MJX) + CPU learner — fully GPU-free",
}

HOW_TO_READ = (
    "One section per dm_control task, and inside it every agent at matched "
    "hyperparameters and a matched step budget, on both physics "
    "implementations. Score is the dm_control episode return, so 1000 is the "
    "ceiling on every task here and the sections are directly comparable to "
    "each other. Three caveats belong in any ranking: PPO is on-policy and "
    "therefore not replay-ratio comparable (judge it on score-vs-steps and "
    "score-vs-wall-clock, never on gradient steps); MPO pays ~20x per gradient "
    "step for its 20 action samples; and ranks should come from the back-half "
    "mean of a curve, not a peak epoch. Where the two cells disagree on SCORE "
    "that is a finding about one of the two implementations of the task — they "
    "are independent codebases, not the same program on different hardware. "
    "Where they disagree on WALL-CLOCK, that is the cells doing their job."
)


def _import_workspaces():
    try:
        import wandb_workspaces.reports.v2 as wr
    except ImportError:
        sys.exit(
            "wandb-workspaces is not installed — it is an optional extra.\n"
            "  uv sync --extra report\n"
            "(or `uv add wandb-workspaces` if you are not using the extras)."
        )
    return wr


def discover(entity, prefix, grid_name, tasks=None):
    """Group the grid's runs into {task: {cell: [agent, ...]}}.

    Reading the grid back from W&B rather than from the shell script's own list
    means the report describes what actually RAN — a task or cell that failed or
    was never launched simply does not get a section, instead of an empty one.

    Projects are probed by NAME rather than listed, because `Api().projects()`
    needs an entity and returns everything in it; asking for the 25 the suite
    defines is both cheaper and immune to unrelated projects that happen to
    share the prefix. A project that does not exist yet raises, which here just
    means "this task has not run".
    """
    import wandb

    api = wandb.Api()
    grid = defaultdict(lambda: defaultdict(list))
    found = 0
    for task in tasks or DMC_TASKS:
        project = f"{prefix}{task}"
        path = f"{entity}/{project}" if entity else project
        try:
            runs = list(api.runs(path, filters={"group": grid_name}))
        except Exception:
            continue
        for run in runs:
            # Set from release.cell, so a run missing it was not launched by
            # the grid.
            cell = run.job_type
            if not cell:
                continue
            agent = run.name.split(".", 1)[0]
            if agent not in grid[task][cell]:
                grid[task][cell].append(agent)
            found += 1
    return grid, found


def _ordered(keys, order):
    """`order` first (those present), then anything else, alphabetically."""
    known = [k for k in order if k in keys]
    return known + sorted(k for k in keys if k not in order)


def _runset(wr, entity, task, prefix, grid_name, name, cell=None):
    filters = [f'Group = "{grid_name}"']
    if cell:
        filters.append(f'JobType = "{cell}"')
    return wr.Runset(
        entity=entity or "",
        project=f"{prefix}{task}",
        name=name,
        filters=" and ".join(filters),
    )


def _task_panels(wr):
    """The two panels every task gets, side by side.

    Deliberately two and not four: at 25 tasks a four-panel grid each is 100
    panels in one report, which nobody scrolls. Score-vs-steps ranks the
    algorithms and score-vs-wall-clock ranks the (algorithm, backend) pairs;
    the per-run diagnostics live in each project's own workspace, which is
    exactly what one-project-per-env buys.
    """
    axis = dict(log_x=False, ignore_outliers=False)
    return [
        wr.LinePlot(
            title="Eval score vs env steps",
            x="steps", y=["test/score"],
            title_x="environment steps", title_y="test/score",
            layout=wr.Layout(x=0, y=0, w=12, h=8), **axis,
        ),
        wr.LinePlot(
            title="Eval score vs wall-clock",
            x="sys/time/total_s", y=["test/score"],
            title_x="wall-clock seconds", title_y="test/score",
            layout=wr.Layout(x=12, y=0, w=12, h=8), **axis,
        ),
    ]


def build(wr, grid, entity, prefix, grid_name, report_project, title, description):
    blocks = [
        wr.H1("What this is"),
        wr.P(description),
        wr.P(HOW_TO_READ),
        wr.H1("The cells"),
        wr.P(
            " · ".join(
                f"{cell}: {blurb}" for cell, blurb in CELL_INFO.items()
                if any(cell in cells for cells in grid.values())
            )
        ),
    ]

    for task in _ordered(grid.keys(), list(DMC_TASKS)):
        cells = grid[task]
        ordered_cells = _ordered(cells.keys(), list(CELL_INFO))
        agents = sorted({a for cell in ordered_cells for a in cells[cell]})
        blocks += [
            wr.H1(task),
            wr.P(
                f"{len(agents)} agents ({', '.join(agents)}) over "
                f"{len(ordered_cells)} cell(s) ({', '.join(ordered_cells)}). "
                f"Project: {prefix}{task}."
            ),
            # One runset per cell rather than per task: the two are different
            # implementations, so they get their own colour groups.
            wr.PanelGrid(
                runsets=[
                    _runset(wr, entity, task, prefix, grid_name, cell, cell=cell)
                    for cell in ordered_cells
                ],
                panels=_task_panels(wr),
            ),
        ]

    return wr.Report(
        project=report_project,
        # "" means "my default entity"; the pydantic model rejects None.
        entity=entity or "",
        title=title,
        description=description,
        width="fluid",
        blocks=blocks,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build the W&B release report from the benchmark runs."
    )
    parser.add_argument("--project-prefix", default=DEFAULT_PROJECT_PREFIX,
                        help="projects are <prefix><Task> "
                             f"(default: {DEFAULT_PROJECT_PREFIX})")
    parser.add_argument("--grid", default=DEFAULT_GRID,
                        help="only include runs whose wandb group is this "
                             f"(default: {DEFAULT_GRID})")
    parser.add_argument("--tasks", default=None,
                        help="comma-separated subset of the suite to report on")
    parser.add_argument("--entity", default=None,
                        help="W&B entity; defaults to your default entity")
    parser.add_argument("--report-project", default=None,
                        help="project the report itself is filed under "
                             "(default: the first task's project)")
    parser.add_argument("--title", default="Roxie v1 — release benchmark")
    parser.add_argument(
        "--description",
        default=(
            "Every agent in roxie on the whole dm_control suite, run twice: "
            "once on GPU through mujoco_playground, once fully GPU-free through "
            "EnvPool's native-MuJoCo pool."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="build the report and print its structure without publishing")
    args = parser.parse_args()

    wr = _import_workspaces()

    tasks = [t.strip() for t in args.tasks.split(",")] if args.tasks else None
    grid, n_runs = discover(args.entity, args.project_prefix, args.grid, tasks)
    if not grid:
        sys.exit(
            f"no runs in group '{args.grid}' found in any "
            f"'{args.project_prefix}<Task>' project.\n"
            "Run scripts/run_release_benchmark.sh first (without --offline)."
        )

    print(f"found {n_runs} runs across {len(grid)} task(s):")
    for task in _ordered(grid.keys(), list(DMC_TASKS)):
        for cell, agents in grid[task].items():
            print(f"  {task:<22} {cell:<13} {len(agents)} agents: "
                  f"{' '.join(sorted(agents))}")

    report_project = args.report_project or (
        f"{args.project_prefix}{_ordered(grid.keys(), list(DMC_TASKS))[0]}"
    )
    report = build(wr, grid, args.entity, args.project_prefix, args.grid,
                   report_project, args.title, args.description)

    if args.dry_run:
        print("\ndry run — report structure:")
        for block in report.blocks:
            kind = type(block).__name__
            if kind in ("H1", "H2", "H3"):
                indent = "  " * (int(kind[1]) - 1)
                print(f"  {indent}{block.text}")
            elif kind == "PanelGrid":
                names = ", ".join(rs.name for rs in block.runsets)
                print(f"      [panel grid: {len(block.panels)} panels over {names}]")
        return

    report.save()
    print(f"\npublished: {report.url}")


if __name__ == "__main__":
    main()
