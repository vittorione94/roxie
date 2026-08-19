"""Assemble the W&B release report from the benchmark runs.

`scripts/run_release_benchmark.sh` streams every run to one W&B project with a
fixed identity — group = suite, job_type = backend cell, name = "<agent>.<cell>",
plus a `release-v1` tag. This script reads that structure back and builds a
report out of it: one section per suite, a panel grid per backend cell, and the
cross-cell comparison for the agents that ran on more than one.

    uv run python roxie/report.py                       # build and publish
    uv run python roxie/report.py --dry-run             # print structure only
    uv run python roxie/report.py --project my-project --entity my-team

It is idempotent in the sense that re-running it after more runs land rebuilds
the report from whatever is now in the project — nothing here is hand-placed.
W&B has no "overwrite report by title" API, so each invocation publishes a NEW
report and prints its URL; delete superseded ones in the UI.

Requires the optional `report` extra (wandb-workspaces):

    uv sync --extra report
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict

DEFAULT_PROJECT = "roxie-release-v1"
DEFAULT_TAG = "release-v1"

# Presentation order and prose for the suites the benchmark defines. A suite
# found in the project but missing here still renders — it just gets no blurb —
# so adding a third task to the grid does not require editing this file.
SUITE_INFO = {
    "walker_walk": (
        "Simple task — WalkerWalk",
        "mujoco_playground's WalkerWalk (dm_control walker, planar, 6 actuators). "
        "Every agent runs the same matched hyperparameters (nets [256, 256] + layer "
        "norm, replay ratio 5.0, batch 512) at 256 parallel envs, and every backend "
        "cell runs the same step budget. Score is the dm_control episode return, so "
        "1000 is the ceiling.",
    ),
    "mocap_cmu_006_13": (
        "Complex task — CMU_006_13 motion tracking",
        "Single-clip overfit of CMU_006_13 (40.8 s, 1631 frames) on the dm_control "
        "CMU humanoid: 56 position-servo actuators, ~1069-dim observation, weighted "
        "pose/velocity/end-effector/root tracking reward with early termination. "
        "Eval is the canonical protocol — start at frame 0, no reset noise, run the "
        "clip to its end — so `test/score` tracks `test/length` closely and both "
        "belong in the reading.",
    ),
}

# Backend cells, in the order they should appear, with the one-liner that says
# what the cell actually IS. Keys are job_type values.
CELL_INFO = {
    "warp_gpu": "GPU physics (mujoco_warp) + GPU learner",
    "mjx_gpu": "GPU physics (MJX) + GPU learner",
    "mjx_cpu": "CPU physics (MJX) + CPU learner — fully GPU-free",
    "envpool_cpu": "CPU physics (native MuJoCo pool) + CPU learner — fully GPU-free",
    "envpool_gpu": "CPU physics (native MuJoCo pool) + GPU learner, async",
}


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


def discover(entity, project, tag):
    """Group the project's tagged runs into {suite: {cell: [agent, ...]}}.

    Reading the grid back from W&B rather than from the shell script's own list
    means the report describes what actually RAN — a cell that failed or was
    never launched simply does not get a panel, instead of getting an empty one.
    """
    import wandb

    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"tags": tag})

    grid = defaultdict(lambda: defaultdict(list))
    found = 0
    for run in runs:
        # `group`/`job_type` are set from release.suite / release.cell in the
        # benchmark yamls; a run missing them was not launched by the grid.
        suite, cell = run.group, run.job_type
        if not suite or not cell:
            continue
        agent = run.name.split(".", 1)[0]
        if agent not in grid[suite][cell]:
            grid[suite][cell].append(agent)
        found += 1
    return grid, found


def _ordered(keys, order):
    """`order` first (those present), then anything else, alphabetically."""
    known = [k for k in order if k in keys]
    return known + sorted(k for k in keys if k not in order)


def _runset(wr, entity, project, name, suite, cell=None, agent=None):
    filters = [f'Group = "{suite}"']
    if cell:
        filters.append(f'JobType = "{cell}"')
    if agent:
        # Run names are exactly "<agent>.<cell>" (set in the benchmark yamls),
        # so with the cell already known this is an exact match rather than a
        # prefix search — no dependence on how the filter language handles
        # partial strings.
        assert cell, "filtering by agent requires the cell (run names are agent.cell)"
        filters.append(f'Name = "{agent}.{cell}"')
    return wr.Runset(
        entity=entity or "",
        project=project,
        name=name,
        filters=" and ".join(filters),
    )


def _curve_panels(wr, suite):
    """The four panels every cell gets, in a 2x2 grid."""
    score_axis = dict(log_x=False, ignore_outliers=False)
    return [
        wr.LinePlot(
            title="Eval score vs env steps",
            x="steps", y=["test/score"],
            title_x="environment steps", title_y="test/score",
            layout=wr.Layout(x=0, y=0, w=12, h=8), **score_axis,
        ),
        wr.LinePlot(
            title="Eval episode length vs env steps",
            x="steps", y=["test/length"],
            title_x="environment steps", title_y="test/length",
            layout=wr.Layout(x=12, y=0, w=12, h=8), **score_axis,
        ),
        wr.LinePlot(
            title="Eval score vs wall-clock",
            x="time/total_s", y=["test/score"],
            title_x="wall-clock seconds", title_y="test/score",
            layout=wr.Layout(x=0, y=8, w=12, h=8), **score_axis,
        ),
        wr.LinePlot(
            title="Throughput (steps/s)",
            x="steps", y=["sps"],
            title_x="environment steps", title_y="steps/s",
            layout=wr.Layout(x=12, y=8, w=12, h=8), **score_axis,
        ),
    ]


def build(wr, grid, entity, project, title, description):
    blocks = [
        wr.H1("What this is"),
        wr.P(description),
        wr.P(
            "Every run in a suite shares one env config, one step budget and one "
            "matched set of agent hyperparameters; the only variables are the "
            "algorithm and the backend cell. Two caveats belong in any ranking: "
            "PPO is on-policy and therefore not replay-ratio comparable (judge it "
            "on score-vs-steps and score-vs-wall-clock, never on gradient steps), "
            "and MPO pays ~20x per gradient step for its 20 action samples. Ranks "
            "should come from the back-half mean of a curve, not a peak epoch."
        ),
    ]

    for suite in _ordered(grid.keys(), list(SUITE_INFO)):
        cells = grid[suite]
        heading, blurb = SUITE_INFO.get(suite, (suite, ""))
        blocks += [wr.H1(heading)]
        if blurb:
            blocks.append(wr.P(blurb))

        ordered_cells = _ordered(cells.keys(), list(CELL_INFO))

        # One panel grid per cell: all agents of that cell overlaid.
        for cell in ordered_cells:
            agents = sorted(cells[cell])
            blocks += [
                wr.H2(f"{cell} — {CELL_INFO.get(cell, 'backend cell')}"),
                wr.P(f"{len(agents)} agents: {', '.join(agents)}."),
                wr.PanelGrid(
                    runsets=[
                        _runset(wr, entity, project, cell, suite, cell=cell)
                    ],
                    panels=_curve_panels(wr, suite),
                ),
            ]

        # Cross-cell comparison, for the agents that ran on more than one cell:
        # one runset per cell so the same agent's curves are directly overlaid.
        multi = sorted(
            {a for cell in ordered_cells for a in cells[cell]
             if sum(a in cells[c] for c in ordered_cells) > 1}
        )
        if len(ordered_cells) > 1 and multi:
            blocks += [
                wr.H2("Same agent, different device placement"),
                wr.P(
                    "The backend A/B: identical agent, identical budget, identical "
                    "task — only where the physics and the learner execute changes. "
                    "The score curves should land in the same place (they are the "
                    "same algorithm); the wall-clock and throughput panels are where "
                    "the cells actually differ. Agents shown: "
                    f"{', '.join(multi)}."
                ),
            ]
            for agent in multi:
                blocks += [
                    wr.H3(agent),
                    wr.PanelGrid(
                        runsets=[
                            _runset(wr, entity, project, cell, suite,
                                    cell=cell, agent=agent)
                            for cell in ordered_cells if agent in cells[cell]
                        ],
                        panels=_curve_panels(wr, suite),
                    ),
                ]

    return wr.Report(
        project=project,
        # "" (not None) means "my default entity" — the pydantic model rejects
        # None here.
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
    parser.add_argument("--project", default=DEFAULT_PROJECT,
                        help=f"W&B project holding the runs (default: {DEFAULT_PROJECT})")
    parser.add_argument("--entity", default=None,
                        help="W&B entity; defaults to your default entity")
    parser.add_argument("--tag", default=DEFAULT_TAG,
                        help=f"only include runs carrying this tag (default: {DEFAULT_TAG})")
    parser.add_argument("--title", default="Roxie v1 — release benchmark")
    parser.add_argument(
        "--description",
        default=(
            "Every agent in roxie, on a simple continuous-control task and on a "
            "humanoid motion-tracking task, across the CPU/GPU backend cells each "
            "task can express."
        ),
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="build the report and print its structure without publishing")
    args = parser.parse_args()

    wr = _import_workspaces()

    grid, n_runs = discover(args.entity, args.project, args.tag)
    if not grid:
        sys.exit(
            f"no runs tagged '{args.tag}' found in project '{args.project}'.\n"
            "Run scripts/run_release_benchmark.sh first (without --offline)."
        )

    print(f"found {n_runs} runs in {len(grid)} suite(s):")
    for suite, cells in grid.items():
        for cell, agents in cells.items():
            print(f"  {suite:<18} {cell:<13} {len(agents)} agents: {' '.join(sorted(agents))}")

    report = build(wr, grid, args.entity, args.project, args.title, args.description)

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
