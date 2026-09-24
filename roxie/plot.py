"""Figures from run CSVs.

Two modes, because the benchmark asks two different questions:

`--path <run(s)>`
    the per-run diagnostic sheet — score, losses, throughput, gradient steps,
    and the wall-clock panels — with every discovered run overlaid.

`--grid --path <root>`
    the release figures. Reads an `outputs/release_v1` tree laid out as
    `<task>/<cell>/<agent>/<stamp>/log.csv` and writes, for each (suite,
    device) group, a panel per task on the step axis above the same panels on
    the wall-clock axis; plus one cross-device wall-clock summary and the table
    view. Light and dark variants of every figure, for a `<picture>` element.

Both read `log.csv`, which the CSV logger writes for every run whether or not
wandb was enabled — so the figure never depends on a network round-trip.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator
from pathlib import Path


LOG_NAME = "log.csv"


def run_label(csv_path: Path, root: Path | None = None) -> str:
    """Derive a concise legend label for a run from its CSV path.

    When the CSV was discovered under a search ``root``, the label is the run
    directory's path relative to that root (e.g. ``ppo/2026-08-14_19-01-59``),
    which keeps sibling runs of a sweep distinguishable. Otherwise it falls back
    to the parent directory name for generic ``log.csv`` files, or the file stem.
    """
    if root is not None:
        rel = csv_path.parent.relative_to(root)
        if rel != Path("."):
            return rel.as_posix()
        return root.name or str(root)
    if csv_path.stem == "log":
        return csv_path.parent.name or csv_path.stem
    return csv_path.stem


def discover_runs(paths):
    """Expand the given files/directories into (label, csv_path) pairs.

    Directories are searched recursively for ``log.csv`` files; plain files are
    used as-is. Duplicate CSVs (e.g. a directory listed twice, or nested roots)
    are collapsed, and colliding labels are disambiguated with the search root.
    """
    found = []
    seen = set()
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"Warning: skipping '{path}' (does not exist)")
            continue
        if path.is_dir():
            matches = sorted(path.rglob(LOG_NAME))
            if not matches:
                print(f"Warning: no {LOG_NAME} found under '{path}'")
            for csv_path in matches:
                if csv_path in seen:
                    continue
                seen.add(csv_path)
                found.append((run_label(csv_path, path), path, csv_path))
        else:
            if path in seen:
                continue
            seen.add(path)
            found.append((run_label(path), path, path))

    # Disambiguate labels that appear under more than one search root.
    counts = {}
    for label, _, _ in found:
        counts[label] = counts.get(label, 0) + 1
    runs = []
    for label, root, csv_path in found:
        if counts[label] > 1 and root.is_dir():
            label = f"{root.name}/{label}" if root.name else str(csv_path.parent)
        runs.append((label, csv_path))
    return runs


def wallclock_unit(runs):
    """Pick a wall-clock unit ('s' / 'min' / 'h') that suits the longest run.

    Returns ``(name, divisor)``; runs without a ``sys/time/total_s`` column are
    ignored, and the fallback is seconds when no run logged wall-clock time.
    """
    longest = 0.0
    for _, df in runs:
        if 'sys/time/total_s' in df.columns and len(df):
            longest = max(longest, float(df['sys/time/total_s'].max()))
    if longest >= 2 * 3600:
        return 'hours', 3600.0
    if longest >= 120:
        return 'minutes', 60.0
    return 'seconds', 1.0


def load_runs(paths):
    """Load each discovered CSV into a (label, DataFrame) pair, skipping unreadable ones."""
    runs = []
    for label, csv_path in discover_runs(paths):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: skipping '{csv_path}' (error reading CSV: {e})")
            continue
        print(f"Loaded {len(df)} rows from {csv_path}")
        runs.append((label, df))
    return runs


# Which task IMPLEMENTATION a cell steps — the axis the figure must not
# collapse. The playground cells share one set of XMLs, so overlaying them IS a
# parity check; EnvPool steps dm_control's own C++ physics off a different XML,
# and the two disagree on integrator (9 of 25 tasks) and sim dt (8). So a
# cross-suite score gap tangles physics with algorithm, and each suite gets its
# own figure.
CELL_SUITE = {
    "warp_gpu": "playground",
    "mjx_gpu": "playground",
    "mjx_cpu": "playground",
    "envpool_cpu": "dm_control",
    "mjbatch_cpu": "mjbatch",
}

# Named in each figure's subtitle.
SUITE_PHYSICS = {
    "playground": "mujoco_playground XMLs, MJX-tuned solver settings",
    "dm_control": "dm_control XMLs, native MuJoCo",
    "mjbatch": "dm_control XMLs, native MuJoCo, batched rollout",
}

DEVICE_LABEL = {"cpu": "CPU", "gpu": "GPU"}

# Named in the speed figure's legend, where the series is the backend cell.
CELL_LABEL = {
    "envpool_cpu": "EnvPool (CPU)",
    "mjbatch_cpu": "mjbatch (CPU)",
    "warp_gpu": "mujoco_warp (GPU)",
    "mjx_gpu": "MJX (GPU)",
    "mjx_cpu": "MJX (CPU)",
}

# The rate the speed figure reports, so a run that has not reached the budget is
# still comparable rather than silently reading as cheap.
RATE_STEPS = 1e8

THEMES = {
    "light": {
        "surface": "#fcfcfb", "primary": "#0b0b0b", "secondary": "#52514e",
        "muted": "#87867f", "grid": "#e6e5e1", "axis": "#cbcac4",
    },
    "dark": {
        "surface": "#1a1a19", "primary": "#ffffff", "secondary": "#c3c2b7",
        "muted": "#8f8e86", "grid": "#33322f", "axis": "#4c4b47",
    },
}

# Colour identifies the agent and so does the dash: the second encoding is what
# keeps the dark-mode green/yellow pair (all-pairs CVD ΔE 6.9, inside the 6–8
# band that is legal only with secondary encoding) apart, and what survives a
# greyscale print.
AGENT_COLORS = {
    "light": {"ddpg": "#2a78d6", "td3": "#eda100", "d4pg": "#e87ba4", "td4": "#008300",
              "sac": "#eb6834", "mpo": "#1baf7a", "ppo": "#4a3aa7"},
    "dark": {"ddpg": "#3987e5", "td3": "#c98500", "d4pg": "#d55181", "td4": "#008300",
             "sac": "#d95926", "mpo": "#199e70", "ppo": "#9085e9"},
}

AGENT_DASH = {
    "ddpg": (), "td3": (5.0, 1.8), "d4pg": (1.5, 1.5), "td4": (6.0, 1.8, 1.5, 1.8),
    "sac": (), "mpo": (5.0, 1.8), "ppo": (1.5, 1.5),
}

# Seven curves cannot share one panel: the palette clears its all-pairs CVD and
# normal-vision floors at four series and no further, and no re-ordering of the
# eight documented hues changes that. The split is by policy class, so a panel
# is also a like-for-like comparison.
FAMILIES = [
    ("deterministic policy", ["ddpg", "td3", "d4pg", "td4"]),
    ("stochastic policy", ["sac", "mpo", "ppo"]),
]

# The caption over each half of a release figure, in row order.
BLOCK_CAPTIONS = {
    "steps": "sample efficiency — return against environment steps",
    "time": "compute efficiency — return against wall-clock time",
}

SCORE_CEILING = 1000.0
SMOOTH_WINDOW = 9
FINAL_EVALS = 10


def cell_device(cell: str) -> str:
    """'gpu' or 'cpu' — which device the cell puts the physics and the learner on."""
    return "gpu" if cell.endswith("_gpu") else "cpu"


def split_groups(grid):
    """``[((suite, device), subgrid), ...]`` — the grid partitioned into figures.

    Suite is the task implementation and device is where the run executed. Both
    have to split: overlaying two physics models hides an algorithm gap behind a
    physics one, and overlaying two devices puts fourteen curves in a panel.
    """
    groups: dict[tuple[str, str], dict] = {}
    for task, cells in grid.items():
        for (agent, cell), df in cells.items():
            key = (CELL_SUITE.get(cell, cell), cell_device(cell))
            groups.setdefault(key, {}).setdefault(task, {})[(agent, cell)] = df
    return sorted(groups.items())


def group_output(stem: str, suite: str, device: str, theme: str) -> str:
    """``images/release`` -> ``images/release-dm_control-cpu-dark.png``."""
    path = Path(stem)
    parts = [path.name, suite, device] + ([theme] if theme == "dark" else [])
    return str(path.with_name("-".join(parts) + ".png"))


def discover_grid(root: Path):
    """Walk a release tree into {task: {(agent, cell): DataFrame}}.

    Expects ``<root>/<task>/<cell>/<agent>/<stamp>/log.csv``, which is the
    layout `experiments/dmc/bench/dmc.yaml` gives `hydra.run.dir`. When an arm
    was run more than once (a resume, a re-run), the LAST timestamp wins — the
    tree is sorted, and a later stamp is the later attempt — and the dropped
    stamps are named on stdout, because a re-run at a changed config is exactly
    what that rule silently resolves.
    """
    grid: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    stamps: dict[tuple[str, str, str], list[str]] = {}
    for csv_path in sorted(root.rglob(LOG_NAME)):
        rel = csv_path.parent.relative_to(root).parts
        if len(rel) != 4:
            continue
        task, cell, agent, stamp = rel
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: skipping '{csv_path}' ({e})")
            continue
        if "test/score" not in df.columns or not len(df):
            continue
        grid.setdefault(task, {})[(agent, cell)] = df
        stamps.setdefault((task, cell, agent), []).append(stamp)
    for (task, cell, agent), seen in sorted(stamps.items()):
        if len(seen) > 1:
            print(f"Warning: {task}/{cell}/{agent} has {len(seen)} runs — "
                  f"using {seen[-1]}, ignoring {', '.join(seen[:-1])}")
    return grid


def final_score(df) -> float:
    """The mean of the last `FINAL_EVALS` evaluations, not the last one alone."""
    return float(df["test/score"].tail(FINAL_EVALS).mean())


def _axis_series(df, axis: str):
    """``(x, label, limit)`` for one run on either the step or the wall-clock axis."""
    if axis == "steps":
        return df["steps"] / 1e6, "environment steps (millions)"
    return df["sys/time/total_s"] / 3600.0, "wall-clock (hours)"


def _style_axes(ax, tokens):
    ax.set_facecolor(tokens["surface"])
    ax.grid(True, linewidth=0.6, color=tokens["grid"])
    ax.set_axisbelow(True)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(tokens["axis"])
        spine.set_linewidth(0.8)
    ax.tick_params(labelsize=7.5, colors=tokens["secondary"], length=3,
                   width=0.8, color=tokens["axis"])


def _place_labels(ax, entries, tokens):
    """Direct-label each curve at its end, pushed apart so the text never overlaps.

    Entries are `(y, agent, colour, x_fraction)`. The label sits in the gutter
    beyond the axes — x in axes fractions, y in data — so reserving room for it
    does not stretch the x-axis past the last step. A leader line runs back to
    the curve, because agents that finish within a few points of each other get
    pushed far enough apart that the nearest label is not theirs.
    """
    gap = SCORE_CEILING * 0.062
    placed = []
    for y, agent, color, xf in sorted(entries, key=lambda e: e[0]):
        shifted = max(y, placed[-1][1] + gap) if placed else y
        placed.append((y, shifted, agent, color, xf))
    overflow = placed[-1][1] - SCORE_CEILING if placed else 0.0
    if overflow > 0:
        placed = [(y, shifted - overflow, a, c, xf) for y, shifted, a, c, xf in placed]
    gutter = ax.get_yaxis_transform()
    for y, shifted, agent, color, xf in placed:
        if abs(shifted - y) > gap * 0.3 or xf < 0.95:
            ax.plot([xf, 1.022], [y, shifted], transform=gutter,
                    color=tokens["muted"], linewidth=0.55, alpha=0.5,
                    clip_on=False, zorder=4)
        ax.plot([1.028], [shifted], marker="o", markersize=3.2, color=color,
                transform=gutter, clip_on=False, zorder=5)
        ax.text(1.062, shifted, agent.upper(), fontsize=6.8, va="center", ha="left",
                transform=gutter, color=tokens["secondary"], clip_on=False)


def plot_release(grid, output: str, suite: str, device: str, theme: str):
    """One release figure: a column per task, the step axis over the clock axis.

    Two captioned halves separated by a rule — the two policy families on
    environment steps, then the same two on wall-clock. A column shares its
    x-limit down both rows of a half, so a slow agent's curve is visibly shorter
    than a fast one's rather than being rescaled to fill the panel.
    """
    tokens = THEMES[theme]
    colors = AGENT_COLORS[theme]
    tasks = sorted(grid)
    present = {agent for cells in grid.values() for agent, _ in cells}
    families = [(family, agents) for family, agents in FAMILIES
                if present.intersection(agents)]
    rows = [(family, agents, axis) for axis in ("steps", "time")
            for family, agents in families]

    half = len(rows) // 2
    fig = plt.figure(figsize=(3.7 * len(tasks), 2.55 * len(rows) + 0.75))
    # A spacer row rather than a uniform hspace: the gap between the halves has
    # to be wider than the gap inside one, and hspace cannot vary per row.
    ratios = [1.0] * half + [0.24] + [1.0] * half
    spec = fig.add_gridspec(len(rows) + 1, len(tasks), height_ratios=ratios)
    axs = [[fig.add_subplot(spec[r if r < half else r + 1, c])
            for c in range(len(tasks))] for r in range(len(rows))]
    fig.patch.set_facecolor(tokens["surface"])

    xmax = {}
    for task in tasks:
        for axis in ("steps", "time"):
            xmax[(task, axis)] = max(
                float(_axis_series(df, axis)[0].max()) for df in grid[task].values())

    drawn = set()
    for r, (family, agents, axis) in enumerate(rows):
        for c, task in enumerate(tasks):
            ax = axs[r][c]
            _style_axes(ax, tokens)
            limit = xmax[(task, axis)]
            entries = []
            for (agent, cell), df in sorted(grid[task].items()):
                if agent not in agents:
                    continue
                drawn.add(agent)
                x, xlabel = _axis_series(df, axis)
                raw = df["test/score"]
                # The eval is ten episodes, so a single point swings; the band
                # behind the line is what shows that rather than hiding it.
                line = raw.rolling(SMOOTH_WINDOW, center=True, min_periods=1).mean()
                style = dict(color=colors[agent], linewidth=1.9,
                             dashes=AGENT_DASH[agent] or (1, 0),
                             solid_capstyle="round", dash_capstyle="round")
                ax.plot(x, raw, color=colors[agent], linewidth=1.0, alpha=0.13)
                ax.plot(x, line, **style, zorder=4)
                entries.append((float(line.iloc[-1]), agent, colors[agent],
                                float(x.iloc[-1]) / (limit * 1.015)))
            ax.set_xlim(0, limit * 1.015)
            ax.set_ylim(0, SCORE_CEILING)
            ax.set_yticks([0, 250, 500, 750, 1000])
            ax.xaxis.set_major_locator(
                MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]))
            _place_labels(ax, entries, tokens)
            if r == 0:
                ax.set_title(task, fontsize=10.5, color=tokens["primary"],
                             fontweight="bold", pad=8)
            if c == 0:
                ax.set_ylabel(f"{family}\nepisode return", fontsize=8.5,
                              color=tokens["secondary"])
            if r in (half - 1, len(rows) - 1):
                ax.set_xlabel(xlabel, fontsize=8.5, color=tokens["secondary"])

    handles = [Line2D([], [], color=colors[a], linewidth=1.9,
                      dashes=AGENT_DASH[a] or (1, 0), label=a.upper())
               for _, agents in families for a in agents if a in drawn]
    legend = fig.legend(handles=handles, loc="lower center", ncol=len(handles),
                        frameon=False, fontsize=8.5,
                        bbox_to_anchor=(0.5, 0.004))
    for text in legend.get_texts():
        text.set_color(tokens["secondary"])

    height = fig.get_figheight()
    fig.tight_layout(rect=[0, 0.05, 1, 1 - 0.98 / height])
    # The end labels live outside the axes, which tight_layout cannot see.
    fig.subplots_adjust(wspace=0.36)

    # Offsets in inches, not figure fractions: the grid is two rows per half at
    # any task count, but the figure's height is not.
    left = axs[0][0].get_position().x0
    right = axs[0][-1].get_position().x1
    captions = []
    for index, axis in enumerate(("steps", "time")):
        # The top half's caption clears the task titles; the bottom half has none.
        clearance = 0.34 if index == 0 else 0.10
        y = axs[index * half][0].get_position().y1 + clearance / height
        captions.append(y)
        fig.text(left, y, BLOCK_CAPTIONS[axis].upper(), fontsize=8.5,
                 fontweight="bold", color=tokens["secondary"], va="bottom")
    # Centred in the band that is actually empty, which starts under the upper
    # half's x-label — an offset from the axes box lands too high by its height.
    fig.canvas.draw()
    label = axs[half - 1][0].xaxis.get_label().get_window_extent(
        fig.canvas.get_renderer()).transformed(fig.transFigure.inverted())
    rule_y = (label.y0 + captions[1] + 0.09 / height) / 2
    fig.add_artist(Line2D([left, right], [rule_y, rule_y], color=tokens["axis"],
                          linewidth=0.9, zorder=0))

    # Stacked up from the first caption rather than down from the figure top:
    # the panel grid is two rows for a one-family group and four for two, and a
    # header pinned to the top leaves a hole above the grid at the short height.
    cells_seen = {cell for c in grid.values() for _, cell in c}
    backend = (CELL_LABEL.get(next(iter(cells_seen)), suite)
               if len(cells_seen) == 1 else DEVICE_LABEL[device])
    plural = "" if len(drawn) == 1 else "s"
    fig.text(0.5, captions[0] + 0.30 / height,
             f"{SUITE_PHYSICS.get(suite, suite)} · curves are a {SMOOTH_WINDOW}-eval "
             "rolling mean over the raw band",
             ha="center", va="bottom", fontsize=8.5, color=tokens["muted"])
    fig.text(0.5, captions[0] + 0.52 / height,
             f"{backend} — episode return on {len(tasks)} tasks, "
             f"{len(drawn)} agent{plural}", ha="center", va="bottom", fontsize=15,
             fontweight="bold", color=tokens["primary"])
    fig.savefig(output, dpi=200, facecolor=tokens["surface"])
    plt.close(fig)
    print(f"Wrote {output}")


def _rounded_barh(ax, y, width, height, color, radius_frac=0.45):
    """A horizontal bar with its value end rounded and its baseline end square."""
    r = min(height * radius_frac, abs(width))
    verts = [(0, y - height / 2), (width - r, y - height / 2)]
    codes = [MplPath.MOVETO, MplPath.LINETO]
    verts += [(width, y - height / 2), (width, y), (width, y + height / 2),
              (width - r, y + height / 2)]
    codes += [MplPath.CURVE3, MplPath.CURVE3, MplPath.CURVE3, MplPath.CURVE3]
    verts += [(0, y + height / 2), (0, y - height / 2)]
    codes += [MplPath.LINETO, MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), facecolor=color, edgecolor="none",
                           zorder=3))


def plot_speed(grid, output: str, theme: str):
    """Wall-clock per 100M environment steps, per agent, backend against backend.

    A rate rather than a total, because a run that has not reached the step
    budget yet would otherwise read as the cheapest arm on the chart. Every
    completed arm here shares one budget, so its bar is the total over five.
    The rate does carry each run's one-off compile and warmup, which a short run
    amortises over fewer steps — that counts against the short run, not for it.
    """
    tokens = THEMES[theme]
    palette = AGENT_COLORS[theme]
    budget = max(float(df["steps"].iloc[-1])
                 for c in grid.values() for df in c.values())
    rates: dict[tuple[str, str], list[float]] = {}
    partial = set()
    for cells in grid.values():
        for (agent, cell), df in cells.items():
            steps = float(df["steps"].iloc[-1])
            if steps <= 0:
                continue
            if steps < budget * 0.99:
                partial.add((agent, cell))
            seconds = float(df["sys/time/total_s"].iloc[-1])
            rates.setdefault((agent, cell), []).append(
                seconds / 3600.0 / (steps / RATE_STEPS))

    cells = sorted({cell for _, cell in rates})
    accent = dict(zip(cells, (palette["ddpg"], palette["sac"], palette["mpo"])))
    mean = {key: sum(v) / len(v) for key, v in rates.items()}
    agents = [a for _, family in FAMILIES for a in family
              if any((a, c) in rates for c in cells)]
    agents.sort(key=lambda a: -max(mean[(a, c)] for c in cells if (a, c) in rates))

    fig, ax = plt.subplots(figsize=(9.6, 0.62 * len(agents) + 2.0))
    fig.patch.set_facecolor(tokens["surface"])
    _style_axes(ax, tokens)
    ax.grid(False, axis="y")

    # A cell keeps its slot whether or not it ran this agent, so the groups stay
    # aligned down the chart instead of recentring per row.
    bar_h = min(0.27, 0.82 / len(cells))
    pitch = bar_h + 0.05
    span = max(v for values in rates.values() for v in values)
    for i, agent in enumerate(agents):
        for k, cell in enumerate(cells):
            if (agent, cell) not in rates:
                continue
            values = rates[(agent, cell)]
            y = i + ((len(cells) - 1) / 2 - k) * pitch
            _rounded_barh(ax, y, mean[(agent, cell)], bar_h, accent[cell])
            # Ticks rather than a bar-length whisker: a line laid along the bar
            # reads as a notch cut out of it.
            for edge in (min(values), max(values)) if len(values) > 1 else ():
                ax.plot([edge, edge], [y - bar_h * 0.72, y + bar_h * 0.72],
                        color=tokens["muted"], linewidth=1.2, zorder=4)
            mark = "*" if (agent, cell) in partial else ""
            ax.text(max(values) + span * 0.016, y,
                    f"{mean[(agent, cell)]:.2f} h{mark}", fontsize=8, va="center",
                    ha="left", color=tokens["secondary"], zorder=5)
        ran = [mean[(agent, c)] for c in cells if (agent, c) in rates]
        if len(ran) > 1:
            ax.text(1.0, i, f"{max(ran) / min(ran):.1f}×",
                    transform=ax.get_yaxis_transform(), fontsize=8.5,
                    va="center", ha="right", color=tokens["primary"],
                    fontweight="bold")

    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([a.upper() for a in agents], fontsize=9,
                       color=tokens["primary"])
    ax.set_ylim(-0.7, len(agents) - 0.3)
    ax.invert_yaxis()
    ax.set_xlim(0, span * 1.18)
    ax.set_xlabel("wall-clock hours per 100M environment steps", fontsize=9,
                  color=tokens["secondary"])
    ax.tick_params(axis="y", length=0)

    handles = [Patch(facecolor=accent[c], label=CELL_LABEL.get(c, c)) for c in cells]
    legend = ax.legend(handles=handles, loc="lower right", frameon=False,
                       fontsize=9, ncol=len(cells), borderaxespad=0,
                       bbox_to_anchor=(1.0, 1.004))
    for text in legend.get_texts():
        text.set_color(tokens["secondary"])

    ax.set_title("Cost of an environment step", fontsize=13.5, fontweight="bold",
                 color=tokens["primary"], loc="left", pad=30)
    note = ("mean over tasks, ticks span them · bold is slowest backend ÷ fastest"
            + (" · * has not reached the step budget, so its one-off compile is "
               "spread over fewer steps" if partial else ""))
    ax.text(0, 1.055, note, transform=ax.transAxes, fontsize=8.5,
            color=tokens["muted"])
    fig.tight_layout(rect=[0, 0, 0.965, 1])
    fig.savefig(output, dpi=200, facecolor=tokens["surface"])
    plt.close(fig)
    print(f"Wrote {output}")


def write_table(grid, output: str):
    """The table view the low-contrast palette slots owe the reader."""
    tasks = sorted(grid)
    agents = [a for _, family in FAMILIES for a in family]
    cells = sorted({cell for c in grid.values() for _, cell in c})
    budget = max(float(df["steps"].iloc[-1])
                 for c in grid.values() for df in c.values())
    lines = ["# Release benchmark — final return and wall-clock", "",
             f"Final return is the mean of the last {FINAL_EVALS} evaluations. "
             f"The budget is {budget / 1e6:.0f}M environment steps; an arm that "
             "stopped short of it is marked with the steps it reached.", ""]
    for cell in cells:
        lines += [f"## {cell}", "",
                  "| agent | " + " | ".join(f"{t} return" for t in tasks)
                  + " | " + " | ".join(f"{t} hours" for t in tasks) + " |",
                  "|---|" + "---|" * (2 * len(tasks))]
        for agent in agents:
            runs = [grid[t].get((agent, cell)) for t in tasks]
            if all(df is None for df in runs):
                continue
            scores = []
            for df in runs:
                if df is None:
                    scores.append("—")
                    continue
                reached = float(df["steps"].iloc[-1])
                short = "" if reached >= budget * 0.99 else f" @{reached / 1e6:.0f}M"
                scores.append(f"{final_score(df):.0f}{short}")
            times = ["—" if df is None
                     else f"{df['sys/time/total_s'].iloc[-1] / 3600:.1f}" for df in runs]
            lines.append(f"| {agent.upper()} | " + " | ".join(scores) + " | "
                         + " | ".join(times) + " |")
        lines.append("")
    Path(output).write_text("\n".join(lines))
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description="Generate learning curves from one or more run log CSVs.")
    parser.add_argument("--csv-path", "--path", dest="csv_path", required=True, nargs="+", type=str,
                        help="Path(s) to log.csv files and/or directories, which are searched "
                             "recursively for log.csv. All discovered runs are overlaid.")
    parser.add_argument("--grid", action="store_true",
                        help="Release figures from the release tree(s) given as "
                             "--path. Several roots merge into one grid, which is how "
                             "a backend benchmarked in its own tree joins the "
                             "comparison. Writes one figure per (suite, device) group, "
                             "a backend speed summary, and the score table; light and "
                             "dark variants of each. --output is a path STEM here, not "
                             "a filename.")
    parser.add_argument("--output", type=str, default="learning_curves.pdf",
                        help="Output PDF, or in --grid mode the path stem the PNGs "
                             "are suffixed onto (e.g. images/release)")
    args = parser.parse_args()

    if args.grid:
        grid: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
        for raw in args.csv_path:
            for task, cells in discover_grid(Path(raw)).items():
                grid.setdefault(task, {}).update(cells)
        if not grid:
            print(f"Error: no runs found under {', '.join(args.csv_path)} "
                  f"(expected <task>/<cell>/<agent>/<stamp>/{LOG_NAME}).")
            return
        stem = Path(args.output)
        stem.parent.mkdir(parents=True, exist_ok=True)
        for theme in ("light", "dark"):
            for (suite, device), subgrid in split_groups(grid):
                plot_release(subgrid, group_output(stem, suite, device, theme),
                             suite, device, theme)
            suffix = "-speed-dark.png" if theme == "dark" else "-speed.png"
            plot_speed(grid, str(stem) + suffix, theme)
        write_table(grid, str(stem) + "-scores.md")
        return

    runs = load_runs(args.csv_path)
    if not runs:
        print("Error: no valid CSV files to plot.")
        return

    multi = len(runs) > 1
    prop_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    colors = [prop_colors[i % len(prop_colors)] for i in range(len(runs))] if prop_colors else [None] * len(runs)

    time_name, time_div = wallclock_unit(runs)

    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    if multi:
        fig.suptitle(f"Training Metrics: {len(runs)} runs", fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f"Training Metrics: {runs[0][0]}", fontsize=16, fontweight='bold')

    for (label, df), color in zip(runs, colors):
        prefix = f"{label}: " if multi else ""

        if 'train/score' in df.columns and 'test/score' in df.columns:
            axs[0, 0].plot(df['steps'], df['train/score'], label=f'{prefix}Train Score',
                           color=color, alpha=0.8, linewidth=2)
            axs[0, 0].plot(df['steps'], df['test/score'], label=f'{prefix}Test Score',
                           color=color, alpha=0.8, linewidth=2, linestyle='--')

        if 'train/length' in df.columns and 'test/length' in df.columns:
            axs[0, 1].plot(df['steps'], df['train/length'], label=f'{prefix}Train Length',
                           color=color, alpha=0.8, linewidth=2)
            axs[0, 1].plot(df['steps'], df['test/length'], label=f'{prefix}Test Length',
                           color=color, alpha=0.8, linewidth=2, linestyle='--')

        if 'train/loss/actor' in df.columns:
            axs[1, 0].plot(df['steps'], df['train/loss/actor'], label=f'{prefix}Actor Loss',
                           color=color if multi else 'green', alpha=0.8, linewidth=2)

        if 'train/loss/critic' in df.columns:
            axs[1, 1].plot(df['steps'], df['train/loss/critic'], label=f'{prefix}Critic Loss',
                           color=color if multi else 'red', alpha=0.8, linewidth=2)

        if 'sys/sps' in df.columns:
            axs[2, 0].plot(df['steps'], df['sys/sps'], label=f'{prefix}SPS',
                           color=color if multi else 'purple', alpha=0.8, linewidth=2)

        if 'train/gradient_steps' in df.columns:
            axs[2, 1].plot(df['steps'], df['train/gradient_steps'], label=f'{prefix}Gradient Steps',
                           color=color if multi else 'orange', alpha=0.8, linewidth=2)

        # What ranks runs reaching the same score at different throughputs.
        if 'sys/time/total_s' not in df.columns:
            continue
        wall = df['sys/time/total_s'] / time_div

        if 'train/score' in df.columns and 'test/score' in df.columns:
            axs[3, 0].plot(wall, df['train/score'], label=f'{prefix}Train Score',
                           color=color, alpha=0.8, linewidth=2)
            axs[3, 0].plot(wall, df['test/score'], label=f'{prefix}Test Score',
                           color=color, alpha=0.8, linewidth=2, linestyle='--')

        if 'steps' in df.columns:
            axs[3, 1].plot(wall, df['steps'], label=f'{prefix}Steps',
                           color=color if multi else 'brown', alpha=0.8, linewidth=2)

    time_label = f'Wall-Clock Time ({time_name})'
    titles = [
        ('Score vs Steps', 'Environment Steps', 'Score'),
        ('Episode Length vs Steps', 'Environment Steps', 'Length'),
        ('Actor Loss vs Steps', 'Environment Steps', 'Loss'),
        ('Critic Loss vs Steps', 'Environment Steps', 'Loss'),
        ('Steps Per Second (SPS) vs Steps', 'Environment Steps', 'SPS'),
        ('Gradient Steps vs Env Steps', 'Environment Steps', 'Gradient Steps'),
        ('Score vs Wall-Clock Time', time_label, 'Score'),
        ('Env Steps vs Wall-Clock Time', time_label, 'Environment Steps'),
    ]
    for ax, (title, xlabel, ylabel) in zip(axs.flat, titles):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    try:
        plt.savefig(args.output, format='pdf', bbox_inches='tight')
        print(f"Successfully generated plots and saved to: {args.output}")
    except Exception as e:
        print(f"Error saving PDF: {e}")


if __name__ == "__main__":
    main()
