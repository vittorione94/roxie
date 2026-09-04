"""Figures from run CSVs.

Two modes, because the benchmark asks two different questions:

``--path <run(s)>``     the per-run diagnostic sheet — score, losses, throughput,
                        gradient steps, and the wall-clock panels — with every
                        discovered run overlaid.
``--grid --path <root>``  the release figure: one cell per ENV, every agent's
                        eval curve inside it. Reads an ``outputs/release_v1``
                        tree laid out as ``<task>/<cell>/<agent>/<stamp>/log.csv``,
                        and writes ONE FIGURE PER SUITE — see ``CELL_SUITE``.

Both read ``log.csv``, which the CSV logger writes for every run whether or not
wandb was enabled — so the figure never depends on a network round-trip.
"""

import argparse
import math
import pandas as pd
import matplotlib.pyplot as plt
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


# Runs logged before the `train/` `test/` `sys/` rename carry bare metric names.
_LEGACY_COLUMNS = {
    "score": "train/score",
    "score/std": "train/score/std",
    "length": "train/length",
    "length/std": "train/length/std",
    "episodes/epoch": "train/episodes/epoch",
    "episodes/total": "train/episodes/total",
    "gradient_steps": "train/gradient_steps",
    "loss/actor": "train/loss/actor",
    "loss/critic": "train/loss/critic",
    "sps": "sys/sps",
    "time/total_s": "sys/time/total_s",
    "time/epoch_s": "sys/time/epoch_s",
}


def normalize_columns(df):
    """Rename legacy metric columns to the current scheme, in place-ish.

    Only fills a target that is not already present, so a run that somehow has
    both spellings keeps the current one. Prefix families (`reward/`, `noise/`,
    `gpu/`, `mem/`, agent diagnostics) are remapped by their first segment.
    """
    renames = {
        old: new for old, new in _LEGACY_COLUMNS.items()
        if old in df.columns and new not in df.columns
    }
    for col in df.columns:
        head = col.split("/", 1)[0]
        if head in ("reward", "noise", "mining", "td3", "ppo"):
            target = f"train/{col}"
        elif head in ("gpu", "mem"):
            target = f"sys/{col}"
        else:
            continue
        if target not in df.columns:
            renames[col] = target
    return df.rename(columns=renames) if renames else df


def load_runs(paths):
    """Load each discovered CSV into a (label, DataFrame) pair, skipping unreadable ones."""
    runs = []
    for label, csv_path in discover_runs(paths):
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: skipping '{csv_path}' (error reading CSV: {e})")
            continue
        df = normalize_columns(df)
        print(f"Loaded {len(df)} rows from {csv_path}")
        runs.append((label, df))
    return runs


# Line style per backend cell: within a suite the GPU and CPU curves of one
# agent should overlap, which is the reading the figure exists for.
CELL_STYLE = {
    "warp_gpu": ("-", 1.6),
    "mjx_gpu": ("-", 1.6),
    "envpool_cpu": ("--", 1.3),
    "mjx_cpu": ("--", 1.3),
}

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
}

# Named in each figure's subtitle.
SUITE_PHYSICS = {
    "playground": "mujoco_playground XMLs, MJX-tuned solver settings",
    "dm_control": "dm_control XMLs, native MuJoCo",
}


def split_suites(grid):
    """``[(suite, subgrid), ...]`` — the grid partitioned by task implementation.

    A cell the map does not know keeps its own name as its suite, so an
    unrecognised backend gets its own figure rather than being silently folded
    into someone else's physics.
    """
    suites: dict[str, dict] = {}
    for task, cells in grid.items():
        for (agent, cell), df in cells.items():
            suite = CELL_SUITE.get(cell, cell)
            suites.setdefault(suite, {}).setdefault(task, {})[(agent, cell)] = df
    return sorted(suites.items())


def suite_output(output: str, suite: str) -> str:
    """``release.pdf`` -> ``release-playground.pdf``."""
    path = Path(output)
    return str(path.with_name(f"{path.stem}-{suite}{path.suffix}"))


def discover_grid(root: Path):
    """Walk a release tree into {task: {(agent, cell): DataFrame}}.

    Expects ``<root>/<task>/<cell>/<agent>/<stamp>/log.csv``, which is the
    layout `experiments/dmc/bench/dmc.yaml` gives `hydra.run.dir`. When an arm
    was run more than once (a resume, a re-run), the LAST timestamp wins —
    the tree is sorted, and a later stamp is the later attempt.
    """
    grid: dict[str, dict[tuple[str, str], pd.DataFrame]] = {}
    for csv_path in sorted(root.rglob(LOG_NAME)):
        rel = csv_path.parent.relative_to(root).parts
        if len(rel) != 4:
            continue
        task, cell, agent, _stamp = rel
        try:
            df = normalize_columns(pd.read_csv(csv_path))
        except Exception as e:
            print(f"Warning: skipping '{csv_path}' ({e})")
            continue
        if "test/score" not in df.columns or not len(df):
            continue
        grid.setdefault(task, {})[(agent, cell)] = df
    return grid


def plot_grid(grid, output: str, metric: str = "test/score", suite: str | None = None):
    """The release figure for ONE suite: a panel per env, every agent inside it.

    Takes an already-discovered (and suite-filtered) grid rather than a root, so
    a 350-run tree is read from disk once and drawn once per suite.
    """
    tasks = sorted(grid)
    agents = sorted({agent for cells in grid.values() for agent, _ in cells})
    prop = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    # Colour is the agent and nothing else, so an arm keeps its colour across
    # every panel.
    color = {a: (prop[i % len(prop)] if prop else None) for i, a in enumerate(agents)}

    ncols = min(5, len(tasks))
    nrows = math.ceil(len(tasks) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.2 * nrows),
                            squeeze=False)

    cells_seen = set()
    for index, (ax, task) in enumerate(zip(axs.flat, tasks)):
        for (agent, cell), df in sorted(grid[task].items()):
            style, width = CELL_STYLE.get(cell, ("-", 1.4))
            cells_seen.add(cell)
            ax.plot(df["steps"], df[metric], color=color[agent],
                    linestyle=style, linewidth=width, alpha=0.9)
        ax.set_title(task, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.tick_params(labelsize=8)
        # dm_control returns are bounded by construction, so pinning the axis
        # keeps panels comparable instead of each auto-scaling to its best arm.
        ax.set_ylim(0, 1000)
        if index % ncols == 0:
            ax.set_ylabel(metric, fontsize=9)
        if index >= len(tasks) - ncols:
            ax.set_xlabel("environment steps", fontsize=9)
    for ax in axs.flat[len(tasks):]:
        ax.axis("off")

    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=color[a], label=a) for a in agents]
    handles += [
        Line2D([], [], color="0.35", linestyle=CELL_STYLE.get(c, ("-", 1.4))[0],
               label=c)
        for c in sorted(cells_seen)
    ]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(handles), 10), frameon=False, fontsize=9)
    title = f"{metric} vs environment steps — {len(tasks)} tasks, {len(agents)} agents"
    physics = None
    if suite:
        title = f"{suite} — {title}"
        physics = SUITE_PHYSICS.get(suite)
    # Header lines are offset a fixed number of INCHES from the top: the grid is
    # 1 row for a pilot and 5 for the full release, and a fractional offset that
    # looks right at one height collides with the panels at the other.
    height = fig.get_figheight()
    fig.suptitle(title, fontsize=14, fontweight="bold", va="top",
                 y=1 - 0.15 / height)
    if physics:
        fig.text(0.5, 1 - 0.46 / height, physics, ha="center", va="top",
                 fontsize=9, color="0.35")
    # The legend is a figure-level artist, which tight_layout ignores.
    bottom = 0.9 / height
    fig.tight_layout(rect=[0, bottom, 1, 1 - 0.72 / height])

    try:
        fig.savefig(output, format="pdf", bbox_inches="tight")
        print(f"Wrote {len(tasks)}x{len(agents)} grid to: {output}")
    except Exception as e:
        print(f"Error saving PDF: {e}")


def main():
    parser = argparse.ArgumentParser(description="Generate learning curves from one or more run log CSVs.")
    parser.add_argument("--csv-path", "--path", dest="csv_path", required=True, nargs="+", type=str,
                        help="Path(s) to log.csv files and/or directories, which are searched "
                             "recursively for log.csv. All discovered runs are overlaid.")
    parser.add_argument("--grid", action="store_true",
                        help="Release figure: one panel per env, every agent inside it. "
                             "Takes a single release tree (outputs/release_v1) as --path. "
                             "Writes ONE FIGURE PER SUITE (playground / dm_control), "
                             "suffixed onto --output, because the two do not share a "
                             "physics model on every task.")
    parser.add_argument("--metric", default="test/score",
                        help="Metric plotted in --grid mode (default: test/score)")
    parser.add_argument("--output", type=str, default="learning_curves.pdf",
                        help="Path to save the output PDF (default: learning_curves.pdf)")
    args = parser.parse_args()

    if args.grid:
        if len(args.csv_path) != 1:
            print("Error: --grid takes exactly one --path (a release tree root).")
            return
        root = Path(args.csv_path[0])
        grid = discover_grid(root)
        if not grid:
            print(f"Error: no runs found under '{root}' "
                  f"(expected <task>/<cell>/<agent>/<stamp>/{LOG_NAME}).")
            return
        suites = split_suites(grid)
        for suite, subgrid in suites:
            # The suffix exists only to stop two suites overwriting each other.
            out = suite_output(args.output, suite) if len(suites) > 1 else args.output
            plot_grid(subgrid, out, args.metric, suite)
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
