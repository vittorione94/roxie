import argparse
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


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate learning curves from one or more run log CSVs.")
    parser.add_argument("--csv-path", "--path", dest="csv_path", required=True, nargs="+", type=str,
                        help="Path(s) to log.csv files and/or directories, which are searched "
                             "recursively for log.csv. All discovered runs are overlaid.")
    parser.add_argument("--output", type=str, default="learning_curves.pdf",
                        help="Path to save the output PDF (default: learning_curves.pdf)")
    args = parser.parse_args()

    runs = load_runs(args.csv_path)
    if not runs:
        print("Error: no valid CSV files to plot.")
        return

    multi = len(runs) > 1
    # Consistent color per run so all of a run's curves share a hue.
    prop_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
    colors = [prop_colors[i % len(prop_colors)] for i in range(len(runs))] if prop_colors else [None] * len(runs)

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    if multi:
        fig.suptitle(f"Training Metrics: {len(runs)} runs", fontsize=16, fontweight='bold')
    else:
        fig.suptitle(f"Training Metrics: {runs[0][0]}", fontsize=16, fontweight='bold')

    for (label, df), color in zip(runs, colors):
        prefix = f"{label}: " if multi else ""

        # Plot 1: Scores vs Steps
        if 'score' in df.columns and 'test/score' in df.columns:
            axs[0, 0].plot(df['steps'], df['score'], label=f'{prefix}Train Score',
                           color=color, alpha=0.8, linewidth=2)
            axs[0, 0].plot(df['steps'], df['test/score'], label=f'{prefix}Test Score',
                           color=color, alpha=0.8, linewidth=2, linestyle='--')

        # Plot 2: Episode Length vs Steps
        if 'length' in df.columns and 'test/length' in df.columns:
            axs[0, 1].plot(df['steps'], df['length'], label=f'{prefix}Train Length',
                           color=color, alpha=0.8, linewidth=2)
            axs[0, 1].plot(df['steps'], df['test/length'], label=f'{prefix}Test Length',
                           color=color, alpha=0.8, linewidth=2, linestyle='--')

        # Plot 3: Actor Loss vs Steps
        if 'loss/actor' in df.columns:
            axs[1, 0].plot(df['steps'], df['loss/actor'], label=f'{prefix}Actor Loss',
                           color=color if multi else 'green', alpha=0.8, linewidth=2)

        # Plot 4: Critic Loss vs Steps
        if 'loss/critic' in df.columns:
            axs[1, 1].plot(df['steps'], df['loss/critic'], label=f'{prefix}Critic Loss',
                           color=color if multi else 'red', alpha=0.8, linewidth=2)

        # Plot 5: SPS vs Steps
        if 'sps' in df.columns:
            axs[2, 0].plot(df['steps'], df['sps'], label=f'{prefix}SPS',
                           color=color if multi else 'purple', alpha=0.8, linewidth=2)

        # Plot 6: Gradient Steps vs Steps
        if 'gradient_steps' in df.columns:
            axs[2, 1].plot(df['steps'], df['gradient_steps'], label=f'{prefix}Gradient Steps',
                           color=color if multi else 'orange', alpha=0.8, linewidth=2)

    # Titles / labels / grid are per-axis and shared across runs.
    titles = [
        ('Score vs Steps', 'Environment Steps', 'Score'),
        ('Episode Length vs Steps', 'Environment Steps', 'Length'),
        ('Actor Loss vs Steps', 'Environment Steps', 'Loss'),
        ('Critic Loss vs Steps', 'Environment Steps', 'Loss'),
        ('Steps Per Second (SPS) vs Steps', 'Environment Steps', 'SPS'),
        ('Gradient Steps vs Env Steps', 'Environment Steps', 'Gradient Steps'),
    ]
    for ax, (title, xlabel, ylabel) in zip(axs.flat, titles):
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle='--', alpha=0.6)
        # Only show a legend where something was actually plotted.
        if ax.get_legend_handles_labels()[0]:
            ax.legend()

    # Adjust layout to prevent overlapping text and accommodate the main title
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])

    # Save the figure to a PDF
    try:
        plt.savefig(args.output, format='pdf', bbox_inches='tight')
        print(f"Successfully generated plots and saved to: {args.output}")
    except Exception as e:
        print(f"Error saving PDF: {e}")


if __name__ == "__main__":
    main()
