import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def run_label(csv_path: Path) -> str:
    """Derive a concise legend label for a run from its CSV path.

    Uses the parent directory name when the file is a generic ``log.csv``,
    otherwise the file stem, so overlaid runs stay distinguishable.
    """
    if csv_path.stem == "log":
        return csv_path.parent.name or csv_path.stem
    return csv_path.stem


def load_runs(csv_paths):
    """Load each CSV into a (label, DataFrame) pair, skipping unreadable ones."""
    runs = []
    for raw in csv_paths:
        csv_path = Path(raw)
        if not csv_path.exists():
            print(f"Warning: skipping '{csv_path}' (does not exist)")
            continue
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"Warning: skipping '{csv_path}' (error reading CSV: {e})")
            continue
        print(f"Loaded {len(df)} rows from {csv_path}")
        runs.append((run_label(csv_path), df))
    return runs


def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate learning curves from one or more TD3 log CSVs.")
    parser.add_argument("--csv-path", required=True, nargs="+", type=str,
                        help="Path(s) to one or more input log.csv files. Multiple runs are overlaid.")
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
