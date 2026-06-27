import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Generate learning curves from a TD3 log CSV.")
    parser.add_argument("--csv-path", required=True, type=str, help="Path to the input log.csv file")
    parser.add_argument("--output", type=str, default="learning_curves.pdf", help="Path to save the output PDF (default: learning_curves.pdf)")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    if not csv_path.exists():
        print(f"Error: The file '{csv_path}' does not exist.")
        return

    # Load the data
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} rows from {csv_path.name}")
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    # Create the figure and subplots
    fig, axs = plt.subplots(3, 2, figsize=(15, 15))
    fig.suptitle(f"Training Metrics: {csv_path.name}", fontsize=16, fontweight='bold')

    # Plot 1: Scores vs Steps
    if 'score' in df.columns and 'test/score' in df.columns:
        axs[0, 0].plot(df['steps'], df['score'], label='Train Score', alpha=0.8, linewidth=2)
        axs[0, 0].plot(df['steps'], df['test/score'], label='Test Score', alpha=0.8, linewidth=2)
        axs[0, 0].set_title('Score vs Steps')
        axs[0, 0].set_xlabel('Environment Steps')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Episode Length vs Steps
    if 'length' in df.columns and 'test/length' in df.columns:
        axs[0, 1].plot(df['steps'], df['length'], label='Train Length', alpha=0.8, linewidth=2)
        axs[0, 1].plot(df['steps'], df['test/length'], label='Test Length', alpha=0.8, linewidth=2)
        axs[0, 1].set_title('Episode Length vs Steps')
        axs[0, 1].set_xlabel('Environment Steps')
        axs[0, 1].set_ylabel('Length')
        axs[0, 1].legend()
        axs[0, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 3: Actor Loss vs Steps
    if 'loss/actor' in df.columns:
        axs[1, 0].plot(df['steps'], df['loss/actor'], color='green', alpha=0.8, linewidth=2)
        axs[1, 0].set_title('Actor Loss vs Steps')
        axs[1, 0].set_xlabel('Environment Steps')
        axs[1, 0].set_ylabel('Loss')
        axs[1, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 4: Critic Loss vs Steps
    if 'loss/critic' in df.columns:
        axs[1, 1].plot(df['steps'], df['loss/critic'], color='red', alpha=0.8, linewidth=2)
        axs[1, 1].set_title('Critic Loss vs Steps')
        axs[1, 1].set_xlabel('Environment Steps')
        axs[1, 1].set_ylabel('Loss')
        axs[1, 1].grid(True, linestyle='--', alpha=0.6)

    # Plot 5: SPS vs Steps
    if 'sps' in df.columns:
        axs[2, 0].plot(df['steps'], df['sps'], color='purple', alpha=0.8, linewidth=2)
        axs[2, 0].set_title('Steps Per Second (SPS) vs Steps')
        axs[2, 0].set_xlabel('Environment Steps')
        axs[2, 0].set_ylabel('SPS')
        axs[2, 0].grid(True, linestyle='--', alpha=0.6)

    # Plot 6: Gradient Steps vs Steps
    if 'gradient_steps' in df.columns:
        axs[2, 1].plot(df['steps'], df['gradient_steps'], color='orange', alpha=0.8, linewidth=2)
        axs[2, 1].set_title('Gradient Steps vs Env Steps')
        axs[2, 1].set_xlabel('Environment Steps')
        axs[2, 1].set_ylabel('Gradient Steps')
        axs[2, 1].grid(True, linestyle='--', alpha=0.6)

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