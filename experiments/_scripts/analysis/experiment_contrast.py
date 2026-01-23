#!/usr/bin/env python3
"""
Compare evaluator performance across two experiments.

This script loads aggregated performance CSV files from two experiments and creates
a diverging stacked bar chart showing the difference (exp1 - exp2) per model and dataset.

Usage:
    uv run experiments/_scripts/analysis/experiment_contrast.py \
        --exp1_file data/analysis/_aggregated_data/ICML_01_.../aggregated_performance.csv \
        --exp2_file data/analysis/_aggregated_data/ICML_02_.../aggregated_performance.csv \
        --exp1_name ICML_01 \
        --exp2_name ICML_02 \
        --model_names -set dr

Output:
    - data/analysis/_aggregated_data/{exp1-vs-exp2}/{timestamp}/
        - performance_contrast.csv: Difference data (exp1 - exp2)
        - performance_contrast.png: Diverging stacked bar chart
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import expand_model_names


def extract_dataset_name(full_path: str) -> str:
    """
    Extract short dataset name from full path.

    Examples:
        "wikisum/training_set_1-20+test_set_1-30" -> "wikisum"
        "sharegpt/english_26+english2_74" -> "sharegpt"
        "bigcodebench/instruct_1-50" -> "bigcodebench"
    """
    return full_path.split("/")[0]


def load_and_compare(
    exp1_file: Path,
    exp2_file: Path,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load both performance files and compute difference (exp1 - exp2).

    Args:
        exp1_file: Path to first experiment's aggregated_performance.csv
        exp2_file: Path to second experiment's aggregated_performance.csv
        model_order: Optional list of models to filter/order

    Returns:
        DataFrame with models as index, datasets as columns, differences as values
    """
    # Load both files
    df1 = pd.read_csv(exp1_file, index_col=0)
    df2 = pd.read_csv(exp2_file, index_col=0)

    # Find common models (exclude models that don't exist in both)
    common_models = df1.index.intersection(df2.index)

    if len(common_models) == 0:
        raise ValueError("No common models found between the two experiments!")

    # Filter to common models
    df1 = df1.loc[common_models]
    df2 = df2.loc[common_models]

    # Find common datasets
    common_datasets = df1.columns.intersection(df2.columns)

    if len(common_datasets) == 0:
        raise ValueError("No common datasets found between the two experiments!")

    # Filter to common datasets
    df1 = df1[common_datasets]
    df2 = df2[common_datasets]

    # Compute difference: exp1 - exp2
    df_diff = df1 - df2

    # Filter and order models if specified
    if model_order:
        available_models = [m for m in model_order if m in df_diff.index]
        if available_models:
            df_diff = df_diff.reindex(available_models)

    return df_diff


def plot_diverging_stacked_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    exp1_name: str,
    exp2_name: str,
):
    """
    Create a diverging stacked bar chart showing exp1 - exp2 differences.

    Args:
        df: DataFrame with models as index, datasets as columns, differences as values
        output_path: Path to save the plot
        exp1_name: Name of first experiment (for title)
        exp2_name: Name of second experiment (for title)
    """
    print("Generating diverging stacked bar chart...")

    # Remove models where all values are 0
    df = df.loc[(df != 0).any(axis=1)]

    if df.empty:
        print("  ⚠ No data to plot (all differences are zero)")
        return

    # Shorten dataset names for legend
    short_names = [extract_dataset_name(name) for name in df.columns]
    df.columns = short_names

    # Sort by total absolute difference
    df["_total_abs"] = df.abs().sum(axis=1)
    df = df.sort_values("_total_abs", ascending=False)
    df = df.drop(columns=["_total_abs"])

    # Separate positive and negative values
    df_positive = df.copy()
    df_negative = df.copy()
    df_positive[df_positive < 0] = 0
    df_negative[df_negative > 0] = 0

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, max(8, len(df) * 0.4)))

    # Choose colors for datasets
    n_datasets = len(df.columns)
    if n_datasets <= 4:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
    elif n_datasets <= 8:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

    # Stack negative values to the left (from 0 going negative)
    bottom_neg = np.zeros(len(df))
    for i, (dataset, color) in enumerate(zip(df.columns, colors)):
        values = df_negative[dataset].values
        ax.barh(
            range(len(df)),
            values,
            left=bottom_neg,
            label=dataset,
            color=color,
            alpha=0.7,
        )
        bottom_neg += values

    # Stack positive values to the right (from 0 going positive)
    bottom_pos = np.zeros(len(df))
    for i, (dataset, color) in enumerate(zip(df.columns, colors)):
        values = df_positive[dataset].values
        ax.barh(range(len(df)), values, left=bottom_pos, color=color, alpha=0.7)
        bottom_pos += values

    # Add reference line at 0
    ax.axvline(
        x=0,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="No difference",
    )

    # Set x-axis limits
    max_pos = df_positive.sum(axis=1).max()
    max_neg = abs(df_negative.sum(axis=1).min())
    max_val = max(max_pos, max_neg)
    padding = max(max_val * 0.1, 0.05)
    ax.set_xlim(-max_val - padding, max_val + padding)

    # Set y-axis labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()  # Top to bottom

    # Labels and title
    ax.set_xlabel(
        f"Performance Difference ({exp1_name} - {exp2_name})",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    title = f"Performance Contrast: {exp1_name} vs {exp2_name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add legend
    ax.legend(loc="lower right", fontsize=10)

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add total value labels
    totals = df.sum(axis=1)
    for i, (model, total) in enumerate(totals.items()):
        if total != 0:
            label_x = (
                total + (padding * 0.02) if total >= 0 else total - (padding * 0.02)
            )
            ax.text(
                label_x,
                i,
                f"{total:.3f}",
                va="center",
                ha="left" if total >= 0 else "right",
                fontsize=9,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved diverging bar chart to: {output_path}")
    plt.close()


def main():
    # Preprocess sys.argv to handle -set before argparse sees it
    if "--model_names" in sys.argv:
        model_names_idx = sys.argv.index("--model_names")
        for i in range(model_names_idx + 1, len(sys.argv)):
            if sys.argv[i] == "-set" and (
                i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
            ):
                sys.argv[i] = "SET_PLACEHOLDER"

    parser = argparse.ArgumentParser(
        description="Compare evaluator performance across two experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--exp1_file",
        type=str,
        required=True,
        help="Path to first experiment's aggregated_performance.csv",
    )
    parser.add_argument(
        "--exp2_file",
        type=str,
        required=True,
        help="Path to second experiment's aggregated_performance.csv",
    )
    parser.add_argument(
        "--exp1_name",
        type=str,
        required=True,
        help="Name of first experiment (for display)",
    )
    parser.add_argument(
        "--exp2_name",
        type=str,
        required=True,
        help="Name of second experiment (for display)",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to include (filters and orders results). "
        "Supports -set notation (e.g., --model_names -set dr) or explicit names",
    )

    args = parser.parse_args()

    # Restore -set from placeholder
    args.model_names = [
        arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names
    ]

    # Expand model set references
    model_order = expand_model_names(args.model_names)
    print(f"Model filter/order: {', '.join(model_order)}\n")

    exp1_file = Path(args.exp1_file)
    exp2_file = Path(args.exp2_file)

    # Validate files exist
    if not exp1_file.exists():
        print(f"Error: File not found: {exp1_file}")
        return

    if not exp2_file.exists():
        print(f"Error: File not found: {exp2_file}")
        return

    print(f"{'='*70}")
    print("PERFORMANCE CONTRAST")
    print(f"{'='*70}")
    print(f"Experiment 1: {args.exp1_name}")
    print(f"  File: {exp1_file}")
    print(f"Experiment 2: {args.exp2_name}")
    print(f"  File: {exp2_file}")
    print()

    # Load and compute differences
    print("Loading and comparing performance data...")
    try:
        df_diff = load_and_compare(exp1_file, exp2_file, model_order=model_order)
        print(
            f"  ✓ Computed differences: {df_diff.shape[0]} models × {df_diff.shape[1]} datasets\n"
        )
    except ValueError as e:
        print(f"  ✗ Error: {e}")
        return

    if df_diff.empty:
        print("⚠ No data to analyze after filtering!")
        return

    # Create output directory
    comparison_name = f"{args.exp1_name}-vs-{args.exp2_name}"
    output_base = Path("data/analysis/_aggregated_data") / comparison_name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Save difference CSV
    csv_path = output_dir / "performance_contrast.csv"
    df_diff.to_csv(csv_path)
    print(f"  ✓ Saved difference data to: {csv_path}\n")

    # Generate plot
    plot_path = output_dir / "performance_contrast.png"
    plot_diverging_stacked_bar_chart(df_diff, plot_path, args.exp1_name, args.exp2_name)
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Performance Difference (Total Across Datasets)")
    print(f"{'='*70}\n")
    totals = df_diff.sum(axis=1).sort_values(ascending=False)
    print(totals.round(3))
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • performance_contrast.csv: Difference data (exp1 - exp2)")
    print("  • performance_contrast.png: Diverging stacked bar chart")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
