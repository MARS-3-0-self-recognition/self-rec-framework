#!/usr/bin/env python3
"""
Aggregate evaluator performance across multiple datasets.

This script loads evaluator_performance.csv files from multiple datasets and creates
a stacked bar chart showing performance across datasets.

Usage:
    uv run experiments/_scripts/analysis/performance_aggregate.py \
        --performance_files data/analysis/wikisum/.../evaluator_performance.csv \
                              data/analysis/sharegpt/.../evaluator_performance.csv \
        --dataset_names "wikisum/training_set_1-20+test_set_1-30" "sharegpt/english_26+english2_74" \
        --model_names -set dr

Output:
    - data/analysis/_aggregated_data/{datetime}/
        - aggregated_performance.csv: Merged performance data
        - aggregated_performance.png: Stacked bar chart
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
    # Take the first part before the first "/"
    return full_path.split("/")[0]


def load_performance_data(
    performance_files: list[Path],
    dataset_names: list[str],
    column_name: str = "performance",
) -> pd.DataFrame:
    """
    Load evaluator performance or deviation data from multiple CSV files and merge into a single DataFrame.

    Args:
        performance_files: List of paths to evaluator_performance.csv or evaluator_deviation.csv files
        dataset_names: List of dataset names/labels (same order as files)
        column_name: Name of the column to extract ('performance' or 'deviation')

    Returns:
        DataFrame with models as index, datasets as columns, scores as values
    """
    merged_data = {}

    for file_path, dataset_name in zip(performance_files, dataset_names):
        if not file_path.exists():
            print(f"  ⚠ Warning: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path, index_col=0)

        # Extract specified column
        if column_name in df.columns:
            merged_data[dataset_name] = df[column_name]
        else:
            print(f"  ⚠ Warning: '{column_name}' column not found in {file_path}")
            continue

    if not merged_data:
        raise ValueError(f"No valid {column_name} data loaded!")

    # Create DataFrame with datasets as columns, models as index
    result_df = pd.DataFrame(merged_data)

    # Fill missing values with 0 (models not present in a dataset)
    result_df = result_df.fillna(0)

    return result_df


def plot_stacked_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
    is_deviation: bool = False,
):
    """
    Create a stacked bar chart showing evaluator performance or deviation across datasets.

    For performance: regular stacked bars (all positive, stacked rightward)
    For deviation: diverging stacked bars (positive right, negative left, centered at 0)

    Args:
        df: DataFrame with models as index, datasets as columns, scores as values
        output_path: Path to save the plot
        experiment_title: Optional title for the plot
        is_deviation: If True, create diverging chart (positive/negative), else regular stacked
    """
    chart_type = "deviation" if is_deviation else "performance"
    print(f"Generating stacked bar chart ({chart_type})...")

    # Remove models where all values are 0 (not present in any dataset)
    df = df.loc[(df != 0).any(axis=1)]

    if df.empty:
        print(f"  ⚠ No data to plot (all models have zero {chart_type})")
        return

    # Shorten dataset names for legend (extract just the dataset name)
    short_names = [extract_dataset_name(name) for name in df.columns]
    df.columns = short_names

    if is_deviation:
        # Diverging stacked bar chart
        # Separate positive and negative values
        df_positive = df.copy()
        df_negative = df.copy()
        df_positive[df_positive < 0] = 0
        df_negative[df_negative > 0] = 0

        # Sort by total absolute deviation
        df["_total_abs"] = df.abs().sum(axis=1)
        df = df.sort_values("_total_abs", ascending=False)
        df = df.drop(columns=["_total_abs"])
        df_positive = df_positive.reindex(df.index)
        df_negative = df_negative.reindex(df.index)

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

        # Add reference line at 0 (chance level)
        ax.axvline(
            x=0,
            color="black",
            linestyle="--",
            linewidth=1,
            alpha=0.5,
            label="Chance (0.5)",
        )

        # Set x-axis limits
        max_pos = df_positive.sum(axis=1).max()
        max_neg = abs(df_negative.sum(axis=1).min())
        max_val = max(max_pos, max_neg)
        padding = max(max_val * 0.1, 0.05)
        ax.set_xlim(-max_val - padding, max_val + padding)

        xlabel = "Aggregated Deviation from Chance (Sum Across Datasets)"
        title = "Evaluator Deviation from Chance Across Datasets (Diverging Stacked)"
    else:
        # Regular stacked bar chart
        # Sort by total performance (sum across all datasets) - descending
        df["_total"] = df.sum(axis=1)
        df = df.sort_values("_total", ascending=False)
        df = df.drop(columns=["_total"])

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

        # Create stacked bar chart
        bottom = np.zeros(len(df))
        for i, (dataset, color) in enumerate(zip(df.columns, colors)):
            values = df[dataset].values
            ax.barh(range(len(df)), values, left=bottom, label=dataset, color=color)
            bottom += values

        # Set x-axis limits
        max_val = df.sum(axis=1).max()
        padding = max(max_val * 0.1, 0.05)
        ax.set_xlim(0, max_val + padding)

        xlabel = "Aggregated Performance Score (Sum Across Datasets)"
        title = "Evaluator Performance Across Datasets (Stacked)"

    # Set y-axis labels
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df.index)
    ax.invert_yaxis()  # Top to bottom

    # Labels and title
    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add legend
    ax.legend(loc="lower right" if not is_deviation else "upper right", fontsize=10)

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add total value labels
    if is_deviation:
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
    else:
        totals = df.sum(axis=1)
        for i, (model, total) in enumerate(totals.items()):
            if total > 0:
                ax.text(
                    total + padding * 0.02,
                    i,
                    f"{total:.3f}",
                    va="center",
                    ha="left",
                    fontsize=9,
                )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved stacked bar chart to: {output_path}")
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
        description="Aggregate evaluator performance across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--performance_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluator_performance.csv files from different datasets",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="Names/labels for each dataset (same order as performance_files)",
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

    # Validate inputs
    if len(args.performance_files) != len(args.dataset_names):
        print("Error: Number of performance files must match number of dataset names")
        return

    performance_files = [Path(f) for f in args.performance_files]
    dataset_names = args.dataset_names

    # Validate all files exist
    for file_path in performance_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

    print(f"{'='*70}")
    print("PERFORMANCE AGGREGATION")
    print(f"{'='*70}")
    print(f"Datasets: {len(dataset_names)}")
    for name in dataset_names:
        print(f"  • {name}")
    print()

    # Extract experiment name from first file path if available
    experiment_title = ""
    experiment_name_full = ""
    first_file_parts = performance_files[0].parts
    if len(first_file_parts) >= 3:
        # Path: data/analysis/{dataset}/{experiment}/evaluator_performance/evaluator_performance.csv
        experiment_name_full = first_file_parts[
            -3
        ]  # Get full experiment name (e.g., "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")

        # For display title, remove leading number if present
        if "_" in experiment_name_full:
            parts = experiment_name_full.split("_", 1)
            if parts[0].isdigit():
                experiment_title = parts[
                    1
                ]  # For display title (e.g., "UT_PW-Q_Rec_NPr_FA_Inst")
            else:
                experiment_title = experiment_name_full
        else:
            experiment_title = experiment_name_full

    # Create output directory with experiment name and timestamp
    output_base = Path("data/analysis/_aggregated_data")
    if experiment_name_full:
        output_base = (
            output_base / experiment_name_full
        )  # Use full name for directory (e.g., "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # ============================================================================
    # Process Performance Data
    # ============================================================================

    print("Loading performance data...")
    df_perf = load_performance_data(
        performance_files, dataset_names, column_name="performance"
    )
    print(f"  ✓ Loaded data: {df_perf.shape[0]} models × {df_perf.shape[1]} datasets\n")

    # Filter and order models
    if model_order:
        available_models = [m for m in model_order if m in df_perf.index]
        if available_models:
            df_perf = df_perf.reindex(available_models)
        else:
            print("  ⚠ Warning: No models from filter list found in data")

    # Remove models with all zeros
    df_perf = df_perf.loc[(df_perf != 0).any(axis=1)]

    if not df_perf.empty:
        # Save merged CSV
        csv_path = output_dir / "aggregated_performance.csv"
        df_perf.to_csv(csv_path)
        print(f"  ✓ Saved aggregated performance data to: {csv_path}\n")

        # Generate plot
        plot_path = output_dir / "aggregated_performance.png"
        plot_stacked_bar_chart(
            df_perf, plot_path, experiment_title=experiment_title, is_deviation=False
        )
        print()

        # Display preview
        print(f"{'='*70}")
        print("PREVIEW: Aggregated Performance (Total Across Datasets)")
        print(f"{'='*70}\n")
        totals = df_perf.sum(axis=1).sort_values(ascending=False)
        print(totals.round(3))
        print()

    # ============================================================================
    # Process Deviation Data
    # ============================================================================

    # Build paths to deviation files (same structure as performance files)
    deviation_files = []
    for perf_file in performance_files:
        # Replace 'evaluator_performance.csv' with 'evaluator_deviation.csv'
        dev_file = perf_file.parent / "evaluator_deviation.csv"
        deviation_files.append(dev_file)

    # Check which deviation files exist
    existing_deviation_files = []
    existing_deviation_names = []
    for dev_file, dataset_name in zip(deviation_files, dataset_names):
        if dev_file.exists():
            existing_deviation_files.append(dev_file)
            existing_deviation_names.append(dataset_name)
        else:
            print(f"  ⚠ Warning: Deviation file not found: {dev_file}")

    df_dev = None
    if existing_deviation_files:
        print("Loading deviation data...")
        df_dev = load_performance_data(
            existing_deviation_files, existing_deviation_names, column_name="deviation"
        )
        print(
            f"  ✓ Loaded data: {df_dev.shape[0]} models × {df_dev.shape[1]} datasets\n"
        )

        # Filter and order models (use same order as performance)
        if model_order:
            available_models = [m for m in model_order if m in df_dev.index]
            if available_models:
                df_dev = df_dev.reindex(available_models)

        # Remove models with all zeros
        df_dev = df_dev.loc[(df_dev != 0).any(axis=1)]

        if not df_dev.empty:
            # Save merged CSV
            csv_path = output_dir / "aggregated_deviation.csv"
            df_dev.to_csv(csv_path)
            print(f"  ✓ Saved aggregated deviation data to: {csv_path}\n")

            # Generate diverging stacked bar chart
            plot_path = output_dir / "aggregated_deviation.png"
            plot_stacked_bar_chart(
                df_dev, plot_path, experiment_title=experiment_title, is_deviation=True
            )
            print()

            # Display preview
            print(f"{'='*70}")
            print("PREVIEW: Aggregated Deviation (Total Across Datasets)")
            print(f"{'='*70}\n")
            totals = df_dev.sum(axis=1).sort_values(ascending=False)
            print(totals.round(3))
            print()

    # ============================================================================
    # Summary
    # ============================================================================

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    if not df_perf.empty:
        print("  • aggregated_performance.csv: Merged performance data")
        print("  • aggregated_performance.png: Stacked bar chart")
    if df_dev is not None and not df_dev.empty:
        print("  • aggregated_deviation.csv: Merged deviation data")
        print("  • aggregated_deviation.png: Diverging stacked bar chart")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
