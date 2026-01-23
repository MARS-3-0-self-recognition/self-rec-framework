#!/usr/bin/env python3
"""
Compute dataset-level statistics from aggregated performance data.

This script loads aggregated_performance.csv and computes:
- Average recognition accuracy per dataset (across all models)
- Standard deviation per dataset (across all models)

Creates a bar chart with error bars showing these statistics.

Usage:
    uv run experiments/_scripts/analysis/performance_contrast.py \
        --aggregated_file data/analysis/_aggregated_data/.../aggregated_performance.csv \
        --model_names -set dr

Output:
    - Same directory as input file:
        - dataset_averages.csv: Average, std dev, and % above chance per dataset
        - dataset_averages.png: Bar chart with error bars
        - dataset_above_chance.png: Percentage of models above chance per dataset
"""

import argparse
import sys
from pathlib import Path

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


def load_and_filter_data(
    aggregated_file: Path,
    model_order: list[str] | None = None,
) -> pd.DataFrame:
    """
    Load aggregated performance data and filter models.

    If a model has NaN for any dataset, exclude it from ALL datasets.

    Args:
        aggregated_file: Path to aggregated_performance.csv
        model_order: Optional list of models to filter/order

    Returns:
        Filtered DataFrame with models as index, datasets as columns
    """
    df = pd.read_csv(aggregated_file, index_col=0)

    # Filter and order models if specified
    if model_order:
        available_models = [m for m in model_order if m in df.index]
        if available_models:
            df = df.reindex(available_models)
        else:
            print("  ⚠ Warning: No models from filter list found in data")

    # Remove models that have NaN in ANY dataset (exclude everywhere)
    models_with_nan = df.isna().any(axis=1)
    if models_with_nan.any():
        excluded = df.index[models_with_nan].tolist()
        print(
            f"  ⚠ Excluding {len(excluded)} model(s) with missing data: {', '.join(excluded)}"
        )
        df = df.loc[~models_with_nan]

    # Fill any remaining NaN with 0 (shouldn't happen after above, but just in case)
    df = df.fillna(0)

    return df


def compute_dataset_statistics(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """
    Compute average and standard deviation per dataset.

    Args:
        df: DataFrame with models as rows, datasets as columns

    Returns:
        Tuple of (averages, std_devs) - both Series with datasets as index
    """
    # Average across models (row mean)
    averages = df.mean(axis=0)

    # Standard deviation across models (row std)
    std_devs = df.std(axis=0)

    return averages, std_devs


def compute_above_chance_percentage(
    df: pd.DataFrame, chance_level: float = 0.5
) -> pd.Series:
    """
    Compute percentage of models scoring above chance per dataset.

    Args:
        df: DataFrame with models as rows, datasets as columns
        chance_level: Threshold for "above chance" (default: 0.5)

    Returns:
        Series with datasets as index, percentages (0-100) as values
    """
    # Count models above chance per dataset
    above_chance = (df > chance_level).sum(axis=0)

    # Total number of models
    total_models = len(df)

    # Compute percentage
    percentages = (above_chance / total_models) * 100

    return percentages


def plot_dataset_bar_chart(
    values: pd.Series,
    errors: pd.Series,
    output_path: Path,
    title: str,
    ylabel: str,
    experiment_title: str = "",
):
    """
    Create a bar chart with error bars showing dataset-level statistics.

    Args:
        values: Series with datasets as index, average values
        errors: Series with datasets as index, standard deviation values
        output_path: Path to save the plot
        title: Chart title
        ylabel: Y-axis label
        experiment_title: Optional experiment name for title
    """
    print(f"Generating bar chart with error bars: {title}...")

    if len(values) == 0:
        print("  ⚠ No data to plot")
        return

    # Ensure errors and values have same index order
    errors = errors.reindex(values.index)

    # Shorten dataset names
    short_names = [extract_dataset_name(name) for name in values.index]
    values.index = short_names
    errors.index = short_names

    # Sort by value (descending)
    sort_order = values.sort_values(ascending=False).index
    values = values.reindex(sort_order)
    errors = errors.reindex(sort_order)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose distinct colors for datasets
    n_datasets = len(values)
    if n_datasets <= 4:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
    elif n_datasets <= 8:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

    # Create bar chart with error bars
    x_pos = range(len(values))
    ax.bar(
        x_pos,
        values.values,
        color=colors,
        yerr=errors.values,
        capsize=5,
        error_kw={"elinewidth": 1.5, "capthick": 1.5, "alpha": 0.7},
    )

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(values.index, rotation=45, ha="right")

    # Set y-axis limits
    min_val = (values - errors).min()
    max_val = (values + errors).max()
    val_range = max_val - min_val
    padding = max(val_range * 0.1, 0.05)
    ax.set_ylim(max(0, min_val - padding), max_val + padding)

    # Labels and title
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")

    full_title = title
    if experiment_title:
        full_title = f"{title}\n{experiment_title}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=20)

    # Add value labels on top of bars (above error bar)
    for i, (dataset, val, err) in enumerate(
        zip(values.index, values.values, errors.values)
    ):
        label_y = val + err + padding * 0.02
        ax.text(i, label_y, f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    # Add grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved bar chart to: {output_path}")
    plt.close()


def plot_above_chance_bar_chart(
    percentages: pd.Series,
    output_path: Path,
    title: str,
    experiment_title: str = "",
):
    """
    Create a bar chart showing percentage of models above chance per dataset.

    Args:
        percentages: Series with datasets as index, percentages (0-100) as values
        output_path: Path to save the plot
        title: Chart title
        experiment_title: Optional experiment name for title
    """
    print(f"Generating bar chart: {title}...")

    if len(percentages) == 0:
        print("  ⚠ No data to plot")
        return

    # Shorten dataset names
    short_names = [extract_dataset_name(name) for name in percentages.index]
    percentages.index = short_names

    # Sort by value (descending)
    percentages = percentages.sort_values(ascending=False)

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Choose distinct colors for datasets
    n_datasets = len(percentages)
    if n_datasets <= 4:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
    elif n_datasets <= 8:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

    # Create bar chart
    x_pos = range(len(percentages))
    ax.bar(x_pos, percentages.values, color=colors)

    # Add reference line at 50% (half the models)
    ax.axhline(
        y=50,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        alpha=0.5,
        label="50% (Half of Models)",
    )

    # Set x-axis labels
    ax.set_xticks(x_pos)
    ax.set_xticklabels(percentages.index, rotation=45, ha="right")

    # Set y-axis limits (0-100%)
    ax.set_ylim(0, 105)

    # Labels and title
    ax.set_ylabel(
        "Percentage of Models Above Chance (%)", fontsize=12, fontweight="bold"
    )
    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")

    full_title = title
    if experiment_title:
        full_title = f"{title}\n{experiment_title}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=20)

    # Add value labels on top of bars
    for i, (dataset, pct) in enumerate(percentages.items()):
        ax.text(
            i,
            pct + 2,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Add grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    ax.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved bar chart to: {output_path}")
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
        description="Compute dataset-level statistics from aggregated performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--aggregated_file",
        type=str,
        required=True,
        help="Path to aggregated_performance.csv file",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to include (filters results). "
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

    aggregated_file = Path(args.aggregated_file)

    # Validate file exists
    if not aggregated_file.exists():
        print(f"Error: File not found: {aggregated_file}")
        return

    print(f"{'='*70}")
    print("DATASET-LEVEL STATISTICS")
    print(f"{'='*70}")
    print(f"Input file: {aggregated_file}\n")

    # Load and filter data
    print("Loading and filtering data...")
    df = load_and_filter_data(aggregated_file, model_order=model_order)
    print(f"  ✓ Loaded data: {df.shape[0]} models × {df.shape[1]} datasets\n")

    if df.empty:
        print("⚠ No data to analyze after filtering!")
        return

    # Compute statistics
    print("Computing dataset statistics...")
    averages, std_devs = compute_dataset_statistics(df)
    print(f"  ✓ Computed averages and standard deviations for {len(averages)} datasets")

    # Compute percentage above chance
    above_chance_pct = compute_above_chance_percentage(df, chance_level=0.5)
    print(
        f"  ✓ Computed percentage above chance for {len(above_chance_pct)} datasets\n"
    )

    # Determine output directory (same as input file)
    output_dir = aggregated_file.parent

    # Extract experiment name from path if available
    experiment_title = ""
    path_parts = aggregated_file.parts
    if len(path_parts) >= 2:
        # Path: .../_aggregated_data/{experiment}/{timestamp}/aggregated_performance.csv
        exp_name = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        if "_" in exp_name:
            parts = exp_name.split("_", 1)
            if parts[0].isdigit():
                experiment_title = parts[1]
            else:
                experiment_title = exp_name
        else:
            experiment_title = exp_name

    # Save CSV file with both averages and std devs
    stats_df = pd.DataFrame(
        {
            "average_accuracy": averages,
            "std_dev": std_devs,
            "pct_above_chance": above_chance_pct,
        }
    )
    averages_path = output_dir / "dataset_averages.csv"
    stats_df.to_csv(averages_path)
    print(f"  ✓ Saved dataset statistics to: {averages_path}\n")

    # Generate plots
    averages_plot_path = output_dir / "dataset_averages.png"
    plot_dataset_bar_chart(
        averages,
        std_devs,
        averages_plot_path,
        title="Average Recognition Accuracy per Dataset",
        ylabel="Average Accuracy (Across All Models)",
        experiment_title=experiment_title,
    )
    print()

    above_chance_plot_path = output_dir / "dataset_above_chance.png"
    plot_above_chance_bar_chart(
        above_chance_pct,
        above_chance_plot_path,
        title="Percentage of Models Above Chance (0.5) per Dataset",
        experiment_title=experiment_title,
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Dataset Averages (with Std Dev)")
    print(f"{'='*70}\n")
    preview_df = stats_df.sort_values("average_accuracy", ascending=False)
    print(preview_df.round(3))
    print()

    print(f"{'='*70}")
    print("PREVIEW: Percentage Above Chance")
    print(f"{'='*70}\n")
    print(above_chance_pct.sort_values(ascending=False).round(1).astype(str) + "%")
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(
        "  • dataset_averages.csv: Average accuracy, std dev, and % above chance per dataset"
    )
    print("  • dataset_averages.png: Bar chart with error bars")
    print("  • dataset_above_chance.png: Percentage of models above chance per dataset")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
