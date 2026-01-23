#!/usr/bin/env python3
"""
Calculate average evaluator performance from recognition accuracy data.

This script loads the accuracy pivot table from recognition_accuracy.py and computes
performance metrics for each evaluator model.

For pairwise format: Computes row means (average accuracy across all treatments).
For individual format: Computes D_j = (C_j + mean(T_i)) / 2 - 0.5 (deviation from chance).

Usage:
    uv run experiments/_scripts/analysis/evaluator_performance.py \
        --results_dir data/results/wikisum/training_set_1-20/EXP_NAME \
        --model_names -set dr

Output:
    - data/analysis/{dataset}/{subset}/{experiment}/evaluator_performance/
        - evaluator_performance.csv: Performance scores per evaluator
        - evaluator_performance.png: Bar chart visualization
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import expand_model_names, get_model_family_colors


def get_experiment_name_mapping() -> dict[str, str]:
    """Map experiment code abbreviations to full descriptive names."""
    return {
        "UT_PW-Q_Rec_Pr": "User Tags Pairwise Query Primed",
        "UT_PW-Q_Rec_NPr": "User Tags Pairwise Query Unprimed",
        "UT_PW-C_Rec_Pr": "User Tags Pairwise Conversation Primed",
        "UT_PW-C_Rec_NPr": "User Tags Pairwise Conversation Unprimed",
        "UT_IND-Q_Rec_Pr": "User Tags Individual Query Primed",
        "UT_IND-Q_Rec_NPr": "User Tags Individual Query Unprimed",
        "UT_IND-C_Rec_Pr": "User Tags Individual Conversation Primed",
        "UT_IND-C_Rec_NPr": "User Tags Individual Conversation Unprimed",
        "AT_PW-Q_Rec_Pr": "Assistant Tags Pairwise Query Primed",
        "AT_PW-Q_Rec_NPr": "Assistant Tags Pairwise Query Unprimed",
        "AT_PW-C_Rec_Pr": "Assistant Tags Pairwise Conversation Primed",
        "AT_PW-C_Rec_NPr": "Assistant Tags Pairwise Conversation Unprimed",
        "AT_IND-Q_Rec_Pr": "Assistant Tags Individual Query Primed",
        "AT_IND-Q_Rec_NPr": "Assistant Tags Individual Query Unprimed",
        "AT_IND-C_Rec_Pr": "Assistant Tags Individual Conversation Primed",
        "AT_IND-C_Rec_NPr": "Assistant Tags Individual Conversation Unprimed",
    }


def compute_individual_performance(pivot: pd.DataFrame) -> pd.Series:
    """
    Compute performance for individual format: D_j = (C_j + mean(T_i)) / 2.

    For each evaluator model j:
    - C_j = control condition accuracy (diagonal: pivot[j, j])
    - T_i = treatment condition accuracies (off-diagonal: pivot[j, i] where i ≠ j)
    - D_j = (C_j + mean(T_i)) / 2

    This measures average recognition accuracy (0 to 1 scale).
    Values above 0.5 = better than chance, below 0.5 = worse than chance.
    """
    performance_scores = pd.Series(dtype=float, index=pivot.index)

    for j, evaluator in enumerate(pivot.index):
        # C_j: control condition (diagonal)
        if evaluator in pivot.columns:
            C_j = pivot.loc[evaluator, evaluator]
        else:
            C_j = pd.NA

        # T_i: treatment conditions (off-diagonal, excluding j)
        treatment_values = []
        for i, model in enumerate(pivot.columns):
            if model != evaluator:  # Exclude diagonal
                T_i = pivot.loc[evaluator, model]
                if pd.notna(T_i):
                    treatment_values.append(T_i)

        # Compute performance: D_j = (C_j + mean(T_i)) / 2
        if pd.notna(C_j) and len(treatment_values) > 0:
            mean_T = np.mean(treatment_values)
            D_j = (C_j + mean_T) / 2.0
            performance_scores.loc[evaluator] = D_j
        else:
            performance_scores.loc[evaluator] = pd.NA

    return performance_scores


def compute_pairwise_performance(pivot: pd.DataFrame) -> pd.Series:
    """
    Compute performance for pairwise format: row means (average across all treatments).
    """
    return pivot.mean(axis=1, skipna=True)


def compute_individual_deviation(pivot: pd.DataFrame) -> pd.Series:
    """
    Compute deviation from chance for individual format: D_j = (C_j + mean(T_i)) / 2 - 0.5.

    Same as compute_individual_performance but subtracts 0.5 to show deviation from chance.
    Positive values = better than chance, negative = worse than chance.
    """
    performance_scores = compute_individual_performance(pivot)
    return performance_scores - 0.5


def compute_pairwise_deviation(pivot: pd.DataFrame) -> pd.Series:
    """
    Compute deviation from chance for pairwise format: row means - 0.5.

    Same as compute_pairwise_performance but subtracts 0.5 to show deviation from chance.
    Positive values = better than chance, negative = worse than chance.
    """
    performance_scores = compute_pairwise_performance(pivot)
    return performance_scores - 0.5


def plot_evaluator_performance(
    performance_scores: pd.Series,
    output_path: Path,
    experiment_title: str = "",
    is_individual_format: bool = False,
):
    """
    Create horizontal bar chart showing evaluator performance.

    For pairwise: Shows average accuracy across treatments with reference line at 0.5.
    For individual: Shows average recognition accuracy (control + mean treatment) / 2 with reference line at 0.5.
    """
    print("Generating evaluator performance plot...")

    # Filter out NaN values
    valid_scores = performance_scores.dropna()
    if len(valid_scores) == 0:
        print("  ⚠ No valid performance scores to plot")
        return

    # Sort by value (highest first)
    valid_scores = valid_scores.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid_scores) * 0.4)))

    # Get colors based on model family
    model_list = list(valid_scores.index)
    colors = get_model_family_colors(model_list)

    # Color bars by model family (same for both formats)
    bar_colors = [
        colors[model] if model in colors else "steelblue"
        for model in valid_scores.index
    ]

    ax.barh(range(len(valid_scores)), valid_scores.values, color=bar_colors)

    # Add vertical reference line at 0.5 (chance level) for both formats
    ax.axvline(
        x=0.5,
        color="black",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Chance (0.5)",
    )

    # Set y-axis labels
    ax.set_yticks(range(len(valid_scores)))
    ax.set_yticklabels(valid_scores.index)
    ax.invert_yaxis()  # Top to bottom

    # Calculate padding for x-axis (both formats use 0-1 range)
    min_val = valid_scores.min()
    max_val = valid_scores.max()
    val_range = max_val - min_val
    padding = max(val_range * 0.15, 0.05)
    x_min = max(0, min_val - padding)  # Don't go below 0
    x_max = min(1, max_val + padding)  # Don't go above 1

    ax.set_xlim(x_min, x_max)

    # Labels and title
    if is_individual_format:
        xlabel = "Average Recognition Accuracy ((Control + Mean Treatment) / 2)"
        title_suffix = (
            "\n(Average recognition accuracy: values above 0.5 = better than chance)"
        )
    else:
        xlabel = "Average Recognition Accuracy (Across All Treatments)"
        title_suffix = "\n(Average accuracy when model is evaluator: values above 0.5 = better than chance)"

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    title = "Evaluator Performance"
    title += title_suffix

    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add value labels on bars
    for i, (model, val) in enumerate(valid_scores.items()):
        label_x = val + (padding * 0.02)
        ax.text(label_x, i, f"{val:.3f}", va="center", ha="left", fontsize=9)

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved evaluator performance plot to: {output_path}")
    plt.close()


def plot_evaluator_deviation(
    deviation_scores: pd.Series,
    output_path: Path,
    experiment_title: str = "",
    is_individual_format: bool = False,
):
    """
    Create horizontal bar chart showing evaluator deviation from chance.

    Shows deviation from 0.5 (chance) with positive/negative color coding.
    Positive values = better than chance, negative = worse than chance.
    Reference line at 0 (chance level).
    """
    print("Generating evaluator deviation from chance plot...")

    # Filter out NaN values
    valid_scores = deviation_scores.dropna()
    if len(valid_scores) == 0:
        print("  ⚠ No valid deviation scores to plot")
        return

    # Sort by value (highest first)
    valid_scores = valid_scores.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(12, max(8, len(valid_scores) * 0.4)))

    # Get colors based on model family
    model_list = list(valid_scores.index)
    colors = get_model_family_colors(model_list)

    # Color bars: positive (better than chance) vs negative (worse than chance)
    bar_colors = []
    for model, score in valid_scores.items():
        if score >= 0:
            # Positive: better than chance
            bar_colors.append(colors[model] if model in colors else "steelblue")
        else:
            # Negative: worse than chance
            bar_colors.append("coral")

    ax.barh(range(len(valid_scores)), valid_scores.values, color=bar_colors)

    # Add vertical reference line at 0 (chance level)
    ax.axvline(
        x=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Chance (0.5)"
    )

    # Set y-axis labels
    ax.set_yticks(range(len(valid_scores)))
    ax.set_yticklabels(valid_scores.index)
    ax.invert_yaxis()  # Top to bottom

    # Calculate padding for x-axis (allow negative values)
    min_val = valid_scores.min()
    max_val = valid_scores.max()
    val_range = max_val - min_val
    padding = max(val_range * 0.15, 0.05)
    x_min = min_val - padding
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    # Labels and title
    if is_individual_format:
        xlabel = "Deviation from Chance ((Control + Mean Treatment) / 2 - 0.5)"
        title_suffix = "\n(Deviation from chance: positive = better, negative = worse)"
    else:
        xlabel = "Deviation from Chance (Average Accuracy - 0.5)"
        title_suffix = "\n(Deviation from chance: positive = better, negative = worse)"

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    title = "Evaluator Performance: Deviation from Chance"
    title += title_suffix

    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add value labels on bars
    for i, (model, val) in enumerate(valid_scores.items()):
        label_x = val + (padding * 0.02) if val >= 0 else val - (padding * 0.02)
        ax.text(
            label_x,
            i,
            f"{val:.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=9,
        )

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="steelblue", label="Better than chance (≥ 0)"),
        Patch(facecolor="coral", label="Worse than chance (< 0)"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved evaluator deviation plot to: {output_path}")
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
        description="Calculate evaluator performance from recognition accuracy data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to directory containing eval logs (used to determine output path)",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to include (filters and orders results). "
        "Supports -set notation (e.g., --model_names -set dr) or explicit names",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default=None,
        help="Optional custom output directory name (used when combining multiple subsets)",
    )

    args = parser.parse_args()

    # Restore -set from placeholder and expand model sets
    args.model_names = [
        arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names
    ]
    model_order = expand_model_names(args.model_names)
    print(f"Model filter/order: {', '.join(model_order)}\n")

    results_dirs = [Path(d) for d in args.results_dir]

    # Validate all directories exist
    for results_dir in results_dirs:
        if not results_dir.exists():
            print(f"Error: Directory not found: {results_dir}")
            return

    # Use first directory for path derivation
    first_dir = results_dirs[0]

    # Parse path to create matching analysis output path
    parts = first_dir.parts
    if len(parts) >= 4 and parts[0] == "data" and parts[1] == "results":
        dataset_name = parts[2]
        experiment_name = parts[-1]

        if len(results_dirs) == 1:
            relative_path = Path(*parts[2:])
            output_dir = Path("data/analysis") / relative_path
        else:
            if args.output_name:
                subset_name = args.output_name
            else:
                subset_names = []
                for d in results_dirs:
                    d_parts = d.parts
                    if len(d_parts) >= 4:
                        subset_names.append(d_parts[3])
                subset_name = "+".join(subset_names) if subset_names else "combined"

            output_dir = (
                Path("data/analysis") / dataset_name / subset_name / experiment_name
            )
    else:
        output_dir = Path("data/analysis") / first_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create evaluator_performance subdirectory
    performance_dir = output_dir / "evaluator_performance"
    performance_dir.mkdir(parents=True, exist_ok=True)

    # Check for accuracy pivot table
    accuracy_pivot_path = output_dir / "recognition_accuracy" / "accuracy_pivot.csv"

    if not accuracy_pivot_path.exists():
        print("⚠ ERROR: accuracy_pivot.csv not found!")
        print(f"   Expected location: {accuracy_pivot_path}")
        print(
            "\n   Please run recognition_accuracy.py first to generate the pivot table."
        )
        print(
            "   The evaluator performance analysis requires the accuracy pivot table as input.\n"
        )
        return

    # Load pivot table
    print(f"✓ Found accuracy_pivot.csv, loading from: {accuracy_pivot_path}")
    pivot = pd.read_csv(accuracy_pivot_path, index_col=0)

    # Ensure model order is applied
    if model_order:
        row_order = [m for m in model_order if m in pivot.index]
        col_order = [m for m in model_order if m in pivot.columns]

        if not row_order:
            row_order = list(pivot.index)
        if not col_order:
            col_order = list(pivot.columns)

        pivot = pivot.reindex(index=row_order, columns=col_order)

    print(f"  ✓ Loaded pivot table: {pivot.shape[0]} rows × {pivot.shape[1]} columns\n")

    if pivot.empty:
        print("⚠ No data to analyze!")
        return

    # Detect format (individual has diagonal data, pairwise doesn't)
    has_diagonal_data = False
    for model in pivot.index:
        if model in pivot.columns:
            if pd.notna(pivot.loc[model, model]):
                has_diagonal_data = True
                break

    # Extract experiment name for title
    experiment_code = first_dir.name
    if "_" in experiment_code:
        parts = experiment_code.split("_", 1)
        if parts[0].isdigit():
            experiment_code = parts[1]

    experiment_mapping = get_experiment_name_mapping()
    experiment_title = experiment_mapping.get(experiment_code, experiment_code)

    # Compute performance (raw scores 0-1)
    if has_diagonal_data:
        print("Individual format detected: Computing average recognition accuracy\n")
        performance_scores = compute_individual_performance(pivot)
        deviation_scores = compute_individual_deviation(pivot)
    else:
        print(
            "Pairwise format detected: Computing average accuracy across treatments\n"
        )
        performance_scores = compute_pairwise_performance(pivot)
        deviation_scores = compute_pairwise_deviation(pivot)

    # Save performance scores
    performance_path = performance_dir / "evaluator_performance.csv"
    performance_scores.to_frame("performance").to_csv(performance_path)
    print(f"  ✓ Saved performance scores to: {performance_path}")

    # Save deviation scores
    deviation_path = performance_dir / "evaluator_deviation.csv"
    deviation_scores.to_frame("deviation").to_csv(deviation_path)
    print(f"  ✓ Saved deviation scores to: {deviation_path}\n")

    # Generate performance plot (raw values 0-1 with reference line at 0.5)
    plot_path = performance_dir / "evaluator_performance.png"
    plot_evaluator_performance(
        performance_scores,
        plot_path,
        experiment_title=experiment_title,
        is_individual_format=has_diagonal_data,
    )
    print()

    # Generate deviation plot (deviation from chance with pos/neg color coding)
    deviation_plot_path = performance_dir / "evaluator_deviation.png"
    plot_evaluator_deviation(
        deviation_scores,
        deviation_plot_path,
        experiment_title=experiment_title,
        is_individual_format=has_diagonal_data,
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Evaluator Performance (Raw Scores)")
    print(f"{'='*70}\n")
    print(performance_scores.sort_values(ascending=False).round(3))
    print()

    print(f"{'='*70}")
    print("PREVIEW: Evaluator Deviation from Chance")
    print(f"{'='*70}\n")
    print(deviation_scores.sort_values(ascending=False).round(3))
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {performance_dir}")
    print("  • evaluator_performance.csv: Performance scores (0-1 scale)")
    print("  • evaluator_performance.png: Bar chart (raw values with 0.5 reference)")
    print("  • evaluator_deviation.csv: Deviation from chance scores")
    print(
        "  • evaluator_deviation.png: Bar chart (deviation with pos/neg color coding)"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
