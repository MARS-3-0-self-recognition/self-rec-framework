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
import seaborn as sns
from utils import expand_model_names, get_model_family_colors, add_provider_boundaries
from src.helpers.model_names import LM_ARENA_RANKINGS


def get_model_arena_ranking(model_name: str) -> int | None:
    """Get model LM Arena ranking (lower is better)."""
    # Try exact match first
    if model_name in LM_ARENA_RANKINGS:
        return LM_ARENA_RANKINGS[model_name]

    # Try without -thinking suffix
    base_name = model_name.replace("-thinking", "")
    if base_name in LM_ARENA_RANKINGS:
        return LM_ARENA_RANKINGS[base_name]

    # Try without _fw suffix
    if base_name.endswith("_fw"):
        name_without_fw = base_name[:-3]
        if name_without_fw in LM_ARENA_RANKINGS:
            return LM_ARENA_RANKINGS[name_without_fw]

    return None


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


def plot_heatmap_ranked(
    pivot: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
):
    """
    Create heatmap of self-recognition accuracy ordered by LM Arena ranking.
    
    Both rows (evaluators) and columns (comparison models) are ordered by ranking,
    with lowest ranked models (worst) appearing first.
    
    Args:
        pivot: Pivot table with evaluators as rows, treatments as columns
        output_path: Path to save the heatmap image
        experiment_title: Optional experiment name to include in the title
    """
    print("Generating ranked heatmap...")

    # Get rankings for all models
    def get_rank_or_inf(model_name: str) -> float:
        rank = get_model_arena_ranking(model_name)
        # Use negative rank so highest rank numbers (worst models) sort first
        # For models without ranking, use -inf so they appear last
        return -float(rank) if rank is not None else float('-inf')
    
    # Sort rows (evaluators) by ranking (worst/highest rank number first)
    row_ranks = {model: get_rank_or_inf(model) for model in pivot.index}
    sorted_rows = sorted(pivot.index, key=lambda x: row_ranks[x], reverse=True)
    
    # Sort columns (comparison models) by ranking (worst/highest rank number first)
    col_ranks = {model: get_rank_or_inf(model) for model in pivot.columns}
    sorted_cols = sorted(pivot.columns, key=lambda x: col_ranks[x], reverse=True)
    
    # Reorder pivot table
    pivot_ranked = pivot.reindex(index=sorted_rows, columns=sorted_cols)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Check if diagonal has data (individual format) or is empty (pairwise format)
    has_diagonal_data = False
    for model in pivot_ranked.index:
        if model in pivot_ranked.columns:
            if pd.notna(pivot_ranked.loc[model, model]):
                has_diagonal_data = True
                break

    # Create mask for diagonal only if it's pairwise format (no diagonal data)
    mask = pd.DataFrame(False, index=pivot_ranked.index, columns=pivot_ranked.columns)
    if not has_diagonal_data:
        # Pairwise format: mask diagonal (evaluator == treatment doesn't exist)
        for model in pivot_ranked.index:
            if model in pivot_ranked.columns:
                mask.loc[model, model] = True

    # Create heatmap
    sns.heatmap(
        pivot_ranked,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Self-Recognition Accuracy"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Fill diagonal with gray only for pairwise format (when diagonal has no data)
    if not has_diagonal_data:
        for i, model in enumerate(pivot_ranked.index):
            if model in pivot_ranked.columns:
                j = list(pivot_ranked.columns).index(model)
                ax.add_patch(
                    plt.Rectangle((j, i), 1, 1, fill=True, color="lightgray", zorder=10)
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    # Add thicker lines at provider boundaries
    add_provider_boundaries(ax, pivot_ranked)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)\n(Ordered by LM Arena Rank: Worst → Best)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model\n(Ordered by LM Arena Rank: Worst → Best)", fontsize=12, fontweight="bold")

    # Build title with optional experiment name
    if experiment_title:
        title = f"Self-Recognition Accuracy Matrix (Ranked): {experiment_title}\n(How well each model identifies its own outputs vs. others, ordered by LM Arena ranking)"
    else:
        title = "Self-Recognition Accuracy Matrix (Ranked)\n(How well each model identifies its own outputs vs. others, ordered by LM Arena ranking)"

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved ranked heatmap to: {output_path}")

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

    # Try to load counts table for error bars
    counts_pivot_path = output_dir / "recognition_accuracy" / "accuracy_counts.csv"
    pivot_counts = None
    if counts_pivot_path.exists():
        print(f"✓ Found accuracy_counts.csv, loading from: {counts_pivot_path}")
        pivot_counts = pd.read_csv(counts_pivot_path, index_col=0)
        # Ensure same order as pivot
        if model_order:
            row_order = [m for m in model_order if m in pivot_counts.index]
            col_order = [m for m in model_order if m in pivot_counts.columns]
            if row_order:
                pivot_counts = pivot_counts.reindex(index=row_order, columns=col_order)
        print(f"  ✓ Loaded counts table: {pivot_counts.shape[0]} rows × {pivot_counts.shape[1]} columns\n")
    else:
        print(f"  ⚠ accuracy_counts.csv not found (error bars will be unavailable)\n")

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

    # Compute performance (raw scores 0-1) and total sample counts
    if has_diagonal_data:
        print("Individual format detected: Computing average recognition accuracy\n")
        performance_scores = compute_individual_performance(pivot)
        deviation_scores = compute_individual_deviation(pivot)
        # Compute total n_samples per evaluator for individual format
        # n_samples = n_samples(C_j) + sum(n_samples(T_i))
        total_counts = None
        if pivot_counts is not None:
            total_counts = pd.Series(dtype=float, index=pivot.index)
            for evaluator in pivot.index:
                total_n = 0
                # Add control (diagonal) count
                if evaluator in pivot_counts.index and evaluator in pivot_counts.columns:
                    control_n = pivot_counts.loc[evaluator, evaluator]
                    if pd.notna(control_n):
                        total_n += control_n
                # Add treatment counts (off-diagonal)
                if evaluator in pivot_counts.index:
                    treatment_n = pivot_counts.loc[evaluator].sum() - (pivot_counts.loc[evaluator, evaluator] if evaluator in pivot_counts.columns else 0)
                    total_n += treatment_n
                total_counts.loc[evaluator] = total_n if total_n > 0 else pd.NA
    else:
        print(
            "Pairwise format detected: Computing average accuracy across treatments\n"
        )
        performance_scores = compute_pairwise_performance(pivot)
        deviation_scores = compute_pairwise_deviation(pivot)
        # Compute total n_samples per evaluator for pairwise format
        # n_samples = sum(n_samples across all treatments)
        total_counts = None
        if pivot_counts is not None:
            total_counts = pivot_counts.sum(axis=1)

    # Save performance scores with counts
    performance_path = performance_dir / "evaluator_performance.csv"
    perf_df = performance_scores.to_frame("performance")
    if total_counts is not None:
        perf_df["n_samples"] = total_counts
    perf_df.to_csv(performance_path)
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

    # Generate ranked heatmap (ordered by LM Arena ranking)
    heatmap_ranked_path = performance_dir / "accuracy_heatmap_ranked.png"
    plot_heatmap_ranked(
        pivot,
        heatmap_ranked_path,
        experiment_title=experiment_title,
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
    print(
        "  • accuracy_heatmap_ranked.png: Heatmap ordered by LM Arena ranking (lowest ranked first)"
    )
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
