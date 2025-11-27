#!/usr/bin/env python3
"""
Compare accuracy pivot table with agreement matrix.

Takes paths to an accuracy pivot CSV and an agreement matrix CSV,
performs comparisons (e.g., adding the matrices), and generates visualizations.

Usage:
    uv run experiments/scripts/compare_accuracy_agreement.py \
        --accuracy_pivot data/analysis/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr/accuracy_pivot.csv \
        --agreement_matrix data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr/agreement_matrix.csv

Output:
    - data/analysis/{dataset}/{subset}/comparisons/{exp1_code}_vs_{exp2_code}/
        - combined_matrix.csv: Sum of matrices
        - combined_heatmap.png: Visualization
        - accuracy_agreement_stats.txt: Comparison statistics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


def get_model_order() -> list[str]:
    """
    Define the canonical order for models in the pivot table and heatmap.

    Models are organized by company/provider, then ordered from weakest to strongest.

    Returns:
        List of model names in display order
    """
    return [
        # OpenAI (weakest to strongest)
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
        "gpt-5.1",
        # Anthropic (weakest to strongest)
        "haiku-3.5",
        "sonnet-3.7",
        "sonnet-4.5",
        "opus-4.1",
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b",
        "ll-3.1-70b",
        "ll-3.1-405b",
        # Together AI - Qwen (weakest to strongest)
        "qwen-2.5-7b",
        "qwen-2.5-72b",
        "qwen-3.0-80b",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.0",
        "deepseek-3.1",
    ]


def get_model_provider(model_name: str) -> str:
    """
    Get the provider/company for a given model name.

    Args:
        model_name: Short model name (e.g., "gpt-4o", "haiku-3.5")

    Returns:
        Provider name (e.g., "OpenAI", "Anthropic", "Google", "Together-Llama", "Together-Qwen", "Together-DeepSeek")
    """
    model_lower = model_name.lower()

    if model_lower.startswith("gpt-"):
        return "OpenAI"
    elif (
        model_lower.startswith("haiku-")
        or model_lower.startswith("sonnet-")
        or model_lower.startswith("opus-")
    ):
        return "Anthropic"
    elif model_lower.startswith("gemini-"):
        return "Google"
    elif model_lower.startswith("ll-"):
        return "Together-Llama"
    elif model_lower.startswith("qwen-"):
        return "Together-Qwen"
    elif model_lower.startswith("deepseek-"):
        return "Together-DeepSeek"
    else:
        return "Unknown"


def add_provider_boundaries(ax, pivot: pd.DataFrame, linewidth: float = 2.5):
    """
    Add thicker lines at boundaries between different providers.

    This draws vertical and horizontal lines to separate provider families
    in the heatmap for better visual organization.

    Args:
        ax: Matplotlib axes object
        pivot: Pivot table DataFrame with models as index and columns
        linewidth: Width of the boundary lines (default: 2.5)
    """
    # Get providers for each model
    row_providers = [get_model_provider(model) for model in pivot.index]
    col_providers = [get_model_provider(model) for model in pivot.columns]

    # Find provider boundaries (where provider changes)
    # Vertical lines (between columns)
    for j in range(len(col_providers) - 1):
        if col_providers[j] != col_providers[j + 1]:
            # Draw vertical line at boundary
            ax.axvline(x=j + 1, color="black", linewidth=linewidth, zorder=15)

    # Horizontal lines (between rows)
    for i in range(len(row_providers) - 1):
        if row_providers[i] != row_providers[i + 1]:
            # Draw horizontal line at boundary
            ax.axhline(y=i + 1, color="black", linewidth=linewidth, zorder=15)


def parse_analysis_path(csv_path: Path) -> tuple[str, str, str] | None:
    """
    Parse analysis CSV path to extract dataset, subset, and experiment code.

    Expected format: data/analysis/{dataset}/{subset}/{experiment_code}/{filename}.csv

    Args:
        csv_path: Path to CSV file

    Returns:
        (dataset, subset, experiment_code) or None if parsing fails
    """
    parts = list(csv_path.parts)
    if "analysis" in parts:
        analysis_idx = parts.index("analysis")
        if analysis_idx + 3 < len(parts):
            dataset = parts[analysis_idx + 1]
            subset = parts[analysis_idx + 2]
            experiment_code = parts[analysis_idx + 3]
            return dataset, subset, experiment_code
    return None


def get_experiment_number_from_path(csv_path: Path) -> str | None:
    """
    Extract experiment number from CSV path.

    Args:
        csv_path: Path to CSV file

    Returns:
        Experiment number (e.g., "11") or None if not found
    """
    parsed = parse_analysis_path(csv_path)
    if parsed:
        _, _, experiment_code = parsed
        if "_" in experiment_code:
            parts = experiment_code.split("_", 1)
            if parts[0].isdigit():
                return parts[0]
    # Fallback: check parent directory name
    parent_name = csv_path.parent.name
    if "_" in parent_name:
        parts = parent_name.split("_", 1)
        if parts[0].isdigit():
            return parts[0]
    return None


def get_experiment_code_from_path(csv_path: Path) -> str:
    """
    Extract experiment code from CSV path.

    Args:
        csv_path: Path to CSV file

    Returns:
        Experiment code (e.g., "UT_PW-Q_Rec_NPr")
    """
    parsed = parse_analysis_path(csv_path)
    if parsed:
        _, _, experiment_code = parsed
        # Remove leading numbers and underscore if present
        if "_" in experiment_code:
            parts = experiment_code.split("_", 1)
            if parts[0].isdigit():
                return parts[1]
        return experiment_code
    # Fallback: use parent directory name
    parent_name = csv_path.parent.name
    if "_" in parent_name:
        parts = parent_name.split("_", 1)
        if parts[0].isdigit():
            return parts[1]
    return parent_name


def load_matrix(csv_path: Path) -> pd.DataFrame:
    """
    Load a matrix from CSV file and reorder according to canonical model order.

    Args:
        csv_path: Path to CSV file

    Returns:
        DataFrame with models reordered
    """
    matrix = pd.read_csv(csv_path, index_col=0)

    # Reorder rows and columns according to canonical model order
    model_order = get_model_order()

    # Filter to only models that exist in the data
    row_order = [m for m in model_order if m in matrix.index]
    col_order = [m for m in model_order if m in matrix.columns]

    # Reindex to apply ordering
    matrix = matrix.reindex(index=row_order, columns=col_order)

    return matrix


def compute_sum(
    accuracy_pivot: pd.DataFrame, agreement_matrix: pd.DataFrame
) -> pd.DataFrame:
    """
    Compute element-wise sum of accuracy pivot and agreement matrix, then subtract 1.

    Args:
        accuracy_pivot: Accuracy pivot table
        agreement_matrix: Agreement matrix

    Returns:
        Combined matrix (sum - 1) with same structure
    """
    print("Computing combined sum ((accuracy_pivot + agreement_matrix) - 1)...")

    # Ensure both matrices are aligned to canonical model order
    model_order = get_model_order()

    # Get union of all models from both matrices, ordered by canonical order
    all_rows = [
        m
        for m in model_order
        if m in accuracy_pivot.index or m in agreement_matrix.index
    ]
    all_cols = [
        m
        for m in model_order
        if m in accuracy_pivot.columns or m in agreement_matrix.columns
    ]

    # Reindex both to canonical order
    accuracy_ordered = accuracy_pivot.reindex(index=all_rows, columns=all_cols)
    agreement_ordered = agreement_matrix.reindex(index=all_rows, columns=all_cols)

    # Compute sum (NaN + value = NaN, so missing values remain NaN)
    combined = accuracy_ordered.add(agreement_ordered, fill_value=0.0)

    # Subtract 1 from all values
    combined = combined - 1.0

    return combined


def plot_combined_heatmap(
    combined_matrix: pd.DataFrame,
    output_path: Path,
    exp1_title: str,
    exp2_title: str,
):
    """
    Create and save heatmap of combined matrix (accuracy + agreement).

    Args:
        combined_matrix: Combined matrix (sum of accuracy and agreement)
        output_path: Path to save the heatmap image
        exp1_title: Title of first experiment (accuracy)
        exp2_title: Title of second experiment (agreement)
    """
    print("Generating combined heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create mask for diagonal (evaluator == treatment) and NaN values
    mask = pd.DataFrame(
        False, index=combined_matrix.index, columns=combined_matrix.columns
    )
    for model in combined_matrix.index:
        if model in combined_matrix.columns:
            mask.loc[model, model] = True
        # Mask NaN values
        for col in combined_matrix.columns:
            if pd.isna(combined_matrix.loc[model, col]):
                mask.loc[model, col] = True

    # Determine value range for colormap (1 = green, 0 = red)
    valid_values = combined_matrix[~mask].values.flatten()
    valid_values = valid_values[~pd.isna(valid_values)]
    if len(valid_values) > 0:
        vmin = float(np.min(valid_values))
        vmax = float(np.max(valid_values))
        # Ensure 1 maps to green, 0 maps to red
        # If max < 1, we still want to use 1 as the upper bound for green
        # If min > 0, we still want to use 0 as the lower bound for red
        vmin = min(vmin, 0.0)  # At least 0 for red
        vmax = max(vmax, 1.0)  # At least 1 for green
    else:
        vmin = 0.0
        vmax = 1.0

    # Create heatmap with 1 = green, 0 = red
    # RdYlGn maps low values to red, high values to green
    sns.heatmap(
        combined_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Red-Yellow-Green (low=red, high=green)
        vmin=vmin,
        vmax=vmax,
        cbar_kws={"label": "Combined Score ((Accuracy + Agreement) - 1)"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Fill diagonal and missing data cells with gray squares
    for i, model in enumerate(combined_matrix.index):
        for j, col in enumerate(combined_matrix.columns):
            if mask.loc[model, col]:  # If masked (diagonal or missing data)
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
    add_provider_boundaries(ax, combined_matrix)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Multi-line title
    title = (
        f"Combined Matrix: (Accuracy + Agreement) - 1\n"
        f"({exp1_title})\n"
        f"PLUS\n"
        f"({exp2_title})"
    )

    ax.set_title(
        title,
        fontsize=13,
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
    print(f"  ✓ Saved combined heatmap to: {output_path}")

    plt.close()


def generate_summary_stats(
    accuracy_pivot: pd.DataFrame,
    agreement_matrix: pd.DataFrame,
    combined_matrix: pd.DataFrame,
    output_path: Path,
    exp1_title: str,
    exp2_title: str,
):
    """Generate and save comparison summary statistics."""
    print("Generating summary statistics...")

    # Flatten and remove NaN/diagonal
    def get_valid_values(matrix):
        values = []
        for i, evaluator in enumerate(matrix.index):
            for j, treatment in enumerate(matrix.columns):
                if evaluator != treatment:  # Skip diagonal
                    val = matrix.iloc[i, j]
                    if pd.notna(val):
                        values.append(val)
        return np.array(values)

    accuracy_vals = get_valid_values(accuracy_pivot)
    agreement_vals = get_valid_values(agreement_matrix)
    combined_vals = get_valid_values(combined_matrix)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("ACCURACY vs AGREEMENT COMPARISON ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("INPUTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Accuracy Matrix: {exp1_title}\n")
        f.write(f"Agreement Matrix: {exp2_title}\n")
        f.write("Combined: (Accuracy + Agreement) - 1\n\n")

        # Accuracy statistics
        if len(accuracy_vals) > 0:
            f.write("ACCURACY MATRIX STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean: {np.mean(accuracy_vals):.3f}\n")
            f.write(f"Median: {np.median(accuracy_vals):.3f}\n")
            f.write(f"Std deviation: {np.std(accuracy_vals):.3f}\n")
            f.write(f"Min: {np.min(accuracy_vals):.3f}\n")
            f.write(f"Max: {np.max(accuracy_vals):.3f}\n")
            f.write(f"Valid cells: {len(accuracy_vals)}\n\n")

        # Agreement statistics
        if len(agreement_vals) > 0:
            f.write("AGREEMENT MATRIX STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean: {np.mean(agreement_vals):.3f}\n")
            f.write(f"Median: {np.median(agreement_vals):.3f}\n")
            f.write(f"Std deviation: {np.std(agreement_vals):.3f}\n")
            f.write(f"Min: {np.min(agreement_vals):.3f}\n")
            f.write(f"Max: {np.max(agreement_vals):.3f}\n")
            f.write(f"Valid cells: {len(agreement_vals)}\n\n")

        # Combined statistics
        if len(combined_vals) > 0:
            f.write("COMBINED MATRIX STATISTICS ((Accuracy + Agreement) - 1)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean: {np.mean(combined_vals):.3f}\n")
            f.write(f"Median: {np.median(combined_vals):.3f}\n")
            f.write(f"Std deviation: {np.std(combined_vals):.3f}\n")
            f.write(f"Min: {np.min(combined_vals):.3f}\n")
            f.write(f"Max: {np.max(combined_vals):.3f}\n")
            f.write(f"Valid cells: {len(combined_vals)}\n\n")

        # Row averages (sorted from largest to smallest)
        f.write("ROW AVERAGES (Evaluator Performance)\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Average combined score for each evaluator model (sorted by average, largest to smallest)\n\n"
        )

        row_averages = []
        evaluator_significance = {}

        # Calculate significance for each evaluator using one-sample t-test
        for evaluator in combined_matrix.index:
            row_vals = []
            for treatment in combined_matrix.columns:
                if evaluator != treatment:  # Skip diagonal
                    val = combined_matrix.loc[evaluator, treatment]
                    if pd.notna(val):
                        row_vals.append(val)
            if row_vals:
                avg = np.mean(row_vals)
                row_averages.append((evaluator, avg))

                # Perform one-sample t-test: test if mean is significantly different from 0
                if len(row_vals) > 1:
                    t_stat, p_val = stats.ttest_1samp(row_vals, 0.0)
                    evaluator_significance[evaluator] = p_val
                else:
                    evaluator_significance[evaluator] = None

        # Sort by average (largest to smallest)
        row_averages.sort(key=lambda x: x[1], reverse=True)

        f.write(f"{'Model':<30} {'Average':<12} {'Significance':<15}\n")
        f.write("-" * 57 + "\n")
        for model, avg in row_averages:
            sig_marker = ""
            if (
                model in evaluator_significance
                and evaluator_significance[model] is not None
            ):
                p_val = evaluator_significance[model]
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"
            f.write(f"{model:<30} {avg:.3f} {sig_marker:<15}\n")
        f.write("\n")
        f.write(
            "Significance: *** p<0.001, ** p<0.01, * p<0.05 (one-sample t-test vs 0)\n\n"
        )

        # Generate evaluator performance plot
        evaluator_plot_path = output_path.parent / "evaluator_performance.png"
        evaluator_series = pd.Series(
            {model: avg for model, avg in row_averages}, name="Combined Score"
        )
        significance_series = pd.Series(evaluator_significance).reindex(
            evaluator_series.index
        )
        plot_evaluator_performance(
            evaluator_series,
            evaluator_plot_path,
            experiment_title=f"{exp1_title} + {exp2_title}",
            ylabel="Average Combined Score ((Accuracy + Agreement) - 1)",
            significance_pvalues=significance_series,
        )

        # Coverage
        f.write("COVERAGE\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Accuracy matrix: {accuracy_pivot.shape[0]} rows × {accuracy_pivot.shape[1]} columns\n"
        )
        f.write(
            f"Agreement matrix: {agreement_matrix.shape[0]} rows × {agreement_matrix.shape[1]} columns\n"
        )
        f.write(
            f"Combined matrix: {combined_matrix.shape[0]} rows × {combined_matrix.shape[1]} columns\n"
        )
        f.write(
            f"Diagonal (N/A): {len([m for m in combined_matrix.index if m in combined_matrix.columns])}\n"
        )
        valid_combined = combined_matrix.notna().sum().sum()
        f.write(f"Valid combined cells: {int(valid_combined)}\n\n")

    print(f"  ✓ Saved summary to: {output_path}")


def get_model_family_colors(model_names: list[str]) -> list[str]:
    """
    Get colors for models based on their family, with lighter shades for weaker models
    and darker shades for stronger models within each family.

    Args:
        model_names: List of model names in order

    Returns:
        List of hex color codes matching the model order
    """
    # Define model families and their base colors (matching logo colors)
    family_colors = {
        "openai": {
            "base": "#10a37f",  # OpenAI green
            "models": ["gpt-4o-mini", "gpt-4.1-mini", "gpt-4o", "gpt-4.1", "gpt-5.1"],
            "shades": [
                "#7dd3b0",
                "#4db896",
                "#2a9d7c",
                "#0d8a6a",
                "#005844",
            ],  # Light to dark
        },
        "anthropic": {
            "base": "#ea580c",  # Claude red-orange
            "models": ["haiku-3.5", "sonnet-3.7", "sonnet-4.5", "opus-4.1"],
            "shades": [
                "#fb923c",
                "#f97316",
                "#ea580c",
                "#c2410c",
            ],  # Light to dark (red-orange)
        },
        "google": {
            "base": "#fbbf24",  # Google yellow
            "models": [
                "gemini-2.0-flash-lite",
                "gemini-2.0-flash",
                "gemini-2.5-flash",
                "gemini-2.5-pro",
            ],
            "shades": [
                "#fef08a",
                "#fde047",
                "#facc15",
                "#eab308",
            ],  # Light to dark yellow
        },
        "llama": {
            "base": "#3b82f6",  # Blue
            "models": ["ll-3.1-8b", "ll-3.1-70b", "ll-3.1-405b"],
            "shades": ["#93c5fd", "#60a5fa", "#3b82f6"],  # Light to dark blue
        },
        "qwen": {
            "base": "#7c3aed",  # Purple
            "models": ["qwen-2.5-7b", "qwen-2.5-72b", "qwen-3.0-80b"],
            "shades": ["#c4b5fd", "#a78bfa", "#7c3aed"],  # Light to dark purple
        },
        "deepseek": {
            "base": "#dc2626",  # Red
            "models": ["deepseek-3.0", "deepseek-3.1"],
            "shades": ["#fca5a5", "#dc2626"],  # Light to dark red
        },
    }

    colors = []
    for model in model_names:
        assigned = False
        for family_name, family_info in family_colors.items():
            if model in family_info["models"]:
                idx = family_info["models"].index(model)
                colors.append(family_info["shades"][idx])
                assigned = True
                break
        if not assigned:
            # Default gray for unknown models
            colors.append("#9ca3af")

    return colors


def plot_evaluator_performance(
    evaluator_avg: pd.Series,
    output_path: Path,
    experiment_title: str = "",
    ylabel: str = "Average Score",
    significance_pvalues: pd.Series | None = None,
):
    """
    Create a bar plot showing evaluator performance averages.

    Args:
        evaluator_avg: Series with evaluator names as index and average values
        output_path: Path to save the plot
        experiment_title: Optional experiment title
        ylabel: Label for y-axis
        significance_pvalues: Optional Series with p-values for significance testing
    """
    print("Generating evaluator performance plot...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Get colors based on model family
    model_list = list(evaluator_avg.index)
    colors = get_model_family_colors(model_list)

    _bars = ax.barh(range(len(evaluator_avg)), evaluator_avg.values, color=colors)

    # Set y-axis labels
    ax.set_yticks(range(len(evaluator_avg)))
    ax.set_yticklabels(evaluator_avg.index)
    ax.invert_yaxis()  # Top to bottom

    # Labels and title
    ax.set_xlabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    title = "Evaluator Performance (Average Across All Treatments)"
    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Calculate padding for x-axis to prevent label overlap
    if len(evaluator_avg) > 0:
        min_val = evaluator_avg.min()
        max_val = evaluator_avg.max()
        val_range = max_val - min_val
        # Add 15% padding on each side
        padding = max(val_range * 0.15, 0.05)  # At least 0.05 padding
        x_min = max(0, min_val - padding)  # Don't go below 0 for positive-only values
        x_max = max_val + padding
        ax.set_xlim(x_min, x_max)

    # Add value labels on bars with significance markers
    for i, (model, val) in enumerate(evaluator_avg.items()):
        # Determine significance marker
        sig_marker = ""
        if significance_pvalues is not None and model in significance_pvalues.index:
            p_val = significance_pvalues[model]
            if pd.notna(p_val):
                if p_val < 0.001:
                    sig_marker = "***"
                elif p_val < 0.01:
                    sig_marker = "**"
                elif p_val < 0.05:
                    sig_marker = "*"

        # Position text on the right side of the bar
        x_pos = val
        ha = "left"
        ax.text(
            x_pos,
            i,
            f" {val:.3f}{sig_marker}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold" if sig_marker else "normal",
        )

    # Add significance legend if any significant values
    if significance_pvalues is not None and any(
        pd.notna(p) and p < 0.05 for p in significance_pvalues.values
    ):
        from matplotlib.patches import Patch

        legend_elements = [
            Patch(facecolor="none", edgecolor="none", label="Significance:"),
            Patch(facecolor="none", edgecolor="none", label="*** p<0.001"),
            Patch(facecolor="none", edgecolor="none", label="** p<0.01"),
            Patch(facecolor="none", edgecolor="none", label="* p<0.05"),
        ]
        ax.legend(
            handles=legend_elements, loc="lower right", fontsize=9, framealpha=0.9
        )

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved evaluator performance plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare accuracy pivot table with agreement matrix"
    )
    parser.add_argument(
        "--accuracy_pivot",
        type=str,
        required=True,
        help="Path to accuracy pivot CSV file",
    )
    parser.add_argument(
        "--agreement_matrix",
        type=str,
        required=True,
        help="Path to agreement matrix CSV file",
    )

    args = parser.parse_args()

    # Convert to Path objects
    accuracy_path = Path(args.accuracy_pivot)
    agreement_path = Path(args.agreement_matrix)

    if not accuracy_path.exists():
        print(f"❌ Error: Accuracy pivot file not found: {accuracy_path}")
        return
    if not agreement_path.exists():
        print(f"❌ Error: Agreement matrix file not found: {agreement_path}")
        return

    # Parse paths to extract experiment info
    accuracy_info = parse_analysis_path(accuracy_path)
    agreement_info = parse_analysis_path(agreement_path)

    # Get experiment codes and numbers
    exp1_code = get_experiment_code_from_path(accuracy_path)
    exp2_code = get_experiment_code_from_path(agreement_path)
    exp1_num = get_experiment_number_from_path(accuracy_path)
    exp2_num = get_experiment_number_from_path(agreement_path)

    # Determine if this is a cross-dataset comparison
    is_cross_dataset = False
    if accuracy_info and agreement_info:
        dataset1, subset1, _ = accuracy_info
        dataset2, subset2, _ = agreement_info
        is_cross_dataset = (dataset1 != dataset2) or (subset1 != subset2)

    # Setup output directory based on comparison type
    if is_cross_dataset:
        # Cross-dataset comparison
        dataset1, subset1, _ = accuracy_info
        dataset2, subset2, _ = agreement_info

        # Use the experiment code (should be the same for both in cross-dataset comparisons)
        experiment_code = (
            exp1_code if exp1_code == exp2_code else f"{exp1_code}_vs_{exp2_code}"
        )

        # Include experiment number in the code if available
        if exp1_num:
            experiment_code = f"{exp1_num}_{experiment_code}"

        # Create comparison name: dataset1_subset1_vs_dataset2_subset2
        comparison_name = f"{dataset1}_{subset1}_vs_{dataset2}_{subset2}"

        output_dir = (
            Path("data/analysis/cross-dataset_comparisons")
            / experiment_code
            / comparison_name
        )
    else:
        # Same dataset/subset comparison
        if accuracy_info:
            dataset, subset, _ = accuracy_info
            # Path: data/analysis/{dataset}/{subset}/comparisons/{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}
            if exp1_num and exp2_num:
                comparison_name = f"{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}"
            else:
                comparison_name = f"{exp1_code}_vs_{exp2_code}"
            output_dir = (
                Path("data/analysis")
                / dataset
                / subset
                / "comparisons"
                / comparison_name
            )
        else:
            # Fallback if path doesn't contain 'analysis'
            if exp1_num and exp2_num:
                comparison_name = f"{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}"
            else:
                comparison_name = f"{exp1_code}_vs_{exp2_code}"
            output_dir = Path("data/analysis/comparisons") / comparison_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("ACCURACY vs AGREEMENT COMPARISON")
    if is_cross_dataset:
        print("(Cross-Dataset Comparison)")
    print(f"{'='*70}")
    print(f"Accuracy Matrix: {exp1_code}")
    print(f"                 {accuracy_path}")
    if is_cross_dataset and accuracy_info:
        print(f"                 Dataset: {accuracy_info[0]}/{accuracy_info[1]}")
    print(f"Agreement Matrix: {exp2_code}")
    print(f"                 {agreement_path}")
    if is_cross_dataset and agreement_info:
        print(f"                 Dataset: {agreement_info[0]}/{agreement_info[1]}")
    print(f"Output dir:       {output_dir}")
    print(f"{'='*70}\n")

    # Load matrices
    print("Loading matrices...")
    accuracy_pivot = load_matrix(accuracy_path)
    print(
        f"  ✓ Loaded accuracy pivot: {accuracy_pivot.shape[0]} rows × {accuracy_pivot.shape[1]} columns"
    )
    agreement_matrix = load_matrix(agreement_path)
    print(
        f"  ✓ Loaded agreement matrix: {agreement_matrix.shape[0]} rows × {agreement_matrix.shape[1]} columns\n"
    )

    # Compute combined sum (subtract 1)
    combined_matrix = compute_sum(accuracy_pivot, agreement_matrix)
    print(
        f"  ✓ Computed combined matrix: {combined_matrix.shape[0]} rows × {combined_matrix.shape[1]} columns\n"
    )

    # Save combined matrix
    combined_csv_path = output_dir / "combined_matrix.csv"
    combined_matrix.to_csv(combined_csv_path)
    print(f"  ✓ Saved combined matrix to: {combined_csv_path}\n")

    # Generate combined heatmap
    heatmap_path = output_dir / "combined_heatmap.png"
    plot_combined_heatmap(combined_matrix, heatmap_path, exp1_code, exp2_code)
    print()

    # Generate summary stats
    summary_path = output_dir / "accuracy_agreement_stats.txt"
    generate_summary_stats(
        accuracy_pivot,
        agreement_matrix,
        combined_matrix,
        summary_path,
        exp1_code,
        exp2_code,
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Combined Matrix ((Accuracy + Agreement) - 1)")
    print(f"{'='*70}\n")
    print(combined_matrix.round(3))
    print()

    print(f"{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print(
        "  • combined_matrix.csv: Combined sum of accuracy and agreement matrices (minus 1)"
    )
    print("  • combined_heatmap.png: Visualization")
    print("  • evaluator_performance.png: Evaluator performance bar chart")
    print("  • accuracy_agreement_stats.txt: Comparison statistics")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
