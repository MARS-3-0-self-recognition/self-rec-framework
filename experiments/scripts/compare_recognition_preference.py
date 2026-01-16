#!/usr/bin/env python3
"""
Compare recognition vs preference experiments with a 4-color heatmap.

This script tests the hypothesis that preference scores are more extreme than
recognition scores. It visualizes the difference (pref - rec) with a custom
4-color scheme based on:
- Green: positive diff when rec > 0.5 (improving from good)
- Blue: positive diff when rec < 0.5 (improving from bad)
- Red: negative diff when rec < 0.5 (worsening from bad)
- Orange: negative diff when rec > 0.5 (worsening from good)

Usage:
    uv run experiments/scripts/compare_recognition_preference.py \
        --recognition data/analysis/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr/accuracy_pivot.csv \
        --preference data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr/accuracy_pivot.csv
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats
from inspect_ai.log import read_eval_log
from src.helpers.model_sets import get_model_set


def get_model_provider(model_name: str) -> str:
    """Get the provider/company for a given model name."""
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
    """Add thicker lines at boundaries between different providers."""
    row_providers = [get_model_provider(model) for model in pivot.index]
    col_providers = [get_model_provider(model) for model in pivot.columns]

    # Vertical lines (between columns)
    for j in range(len(col_providers) - 1):
        if col_providers[j] != col_providers[j + 1]:
            ax.axvline(x=j + 1, color="black", linewidth=linewidth, zorder=15)

    # Horizontal lines (between rows)
    for i in range(len(row_providers) - 1):
        if row_providers[i] != row_providers[i + 1]:
            ax.axhline(y=i + 1, color="black", linewidth=linewidth, zorder=15)


def load_pivot_table(csv_path: Path) -> pd.DataFrame:
    """Load a pivot table and reorder according to canonical model order."""
    matrix = pd.read_csv(csv_path, index_col=0)
    model_order = get_model_set("dr")

    # Filter to only models that exist in the data
    row_order = [m for m in model_order if m in matrix.index]
    col_order = [m for m in model_order if m in matrix.columns]

    # Reindex to apply ordering
    matrix = matrix.reindex(index=row_order, columns=col_order)

    return matrix


def compute_difference(
    rec_pivot: pd.DataFrame, pref_pivot: pd.DataFrame
) -> pd.DataFrame:
    """Compute difference matrix (pref - rec)."""
    print("Computing difference (preference - recognition)...")

    # Ensure both matrices are aligned
    model_order = get_model_set("dr")
    all_rows = [m for m in model_order if m in rec_pivot.index or m in pref_pivot.index]
    all_cols = [
        m for m in model_order if m in rec_pivot.columns or m in pref_pivot.columns
    ]

    # Reindex both to canonical order
    rec_ordered = rec_pivot.reindex(index=all_rows, columns=all_cols)
    pref_ordered = pref_pivot.reindex(index=all_rows, columns=all_cols)

    # Compute difference
    diff = pref_ordered.subtract(rec_ordered, fill_value=0.0)

    return diff


def parse_eval_filename(filename: str) -> tuple[str, str, str] | None:
    """
    Extract evaluator, control, and treatment from eval log filename.

    Expected format: TIMESTAMP_{evaluator}-eval-on-{control}-vs-{treatment}_{UUID}.eval
    """
    try:
        parts = filename.split("_", 1)
        if len(parts) < 2:
            return None

        task_part = parts[1]
        task_name = task_part.rsplit("_", 1)[0]

        if "-eval-on-" not in task_name or "-vs-" not in task_name:
            return None

        evaluator_part, rest = task_name.split("-eval-on-", 1)
        control_part, treatment_part = rest.split("-vs-", 1)

        return evaluator_part, control_part, treatment_part
    except Exception:
        return None


def load_sample_level_results(
    results_dir: Path,
) -> dict[tuple[str, str], list[int | None]]:
    """
    Load sample-level binary results from eval logs.

    Returns dictionary mapping (evaluator, treatment) -> list of outcomes (1/0/None)
    """
    print(f"Loading sample-level results from {results_dir.name}...")

    files_by_pair = {}
    eval_files = list(results_dir.glob("*.eval"))

    for eval_file in eval_files:
        parsed = parse_eval_filename(eval_file.name)
        if not parsed:
            continue

        evaluator, control, treatment = parsed
        key = (evaluator, treatment)

        if key not in files_by_pair:
            files_by_pair[key] = []
        files_by_pair[key].append(eval_file)

    results = {}
    for key, file_list in files_by_pair.items():
        evaluator, treatment = key

        if len(file_list) > 1:
            file_list = sorted(file_list, key=lambda f: f.name, reverse=True)

        for eval_file in file_list:
            try:
                log = read_eval_log(eval_file)

                if log.status != "success":
                    continue

                outcomes = []
                if log.samples:
                    for sample in log.samples:
                        outcome = None
                        if sample.scores:
                            score_obj = sample.scores.get("logprob_scorer")
                            if score_obj is not None and hasattr(score_obj, "value"):
                                score_value = score_obj.value
                                if (
                                    isinstance(score_value, dict)
                                    and "acc" in score_value
                                ):
                                    acc_val = score_value["acc"]
                                    if acc_val == "C":
                                        outcome = 1
                                    elif acc_val == "I":
                                        outcome = 0
                                elif (
                                    score_value == "C"
                                    or score_value == 1
                                    or score_value == 1.0
                                ):
                                    outcome = 1
                                elif (
                                    score_value == "I"
                                    or score_value == 0
                                    or score_value == 0.0
                                ):
                                    outcome = 0
                        outcomes.append(outcome)

                if outcomes:
                    results[key] = outcomes
                    break
            except Exception:
                continue

    print(f"  ✓ Loaded {len(results)} (evaluator, treatment) pairs")
    return results


def compute_paired_ttests(
    results1: dict[tuple[str, str], list[int | None]],
    results2: dict[tuple[str, str], list[int | None]],
    diff: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform paired t-tests comparing sample-level results.

    Returns p-values DataFrame with same structure as diff matrix.
    """
    print("Performing paired t-tests...")

    model_order = get_model_set("dr")
    all_rows = [m for m in model_order if m in diff.index]
    all_cols = [m for m in model_order if m in diff.columns]

    p_values = pd.DataFrame(np.nan, index=all_rows, columns=all_cols)

    tested_cells = 0
    significant_cells = 0

    for evaluator in all_rows:
        for treatment in all_cols:
            if evaluator == treatment:
                continue

            key = (evaluator, treatment)

            if key in results1 and key in results2:
                samples1 = results1[key]
                samples2 = results2[key]

                if len(samples1) == len(samples2) and len(samples1) > 1:
                    valid_pairs = [
                        (s1, s2)
                        for s1, s2 in zip(samples1, samples2)
                        if s1 is not None and s2 is not None
                    ]

                    if len(valid_pairs) < 2:
                        continue

                    valid_samples1, valid_samples2 = zip(*valid_pairs)

                    diffs = np.array(valid_samples1) - np.array(valid_samples2)

                    if np.all(diffs == 0):
                        p_value = 1.0
                    else:
                        t_stat, p_value = stats.ttest_rel(
                            valid_samples1, valid_samples2
                        )

                    p_values.loc[evaluator, treatment] = p_value
                    tested_cells += 1
                    if p_value < 0.05:
                        significant_cells += 1

    print(f"  ✓ Performed {tested_cells} paired t-tests")
    print(f"  ✓ Found {significant_cells} significant differences (p < 0.05)")

    return p_values


def plot_4color_heatmap(
    diff: pd.DataFrame,
    rec_pivot: pd.DataFrame,
    output_path: Path,
    rec_title: str,
    pref_title: str,
    p_values: pd.DataFrame | None = None,
):
    """
    Create a 4-color heatmap based on difference and recognition baseline.

    Colors:
    - Green: diff > 0 AND rec > 0.5 (improving from good)
    - Blue: diff > 0 AND rec < 0.5 (improving from bad)
    - Red: diff < 0 AND rec < 0.5 (worsening from bad)
    - Orange: diff < 0 AND rec > 0.5 (worsening from good)
    """
    print("Generating 4-color heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define colors
    GREEN = "#2ecc71"  # Positive diff, rec > 0.5
    BLUE = "#3498db"  # Positive diff, rec < 0.5
    RED = "#e74c3c"  # Negative diff, rec < 0.5
    ORANGE = "#f39c12"  # Negative diff, rec > 0.5
    GRAY = "#bdc3c7"  # Diagonal/missing

    # Create color matrix
    color_matrix = np.full(diff.shape, GRAY, dtype=object)
    mask = np.zeros(diff.shape, dtype=bool)

    # Process each cell
    for i, evaluator in enumerate(diff.index):
        for j, treatment in enumerate(diff.columns):
            # Mask diagonal
            if evaluator == treatment:
                mask[i, j] = True
                continue

            rec_val = rec_pivot.loc[evaluator, treatment]
            diff_val = diff.loc[evaluator, treatment]

            # Skip NaN values
            if pd.isna(rec_val) or pd.isna(diff_val):
                mask[i, j] = True
                continue

            # Determine color based on rec baseline and diff sign
            if diff_val > 0:  # Positive difference
                if rec_val > 0.5:
                    color_matrix[i, j] = GREEN
                else:
                    color_matrix[i, j] = BLUE
            elif diff_val < 0:  # Negative difference
                if rec_val < 0.5:
                    color_matrix[i, j] = RED
                else:
                    color_matrix[i, j] = ORANGE
            else:  # diff == 0
                # Use a neutral color (light gray)
                color_matrix[i, j] = "#ecf0f1"

    # Draw colored rectangles
    for i in range(len(diff.index)):
        for j in range(len(diff.columns)):
            if not mask[i, j]:
                rect = mpatches.Rectangle(
                    (j, i),
                    1,
                    1,
                    facecolor=color_matrix[i, j],
                    edgecolor="gray",
                    linewidth=0.5,
                    zorder=1,
                )
                ax.add_patch(rect)

                # Add annotation with difference value
                diff_val = diff.iloc[i, j]
                evaluator = diff.index[i]
                treatment = diff.columns[j]

                # Check if significant
                is_significant = False
                if p_values is not None and pd.notna(
                    p_values.loc[evaluator, treatment]
                ):
                    is_significant = p_values.loc[evaluator, treatment] < 0.05

                # Always use black text, bold for significant differences
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    f"{diff_val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9 if is_significant else 8,
                    color="black",
                    fontweight="bold" if is_significant else "normal",
                    zorder=2,
                )

    # Fill diagonal and missing data cells with gray
    for i, model in enumerate(diff.index):
        for j, col in enumerate(diff.columns):
            if mask[i, j]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (j, i),
                        1,
                        1,
                        facecolor=GRAY,
                        edgecolor="gray",
                        linewidth=0.5,
                        zorder=1,
                    )
                )
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                    zorder=2,
                )

    # Add provider boundaries
    add_provider_boundaries(ax, diff)

    # Set axis limits and labels
    ax.set_xlim(0, len(diff.columns))
    ax.set_ylim(len(diff.index), 0)
    ax.set_xticks(np.arange(len(diff.columns)) + 0.5)
    ax.set_xticklabels(diff.columns, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(diff.index)) + 0.5)
    ax.set_yticklabels(diff.index)
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Title
    title = (
        f"Preference vs Recognition: (Pref - Rec)\n"
        f"Recognition: {rec_title}\n"
        f"Preference: {pref_title}"
    )
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(
            facecolor=GREEN, label="Pref > Rec, Rec > 0.5 (Improving from good)"
        ),
        mpatches.Patch(
            facecolor=BLUE, label="Pref > Rec, Rec < 0.5 (Improving from bad)"
        ),
        mpatches.Patch(
            facecolor=ORANGE, label="Pref < Rec, Rec > 0.5 (Worsening from good)"
        ),
        mpatches.Patch(
            facecolor=RED, label="Pref < Rec, Rec < 0.5 (Worsening from bad)"
        ),
    ]
    ax.legend(
        handles=legend_elements, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9
    )

    # Add note about bold values
    if p_values is not None:
        fig.text(
            0.5,
            0.02,
            "Bold values indicate statistically significant differences (p < 0.05)",
            ha="center",
            fontsize=10,
            style="italic",
            bbox=dict(
                boxstyle="round,pad=0.5",
                facecolor="lightyellow",
                edgecolor="gray",
                linewidth=1,
            ),
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved heatmap to: {output_path}")
    plt.close()


def parse_analysis_path(csv_path: Path) -> tuple[str, str, str] | None:
    """Parse analysis CSV path to extract dataset, subset, and experiment code."""
    parts = list(csv_path.parts)
    if "analysis" in parts:
        analysis_idx = parts.index("analysis")
        if analysis_idx + 3 < len(parts):
            dataset = parts[analysis_idx + 1]
            subset = parts[analysis_idx + 2]
            experiment_code = parts[analysis_idx + 3]
            return dataset, subset, experiment_code
    return None


def get_experiment_code_from_path(csv_path: Path) -> str:
    """Extract experiment code from CSV path."""
    parsed = parse_analysis_path(csv_path)
    if parsed:
        _, _, experiment_code = parsed
        # Remove leading numbers and underscore if present
        if "_" in experiment_code:
            parts = experiment_code.split("_", 1)
            if parts[0].isdigit():
                return parts[1]
        return experiment_code
    return csv_path.parent.name


def get_experiment_number_from_path(csv_path: Path) -> str | None:
    """Extract experiment number from CSV path."""
    parsed = parse_analysis_path(csv_path)
    if parsed:
        _, _, experiment_code = parsed
        if "_" in experiment_code:
            parts = experiment_code.split("_", 1)
            if parts[0].isdigit():
                return parts[0]
    parent_name = csv_path.parent.name
    if "_" in parent_name:
        parts = parent_name.split("_", 1)
        if parts[0].isdigit():
            return parts[0]
    return None


def plot_color_categorization(
    green_count: int,
    blue_count: int,
    orange_count: int,
    red_count: int,
    zero_count: int,
    green_sig: int,
    blue_sig: int,
    orange_sig: int,
    red_sig: int,
    output_path: Path,
    rec_title: str,
    pref_title: str,
):
    """
    Create a bar plot showing color categorization statistics.

    Args:
        green_count: Number of green cells (Pref > Rec, Rec > 0.5)
        blue_count: Number of blue cells (Pref > Rec, Rec < 0.5)
        orange_count: Number of orange cells (Pref < Rec, Rec > 0.5)
        red_count: Number of red cells (Pref < Rec, Rec < 0.5)
        zero_count: Number of zero difference cells
        green_sig: Number of significant green cells
        blue_sig: Number of significant blue cells
        orange_sig: Number of significant orange cells
        red_sig: Number of significant red cells
        output_path: Path to save the plot
        rec_title: Recognition experiment title
        pref_title: Preference experiment title
    """
    print("Generating color categorization plot...")

    # Define colors (matching the heatmap)
    GREEN = "#2ecc71"
    BLUE = "#3498db"
    RED = "#e74c3c"
    ORANGE = "#f39c12"
    GRAY = "#bdc3c7"

    # Prepare data
    categories = [
        "Green\n(Pref > Rec,\nRec > 0.5)",
        "Blue\n(Pref > Rec,\nRec < 0.5)",
        "Orange\n(Pref < Rec,\nRec > 0.5)",
        "Red\n(Pref < Rec,\nRec < 0.5)",
        "Zero\nDifference",
    ]
    counts = [green_count, blue_count, orange_count, red_count, zero_count]
    sig_counts = [
        green_sig,
        blue_sig,
        orange_sig,
        red_sig,
        0,
    ]  # Zero has no significance
    colors = [GREEN, BLUE, ORANGE, RED, GRAY]

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Create bars with stacked segments for significant counts
    x_pos = np.arange(len(categories))

    # Calculate significant and non-significant portions
    sig_heights = [min(sig, count) for sig, count in zip(sig_counts, counts)]
    non_sig_counts = [max(0, count - sig) for count, sig in zip(counts, sig_counts)]

    # Create darker colors for significant portion
    darker_colors = []
    for color in colors:
        if color == GREEN:
            darker_colors.append("#27ae60")
        elif color == BLUE:
            darker_colors.append("#2980b9")
        elif color == ORANGE:
            darker_colors.append("#d68910")
        elif color == RED:
            darker_colors.append("#c0392b")
        else:
            darker_colors.append(color)

    # Draw significant portion first (bottom)
    _bars_sig = ax.bar(
        x_pos,
        sig_heights,
        bottom=0,
        color=darker_colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
        label="Significant (p < 0.05)",
    )

    # Draw non-significant portion on top
    _bars_non_sig = ax.bar(
        x_pos,
        non_sig_counts,
        bottom=sig_heights,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
        label="Non-significant",
    )

    # Add value labels on bars
    total = sum(counts)
    for i, (count, sig_count) in enumerate(zip(counts, sig_counts)):
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            label = f"{count}\n({percentage:.1f}%)"
            if sig_count > 0:
                label += f"\n{sig_count} sig"
            # Position label at top of bar
            ax.text(
                i,
                count,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    # Expand y-axis with padding
    max_count = max(counts) if counts else 0
    y_padding = max(max_count * 0.1, 5)  # 10% padding or at least 5 units
    ax.set_ylim(0, max_count + y_padding)

    # Set labels and title
    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Cells", fontsize=12, fontweight="bold")

    title = "Color Categorization Statistics"
    if rec_title and pref_title:
        title = f"{title}\n{rec_title} vs {pref_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Set x-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)

    # Add grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    # Add legend for significant cells
    if any(sig_counts):
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved color categorization plot to: {output_path}")
    plt.close()


def generate_summary_stats(
    diff: pd.DataFrame,
    rec_pivot: pd.DataFrame,
    output_path: Path,
    rec_title: str,
    pref_title: str,
    p_values: pd.DataFrame | None = None,
):
    """Generate summary statistics."""
    print("Generating summary statistics...")

    # Categorize cells
    green_count = 0
    blue_count = 0
    red_count = 0
    orange_count = 0
    zero_count = 0

    # Count significant cells per category
    green_sig = 0
    blue_sig = 0
    red_sig = 0
    orange_sig = 0

    green_diffs = []
    blue_diffs = []
    red_diffs = []
    orange_diffs = []

    for evaluator in diff.index:
        for treatment in diff.columns:
            if evaluator == treatment:
                continue

            rec_val = rec_pivot.loc[evaluator, treatment]
            diff_val = diff.loc[evaluator, treatment]

            if pd.isna(rec_val) or pd.isna(diff_val):
                continue

            # Check if significant
            is_significant = False
            if p_values is not None and pd.notna(p_values.loc[evaluator, treatment]):
                is_significant = p_values.loc[evaluator, treatment] < 0.05

            if diff_val > 0:
                if rec_val > 0.5:
                    green_count += 1
                    green_diffs.append(diff_val)
                    if is_significant:
                        green_sig += 1
                else:
                    blue_count += 1
                    blue_diffs.append(diff_val)
                    if is_significant:
                        blue_sig += 1
            elif diff_val < 0:
                if rec_val < 0.5:
                    red_count += 1
                    red_diffs.append(diff_val)
                    if is_significant:
                        red_sig += 1
                else:
                    orange_count += 1
                    orange_diffs.append(diff_val)
                    if is_significant:
                        orange_sig += 1
            else:
                zero_count += 1

    total = green_count + blue_count + red_count + orange_count + zero_count

    # Generate color categorization plot
    categorization_plot_path = output_path.parent / "color_categorization.png"
    plot_color_categorization(
        green_count,
        blue_count,
        orange_count,
        red_count,
        zero_count,
        green_sig,
        blue_sig,
        orange_sig,
        red_sig,
        categorization_plot_path,
        rec_title,
        pref_title,
    )

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RECOGNITION vs PREFERENCE COMPARISON ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("INPUTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Recognition: {rec_title}\n")
        f.write(f"Preference: {pref_title}\n")
        f.write("Difference: Preference - Recognition\n\n")

        f.write("COLOR CATEGORIZATION\n")
        f.write("-" * 70 + "\n")
        f.write(
            f"Green (Pref > Rec, Rec > 0.5): {green_count} cells ({green_count/total*100:.1f}%)\n"
        )
        if p_values is not None and green_count > 0:
            f.write(
                f"  Significant: {green_sig} cells ({green_sig/green_count*100:.1f}% of green)\n"
            )
        if green_diffs:
            f.write(f"  Mean diff: {np.mean(green_diffs):.3f}\n")
            f.write(f"  Median diff: {np.median(green_diffs):.3f}\n")

        f.write(
            f"\nBlue (Pref > Rec, Rec < 0.5): {blue_count} cells ({blue_count/total*100:.1f}%)\n"
        )
        if p_values is not None and blue_count > 0:
            f.write(
                f"  Significant: {blue_sig} cells ({blue_sig/blue_count*100:.1f}% of blue)\n"
            )
        if blue_diffs:
            f.write(f"  Mean diff: {np.mean(blue_diffs):.3f}\n")
            f.write(f"  Median diff: {np.median(blue_diffs):.3f}\n")

        f.write(
            f"\nOrange (Pref < Rec, Rec > 0.5): {orange_count} cells ({orange_count/total*100:.1f}%)\n"
        )
        if p_values is not None and orange_count > 0:
            f.write(
                f"  Significant: {orange_sig} cells ({orange_sig/orange_count*100:.1f}% of orange)\n"
            )
        if orange_diffs:
            f.write(f"  Mean diff: {np.mean(orange_diffs):.3f}\n")
            f.write(f"  Median diff: {np.median(orange_diffs):.3f}\n")

        f.write(
            f"\nRed (Pref < Rec, Rec < 0.5): {red_count} cells ({red_count/total*100:.1f}%)\n"
        )
        if p_values is not None and red_count > 0:
            f.write(
                f"  Significant: {red_sig} cells ({red_sig/red_count*100:.1f}% of red)\n"
            )
        if red_diffs:
            f.write(f"  Mean diff: {np.mean(red_diffs):.3f}\n")
            f.write(f"  Median diff: {np.median(red_diffs):.3f}\n")

        f.write(
            f"\nZero difference: {zero_count} cells ({zero_count/total*100:.1f}%)\n"
        )
        f.write(f"Total valid cells: {total}\n\n")

        # Overall statistics
        all_diffs = [
            d
            for diffs in [green_diffs, blue_diffs, red_diffs, orange_diffs]
            for d in diffs
        ]
        if all_diffs:
            f.write("OVERALL DIFFERENCE STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean difference: {np.mean(all_diffs):.3f}\n")
            f.write(f"Median difference: {np.median(all_diffs):.3f}\n")
            f.write(f"Std deviation: {np.std(all_diffs):.3f}\n")
            f.write(f"Min: {np.min(all_diffs):.3f}\n")
            f.write(f"Max: {np.max(all_diffs):.3f}\n")
        f.write(
            f"Positive differences: {sum(1 for d in all_diffs if d > 0)} ({sum(1 for d in all_diffs if d > 0)/len(all_diffs)*100:.1f}%)\n"
        )
        f.write(
            f"Negative differences: {sum(1 for d in all_diffs if d < 0)} ({sum(1 for d in all_diffs if d < 0)/len(all_diffs)*100:.1f}%)\n\n"
        )

        # Statistical significance summary
        if p_values is not None:
            sig_count = 0
            total_tests = 0
            for evaluator in p_values.index:
                for treatment in p_values.columns:
                    if evaluator != treatment:
                        p_val = p_values.loc[evaluator, treatment]
                        if pd.notna(p_val):
                            total_tests += 1
                            if p_val < 0.05:
                                sig_count += 1

            if total_tests > 0:
                f.write("STATISTICAL SIGNIFICANCE\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total paired t-tests: {total_tests}\n")
                f.write(
                    f"Significant differences (p < 0.05): {sig_count} ({sig_count/total_tests*100:.1f}%)\n"
                )
                f.write(
                    f"Non-significant: {total_tests - sig_count} ({(total_tests-sig_count)/total_tests*100:.1f}%)\n"
                )
                f.write(
                    "Note: Bold values in heatmap indicate significant differences\n"
                )

    print(f"  ✓ Saved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare recognition vs preference experiments with 4-color heatmap"
    )
    parser.add_argument(
        "--recognition",
        type=str,
        required=True,
        help="Path to recognition accuracy pivot CSV file",
    )
    parser.add_argument(
        "--preference",
        type=str,
        required=True,
        help="Path to preference accuracy pivot CSV file",
    )

    args = parser.parse_args()

    # Convert to Path objects
    rec_path = Path(args.recognition)
    pref_path = Path(args.preference)

    if not rec_path.exists():
        print(f"❌ Error: Recognition pivot file not found: {rec_path}")
        return
    if not pref_path.exists():
        print(f"❌ Error: Preference pivot file not found: {pref_path}")
        return

    # Parse paths to extract experiment info
    rec_info = parse_analysis_path(rec_path)
    pref_info = parse_analysis_path(pref_path)

    # Get experiment codes and numbers
    rec_code = get_experiment_code_from_path(rec_path)
    pref_code = get_experiment_code_from_path(pref_path)
    rec_num = get_experiment_number_from_path(rec_path)
    pref_num = get_experiment_number_from_path(pref_path)

    # Convert analysis paths to results directory paths for statistical testing
    # data/analysis/{dataset}/{subset}/{exp_code}/accuracy_pivot.csv
    # -> data/results/{dataset}/{subset}/{exp_code}/
    rec_results_dir = None
    pref_results_dir = None

    if rec_info:
        dataset, subset, exp_code = rec_info
        rec_results_dir = Path("data/results") / dataset / subset / exp_code
    if pref_info:
        dataset, subset, exp_code = pref_info
        pref_results_dir = Path("data/results") / dataset / subset / exp_code

    # Determine if this is a cross-dataset comparison
    is_cross_dataset = False
    if rec_info and pref_info:
        dataset1, subset1, _ = rec_info
        dataset2, subset2, _ = pref_info
        is_cross_dataset = (dataset1 != dataset2) or (subset1 != subset2)

    # Setup output directory
    if is_cross_dataset:
        dataset1, subset1, _ = rec_info
        dataset2, subset2, _ = pref_info
        experiment_code = (
            rec_code if rec_code == pref_code else f"{rec_code}_vs_{pref_code}"
        )
        if rec_num:
            experiment_code = f"{rec_num}_{experiment_code}"
        comparison_name = f"{dataset1}_{subset1}_vs_{dataset2}_{subset2}"
        output_dir = (
            Path("data/analysis/cross-dataset_comparisons")
            / experiment_code
            / comparison_name
        )
    else:
        if rec_info:
            dataset, subset, _ = rec_info
            if rec_num and pref_num:
                comparison_name = f"{rec_num}_{rec_code}_vs_{pref_num}_{pref_code}"
            else:
                comparison_name = f"{rec_code}_vs_{pref_code}"
            output_dir = (
                Path("data/analysis")
                / dataset
                / subset
                / "comparisons"
                / comparison_name
            )
        else:
            if rec_num and pref_num:
                comparison_name = f"{rec_num}_{rec_code}_vs_{pref_num}_{pref_code}"
            else:
                comparison_name = f"{rec_code}_vs_{pref_code}"
            output_dir = Path("data/analysis/comparisons") / comparison_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("RECOGNITION vs PREFERENCE COMPARISON")
    if is_cross_dataset:
        print("(Cross-Dataset Comparison)")
    print(f"{'='*70}")
    print(f"Recognition: {rec_code}")
    print(f"             {rec_path}")
    print(f"Preference: {pref_code}")
    print(f"            {pref_path}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}\n")

    # Load pivot tables
    print("Loading pivot tables...")
    rec_pivot = load_pivot_table(rec_path)
    print(
        f"  ✓ Loaded recognition: {rec_pivot.shape[0]} rows × {rec_pivot.shape[1]} columns"
    )
    pref_pivot = load_pivot_table(pref_path)
    print(
        f"  ✓ Loaded preference: {pref_pivot.shape[0]} rows × {pref_pivot.shape[1]} columns\n"
    )

    # Compute difference
    diff = compute_difference(rec_pivot, pref_pivot)
    print(f"  ✓ Computed differences: {diff.shape[0]} rows × {diff.shape[1]} columns\n")

    # Load sample-level results for statistical testing
    p_values = None
    if (
        rec_results_dir
        and rec_results_dir.exists()
        and pref_results_dir
        and pref_results_dir.exists()
    ):
        print("Loading sample-level data for statistical tests...\n")
        rec_results = load_sample_level_results(rec_results_dir)
        pref_results = load_sample_level_results(pref_results_dir)
        print()

        # Perform paired t-tests
        p_values = compute_paired_ttests(rec_results, pref_results, diff)
        print()

        # Save p-values matrix
        pvalues_csv_path = output_dir / "pvalues.csv"
        p_values.to_csv(pvalues_csv_path)
        print(f"  ✓ Saved p-values matrix to: {pvalues_csv_path}\n")
    else:
        print("  ⚠ Results directories not found, skipping statistical tests\n")
        if rec_results_dir:
            print(
                f"     Recognition: {rec_results_dir} (exists: {rec_results_dir.exists()})"
            )
        if pref_results_dir:
            print(
                f"     Preference: {pref_results_dir} (exists: {pref_results_dir.exists()})\n"
            )

    # Save difference matrix
    diff_csv_path = output_dir / "preference_recognition_difference.csv"
    diff.to_csv(diff_csv_path)
    print(f"  ✓ Saved difference matrix to: {diff_csv_path}\n")

    # Generate 4-color heatmap
    heatmap_path = output_dir / "preference_recognition_heatmap.png"
    plot_4color_heatmap(diff, rec_pivot, heatmap_path, rec_code, pref_code, p_values)
    print()

    # Generate summary stats
    summary_path = output_dir / "recognition_preference_stats.txt"
    generate_summary_stats(diff, rec_pivot, summary_path, rec_code, pref_code, p_values)
    print()

    print(f"{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • preference_recognition_difference.csv: Difference matrix")
    print("  • preference_recognition_heatmap.png: 4-color visualization")
    print("  • color_categorization.png: Color category statistics")
    print("  • recognition_preference_stats.txt: Comparison statistics")
    if p_values is not None:
        print("  • pvalues.csv: Statistical significance p-values")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
