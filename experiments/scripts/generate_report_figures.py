#!/usr/bin/env python3
"""
Meta script to generate all figures for Performance Report #1.

This script:
1. Runs all necessary analyses using existing scripts
2. Copies figures to a report_1 folder (while keeping originals)
3. Documents figure locations for the report

Usage:
    uv run experiments/scripts/generate_report_figures.py \\
        --recognition_pivot data/analysis/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr/accuracy_pivot.csv \\
        --preference_pivot data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr/accuracy_pivot.csv \\
        --preference_results data/results/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr \\
        --unprimed_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \\
        --primed_results data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr \\
        --alignment_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \\
        --summary_results data/results/wikisum/training_set_1-20/11_UT_PW-Q_Rec_NPr
"""

import argparse
import subprocess
import sys
from pathlib import Path
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats

# Import plotting functions from existing scripts
# We'll import them dynamically to avoid circular dependencies


def run_script(script_name: str, args: list[str]) -> bool:
    """Run an analysis script and return success status."""
    scripts_dir = Path(__file__).parent
    script_path = scripts_dir / script_name
    cmd = ["uv", "run", "python3", str(script_path)] + args
    print(f"\n{'='*70}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    result = subprocess.run(cmd)
    return result.returncode == 0


def copy_figure(source: Path, dest: Path, figure_name: str) -> bool:
    """Copy a figure file to the report directory."""
    if not source.exists():
        print(f"⚠ Warning: Source figure not found: {source}")
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"  ✓ Copied {figure_name} to: {dest}")
    return True


def regenerate_figure_1_with_report_titles(
    comparison_dir: Path,
    report_dir: Path,
    rec_pivot_path: Path,
    pref_pivot_path: Path,
) -> Path | None:
    """Regenerate Figure 1 with report-appropriate titles and labels."""
    print("Regenerating Figure 1 with report titles...")

    # Import necessary functions
    sys.path.insert(0, str(Path(__file__).parent))
    from compare_recognition_preference import (
        load_pivot_table,
        compute_difference,
    )

    if not rec_pivot_path.exists() or not pref_pivot_path.exists():
        print("⚠ Could not find pivot files, using existing figure")
        source = comparison_dir / "color_categorization.png"
        dest = report_dir / "figure_1_color_categorization.png"
        return copy_figure(source, dest, "Figure 1")

    rec_pivot = load_pivot_table(rec_pivot_path)
    pref_pivot = load_pivot_table(pref_pivot_path)
    diff = compute_difference(rec_pivot, pref_pivot)

    # Load p-values if available
    pvalues_path = comparison_dir / "pvalues.csv"
    p_values = None
    if pvalues_path.exists():
        p_values = pd.read_csv(pvalues_path, index_col=0)

    # Count categories
    green_count = blue_count = orange_count = red_count = zero_count = 0
    green_sig = blue_sig = orange_sig = red_sig = 0

    for evaluator in diff.index:
        for treatment in diff.columns:
            if evaluator == treatment:
                continue
            rec_val = rec_pivot.loc[evaluator, treatment]
            diff_val = diff.loc[evaluator, treatment]
            if pd.isna(rec_val) or pd.isna(diff_val):
                continue

            is_significant = False
            if p_values is not None and pd.notna(p_values.loc[evaluator, treatment]):
                is_significant = p_values.loc[evaluator, treatment] < 0.05

            if diff_val > 0:
                if rec_val > 0.5:
                    green_count += 1
                    if is_significant:
                        green_sig += 1
                else:
                    blue_count += 1
                    if is_significant:
                        blue_sig += 1
            elif diff_val < 0:
                if rec_val < 0.5:
                    red_count += 1
                    if is_significant:
                        red_sig += 1
                else:
                    orange_count += 1
                    if is_significant:
                        orange_sig += 1
            else:
                zero_count += 1

    # Recreate the plot with report-appropriate title and labels
    dest = report_dir / "figure_1_color_categorization.png"

    GREEN = "#2ecc71"
    BLUE = "#3498db"
    RED = "#e74c3c"
    ORANGE = "#f39c12"
    GRAY = "#bdc3c7"

    categories = [
        "Green\n(Preference > Recognition,\nRecognition > 0.5)",
        "Blue\n(Preference > Recognition,\nRecognition < 0.5)",
        "Orange\n(Preference < Recognition,\nRecognition > 0.5)",
        "Red\n(Preference < Recognition,\nRecognition < 0.5)",
        "Zero\nDifference",
    ]
    counts = [green_count, blue_count, orange_count, red_count, zero_count]
    sig_counts = [green_sig, blue_sig, orange_sig, red_sig, 0]
    colors = [GREEN, BLUE, ORANGE, RED, GRAY]

    fig, ax = plt.subplots(figsize=(12, 8))
    x_pos = np.arange(len(categories))
    sig_heights = [min(sig, count) for sig, count in zip(sig_counts, counts)]
    non_sig_counts = [max(0, count - sig) for count, sig in zip(counts, sig_counts)]

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

    ax.bar(
        x_pos,
        sig_heights,
        bottom=0,
        color=darker_colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.9,
        label="Significant (p < 0.05)",
    )
    ax.bar(
        x_pos,
        non_sig_counts,
        bottom=sig_heights,
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        alpha=0.7,
        label="Non-significant",
    )

    total = sum(counts)
    for i, (count, sig_count) in enumerate(zip(counts, sig_counts)):
        if count > 0:
            percentage = (count / total * 100) if total > 0 else 0
            label = f"{count}\n({percentage:.1f}%)"
            if sig_count > 0:
                label += f"\n{sig_count} sig"
            ax.text(
                i,
                count,
                label,
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
            )

    max_count = max(counts) if counts else 0
    y_padding = max(max_count * 0.1, 5)
    ax.set_ylim(0, max_count + y_padding)

    ax.set_xlabel("Category", fontsize=12, fontweight="bold")
    ax.set_ylabel(
        "Number of (Evaluator, Treatment) Pairs", fontsize=12, fontweight="bold"
    )
    ax.set_title(
        "Figure 1: Color Categorization Statistics\n(Recognition vs. Preference Comparison)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    ax.set_xticks(x_pos)
    ax.set_xticklabels(categories, fontsize=10)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    if any(sig_counts):
        ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated Figure 1 with report titles: {dest}")
    return dest


def regenerate_figure_2_with_report_titles(
    analysis_dir: Path,
    report_dir: Path,
) -> Path | None:
    """Regenerate Figure 2 with report-appropriate titles."""
    print("Regenerating Figure 2 with report titles...")

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_preference_agreement import (
        add_provider_boundaries,
    )

    # Load agreement matrix
    agreement_path = analysis_dir / "agreement_matrix.csv"
    if not agreement_path.exists():
        print("⚠ Agreement matrix not found, using existing figure")
        source = analysis_dir / "agreement_heatmap.png"
        dest = report_dir / "figure_2_agreement_heatmap.png"
        return copy_figure(source, dest, "Figure 2")

    agreement_matrix = pd.read_csv(agreement_path, index_col=0)

    # Recreate heatmap with report title
    dest = report_dir / "figure_2_agreement_heatmap.png"
    fig, ax = plt.subplots(figsize=(14, 12))

    mask = pd.DataFrame(
        False, index=agreement_matrix.index, columns=agreement_matrix.columns
    )
    for model in agreement_matrix.index:
        if model in agreement_matrix.columns:
            mask.loc[model, model] = True
        for col in agreement_matrix.columns:
            if pd.isna(agreement_matrix.loc[model, col]):
                mask.loc[model, col] = True

    sns.heatmap(
        agreement_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Agreement Score"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    for i, model in enumerate(agreement_matrix.index):
        for j, col in enumerate(agreement_matrix.columns):
            if mask.loc[model, col]:
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

    add_provider_boundaries(ax, agreement_matrix)

    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 2: Preference Agreement Heatmap (Pref-Q Task)\n"
        "Agreement Score = 1 - |P(row prefers self) - P(col prefers row)|",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.20)

    fig.text(
        0.5,
        0.01,
        "High (green): Models agree on quality | "
        "Low (red): Models disagree on quality",
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

    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated Figure 2 with report titles: {dest}")
    return dest


def regenerate_figure_3_with_report_titles(
    analysis_dir: Path,
    report_dir: Path,
) -> Path | None:
    """Regenerate Figure 3 with report-appropriate titles."""
    print("Regenerating Figure 3 with report titles...")

    sys.path.insert(0, str(Path(__file__).parent))
    from analyze_preference_agreement import (
        get_model_family_colors,
    )

    # Load agreement matrix and compute row means
    agreement_path = analysis_dir / "agreement_matrix.csv"
    if not agreement_path.exists():
        print("⚠ Agreement matrix not found, using existing figure")
        source = analysis_dir / "evaluator_agreement_performance.png"
        dest = report_dir / "figure_3_evaluator_agreement_performance.png"
        return copy_figure(source, dest, "Figure 3")

    agreement_matrix = pd.read_csv(agreement_path, index_col=0)
    row_means = agreement_matrix.mean(axis=1, skipna=True).sort_values(ascending=False)

    # Compute significance (one-sample t-test against 0)
    significance_pvalues = {}
    for model in row_means.index:
        row_vals = []
        for col in agreement_matrix.columns:
            if model != col:
                val = agreement_matrix.loc[model, col]
                if pd.notna(val):
                    row_vals.append(val)
        if len(row_vals) > 1:
            t_stat, p_val = stats.ttest_1samp(row_vals, 0.0)
            significance_pvalues[model] = p_val
        else:
            significance_pvalues[model] = None

    significance_series = pd.Series(significance_pvalues).reindex(row_means.index)

    # Regenerate with report title
    dest = report_dir / "figure_3_evaluator_agreement_performance.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    model_list = list(row_means.index)
    colors = get_model_family_colors(model_list)

    _bars = ax.barh(range(len(row_means)), row_means.values, color=colors)
    ax.set_yticks(range(len(row_means)))
    ax.set_yticklabels(row_means.index)
    ax.invert_yaxis()

    min_val = row_means.min()
    max_val = row_means.max()
    val_range = max_val - min_val
    padding = max(val_range * 0.15, 0.05)
    x_min = max(0, min_val - padding)
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Mean Agreement Score", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 3: Evaluator Agreement Performance\n"
        "(Mean Agreement Score When Model Acts as Evaluator)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    for i, (model, val) in enumerate(row_means.items()):
        sig_marker = ""
        if model in significance_series.index and pd.notna(significance_series[model]):
            p_val = significance_series[model]
            if p_val < 0.001:
                sig_marker = "***"
            elif p_val < 0.01:
                sig_marker = "**"
            elif p_val < 0.05:
                sig_marker = "*"
        ax.text(
            val,
            i,
            f" {val:.3f}{sig_marker}",
            va="center",
            ha="left",
            fontsize=9,
            fontweight="bold" if sig_marker else "normal",
        )

    if any(pd.notna(p) and p < 0.05 for p in significance_series.values):
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

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated Figure 3 with report titles: {dest}")
    return dest


def regenerate_figure_4_with_report_titles(
    comparison_dir: Path,
    report_dir: Path,
) -> Path | None:
    """Regenerate Figure 4 with report-appropriate titles."""
    print("Regenerating Figure 4 with report titles...")

    sys.path.insert(0, str(Path(__file__).parent))
    from compare_experiments import (
        get_model_family_colors,
    )

    # Load difference data
    diff_path = comparison_dir / "accuracy_difference.csv"
    if not diff_path.exists():
        print("⚠ Difference matrix not found, using existing figure")
        source = comparison_dir / "evaluator_difference_performance.png"
        dest = report_dir / "figure_4_priming_impact.png"
        return copy_figure(source, dest, "Figure 4")

    diff = pd.read_csv(diff_path, index_col=0)

    # Compute evaluator means and significance
    evaluator_diffs = {}
    evaluator_significance = {}

    for evaluator in diff.index:
        row_vals = []
        for treatment in diff.columns:
            if evaluator != treatment:
                val = diff.loc[evaluator, treatment]
                if pd.notna(val):
                    row_vals.append(val)
        if row_vals:
            mean_diff = np.mean(row_vals)
            evaluator_diffs[evaluator] = mean_diff
            if len(row_vals) > 1:
                t_stat, p_val = stats.ttest_1samp(row_vals, 0.0)
                evaluator_significance[evaluator] = p_val
            else:
                evaluator_significance[evaluator] = None

    evaluator_series = pd.Series(evaluator_diffs).sort_values(ascending=False)
    significance_series = pd.Series(evaluator_significance).reindex(
        evaluator_series.index
    )

    # Regenerate with report title
    dest = report_dir / "figure_4_priming_impact.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    model_list = list(evaluator_series.index)
    colors = get_model_family_colors(model_list)

    _bars = ax.barh(range(len(evaluator_series)), evaluator_series.values, color=colors)
    ax.set_yticks(range(len(evaluator_series)))
    ax.set_yticklabels(evaluator_series.index)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    min_val = evaluator_series.min()
    max_val = evaluator_series.max()
    val_range = max_val - min_val
    padding = max(val_range * 0.15, 0.05)
    x_min = min_val - padding
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel("Mean Difference (Unprimed - Primed)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 4: Task Priming Impact (Unprimed - Primed)\n"
        "Negative values: Priming improved performance | "
        "Positive values: Priming reduced performance",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    for i, (model, val) in enumerate(evaluator_series.items()):
        sig_marker = ""
        if model in significance_series.index and pd.notna(significance_series[model]):
            p_val = significance_series[model]
            if p_val < 0.001:
                sig_marker = "***"
            elif p_val < 0.01:
                sig_marker = "**"
            elif p_val < 0.05:
                sig_marker = "*"
        x_pos = val if val >= 0 else val
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos,
            i,
            f" {val:+.3f}{sig_marker}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold" if sig_marker else "normal",
        )

    if any(pd.notna(p) and p < 0.05 for p in significance_series.values):
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

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated Figure 4 with report titles: {dest}")
    return dest


def regenerate_figure_5_with_report_titles(
    comparison_dir: Path,
    report_dir: Path,
) -> Path | None:
    """Regenerate Figure 5 with report-appropriate titles."""
    print("Regenerating Figure 5 with report titles...")

    sys.path.insert(0, str(Path(__file__).parent))
    from compare_experiments import (
        get_model_family_colors,
    )

    # Load difference data
    diff_path = comparison_dir / "accuracy_difference.csv"
    if not diff_path.exists():
        print("⚠ Difference matrix not found, using existing figure")
        source = comparison_dir / "evaluator_difference_performance.png"
        dest = report_dir / "figure_5_semantic_variance.png"
        return copy_figure(source, dest, "Figure 5")

    diff = pd.read_csv(diff_path, index_col=0)

    # Compute evaluator means and significance
    evaluator_diffs = {}
    evaluator_significance = {}

    for evaluator in diff.index:
        row_vals = []
        for treatment in diff.columns:
            if evaluator != treatment:
                val = diff.loc[evaluator, treatment]
                if pd.notna(val):
                    row_vals.append(val)
        if row_vals:
            mean_diff = np.mean(row_vals)
            evaluator_diffs[evaluator] = mean_diff
            if len(row_vals) > 1:
                t_stat, p_val = stats.ttest_1samp(row_vals, 0.0)
                evaluator_significance[evaluator] = p_val
            else:
                evaluator_significance[evaluator] = None

    evaluator_series = pd.Series(evaluator_diffs).sort_values(ascending=False)
    significance_series = pd.Series(evaluator_significance).reindex(
        evaluator_series.index
    )

    # Regenerate with report title
    dest = report_dir / "figure_5_semantic_variance.png"

    fig, ax = plt.subplots(figsize=(12, 8))
    model_list = list(evaluator_series.index)
    colors = get_model_family_colors(model_list)

    _bars = ax.barh(range(len(evaluator_series)), evaluator_series.values, color=colors)
    ax.set_yticks(range(len(evaluator_series)))
    ax.set_yticklabels(evaluator_series.index)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

    min_val = evaluator_series.min()
    max_val = evaluator_series.max()
    val_range = max_val - min_val
    padding = max(val_range * 0.15, 0.05)
    x_min = min_val - padding
    x_max = max_val + padding
    ax.set_xlim(x_min, x_max)

    ax.set_xlabel(
        "Mean Difference (Alignment - Summary)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Figure 5: Semantic Variance Impact (Alignment - Summary)\n"
        "Positive values: Higher accuracy on alignment tasks | "
        "Negative values: Higher accuracy on summary tasks",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    for i, (model, val) in enumerate(evaluator_series.items()):
        sig_marker = ""
        if model in significance_series.index and pd.notna(significance_series[model]):
            p_val = significance_series[model]
            if p_val < 0.001:
                sig_marker = "***"
            elif p_val < 0.01:
                sig_marker = "**"
            elif p_val < 0.05:
                sig_marker = "*"
        x_pos = val if val >= 0 else val
        ha = "left" if val >= 0 else "right"
        ax.text(
            x_pos,
            i,
            f" {val:+.3f}{sig_marker}",
            va="center",
            ha=ha,
            fontsize=9,
            fontweight="bold" if sig_marker else "normal",
        )

    if any(pd.notna(p) and p < 0.05 for p in significance_series.values):
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

    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(dest, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"  ✓ Generated Figure 5 with report titles: {dest}")
    return dest


def find_comparison_dir(rec_path: Path, pref_path: Path) -> Path | None:
    """Find the comparison directory created by compare_recognition_preference.py."""
    # Parse paths to determine comparison directory
    # Expected: data/analysis/{dataset}/{subset}/{exp}/accuracy_pivot.csv
    rec_parts = rec_path.parts
    pref_parts = pref_path.parts

    if "analysis" not in rec_parts or "analysis" not in pref_parts:
        return None

    rec_idx = rec_parts.index("analysis")
    pref_idx = pref_parts.index("analysis")

    if rec_idx + 3 < len(rec_parts) and pref_idx + 3 < len(pref_parts):
        rec_dataset = rec_parts[rec_idx + 1]
        rec_subset = rec_parts[rec_idx + 2]
        rec_exp = rec_parts[rec_idx + 3]  # e.g., "11_UT_PW-Q_Rec_NPr"

        pref_dataset = pref_parts[pref_idx + 1]
        pref_subset = pref_parts[pref_idx + 2]
        pref_exp = pref_parts[pref_idx + 3]  # e.g., "14_UT_PW-Q_Pref-Q_NPr"

        is_cross_dataset = (rec_dataset != pref_dataset) or (rec_subset != pref_subset)

        if is_cross_dataset:
            # Cross-dataset comparison
            # Extract experiment number and code
            rec_num = (
                rec_exp.split("_", 1)[0]
                if "_" in rec_exp and rec_exp.split("_")[0].isdigit()
                else None
            )
            rec_code = rec_exp.split("_", 1)[1] if "_" in rec_exp else rec_exp
            exp_code = rec_code  # Use rec_code as base
            if rec_num:
                exp_code = f"{rec_num}_{exp_code}"
            comparison_name = (
                f"{rec_dataset}_{rec_subset}_vs_{pref_dataset}_{pref_subset}"
            )
            return (
                Path("data/analysis/cross-dataset_comparisons")
                / exp_code
                / comparison_name
            )
        else:
            # Same dataset comparison - use full experiment names with numbers
            comparison_name = f"{rec_exp}_vs_{pref_exp}"
            return (
                Path("data/analysis")
                / rec_dataset
                / rec_subset
                / "comparisons"
                / comparison_name
            )

    return None


def find_analysis_dir(results_dir: Path) -> Path | None:
    """Find the analysis directory corresponding to a results directory."""
    parts = list(results_dir.parts)
    if "results" in parts:
        results_idx = parts.index("results")
        if results_idx + 2 < len(parts) - 1:
            # Replace 'results' with 'analysis' and keep everything including experiment dir
            # e.g., data/results/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr
            # -> data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr
            parts[results_idx] = "analysis"
            return Path(*parts)  # Keep the experiment directory name
    return None


def find_experiment_comparison_dir(exp1_dir: Path, exp2_dir: Path) -> Path | None:
    """Find the comparison directory created by compare_experiments.py."""
    parts1 = list(exp1_dir.parts)
    parts2 = list(exp2_dir.parts)

    if "results" not in parts1 or "results" not in parts2:
        return None

    results_idx1 = parts1.index("results")
    results_idx2 = parts2.index("results")

    if results_idx1 + 2 < len(parts1) - 1 and results_idx2 + 2 < len(parts2) - 1:
        dataset1 = parts1[results_idx1 + 1]
        subset1 = parts1[results_idx1 + 2]
        exp1_name = parts1[results_idx1 + 3]

        dataset2 = parts2[results_idx2 + 1]
        subset2 = parts2[results_idx2 + 2]
        exp2_name = parts2[results_idx2 + 3]

        is_cross_dataset = (dataset1 != dataset2) or (subset1 != subset2)

        exp1_code = (
            exp1_name.split("_", 1)[1]
            if "_" in exp1_name and exp1_name.split("_")[0].isdigit()
            else exp1_name
        )
        exp2_code = (
            exp2_name.split("_", 1)[1]
            if "_" in exp2_name and exp2_name.split("_")[0].isdigit()
            else exp2_name
        )
        exp1_num = (
            exp1_name.split("_")[0]
            if "_" in exp1_name and exp1_name.split("_")[0].isdigit()
            else None
        )
        exp2_num = (
            exp2_name.split("_")[0]
            if "_" in exp2_name and exp2_name.split("_")[0].isdigit()
            else None
        )

        if is_cross_dataset:
            experiment_code = (
                exp1_code if exp1_code == exp2_code else f"{exp1_code}_vs_{exp2_code}"
            )
            if exp1_num:
                experiment_code = f"{exp1_num}_{experiment_code}"
            comparison_name = f"{dataset1}_{subset1}_vs_{dataset2}_{subset2}"
            return (
                Path("data/analysis/cross-dataset_comparisons")
                / experiment_code
                / comparison_name
            )
        else:
            if exp1_num and exp2_num:
                comparison_name = f"{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}"
            else:
                comparison_name = f"{exp1_code}_vs_{exp2_code}"
            return (
                Path("data/analysis")
                / dataset1
                / subset1
                / "comparisons"
                / comparison_name
            )

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate all figures for Performance Report #1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script runs the necessary analysis scripts and copies the generated figures
to a report_1 folder. The original figures remain in their original locations.

Note: For better figure titles and axis labels, you may want to modify the plotting
functions in the individual analysis scripts to accept custom title/label parameters.
        """,
    )

    parser.add_argument(
        "--recognition_pivot",
        type=str,
        required=True,
        help="Path to recognition accuracy pivot CSV (for Figure 1)",
    )
    parser.add_argument(
        "--preference_pivot",
        type=str,
        required=True,
        help="Path to preference accuracy pivot CSV (for Figure 1)",
    )
    parser.add_argument(
        "--preference_results",
        type=str,
        required=True,
        help="Path to preference experiment results directory (for Figures 2 & 3)",
    )
    parser.add_argument(
        "--unprimed_results",
        type=str,
        required=True,
        help="Path to unprimed experiment results directory (for Figure 4)",
    )
    parser.add_argument(
        "--primed_results",
        type=str,
        required=True,
        help="Path to primed experiment results directory (for Figure 4)",
    )
    parser.add_argument(
        "--alignment_results",
        type=str,
        required=True,
        help="Path to alignment dataset experiment results directory (for Figure 5)",
    )
    parser.add_argument(
        "--summary_results",
        type=str,
        required=True,
        help="Path to summary dataset experiment results directory (for Figure 5)",
    )
    parser.add_argument(
        "--report_dir",
        type=str,
        default="report_1",
        help="Directory to save report figures (default: report_1)",
    )

    args = parser.parse_args()

    # Create report directory
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("GENERATING FIGURES FOR PERFORMANCE REPORT #1")
    print(f"{'='*70}")
    print(f"Report directory: {report_dir.absolute()}")
    print(f"{'='*70}\n")

    figures_generated = []

    # Figure 1: Color Categorization Statistics
    print("\n" + "=" * 70)
    print("FIGURE 1: Color Categorization Statistics")
    print("=" * 70)
    success = run_script(
        "compare_recognition_preference.py",
        [
            "--recognition",
            args.recognition_pivot,
            "--preference",
            args.preference_pivot,
        ],
    )
    if success:
        comparison_dir = find_comparison_dir(
            Path(args.recognition_pivot), Path(args.preference_pivot)
        )
        if comparison_dir:
            # Regenerate with report-appropriate titles
            rec_pivot_path = Path(args.recognition_pivot)
            pref_pivot_path = Path(args.preference_pivot)
            fig1 = regenerate_figure_1_with_report_titles(
                comparison_dir, report_dir, rec_pivot_path, pref_pivot_path
            )
            if fig1:
                figures_generated.append(("Figure 1", fig1))

    # Figures 2 & 3: Preference Agreement Analysis
    print("\n" + "=" * 70)
    print("FIGURES 2 & 3: Preference Agreement Analysis")
    print("=" * 70)
    success = run_script(
        "analyze_preference_agreement.py", ["--results_dir", args.preference_results]
    )
    if success:
        analysis_dir = find_analysis_dir(Path(args.preference_results))
        if analysis_dir:
            # Figure 2: Agreement Heatmap - regenerate with report titles
            fig2 = regenerate_figure_2_with_report_titles(analysis_dir, report_dir)
            if fig2:
                figures_generated.append(("Figure 2", fig2))

            # Figure 3: Evaluator Agreement Performance - regenerate with report titles
            fig3 = regenerate_figure_3_with_report_titles(analysis_dir, report_dir)
            if fig3:
                figures_generated.append(("Figure 3", fig3))

    # Figure 4: Task Priming Impact
    print("\n" + "=" * 70)
    print("FIGURE 4: Task Priming Impact")
    print("=" * 70)
    success = run_script(
        "compare_experiments.py",
        [
            "--experiment1",
            args.unprimed_results,
            "--experiment2",
            args.primed_results,
        ],
    )
    if success:
        comparison_dir = find_experiment_comparison_dir(
            Path(args.unprimed_results), Path(args.primed_results)
        )
        if comparison_dir:
            # Regenerate with report-appropriate titles
            fig4 = regenerate_figure_4_with_report_titles(comparison_dir, report_dir)
            if fig4:
                figures_generated.append(("Figure 4", fig4))

    # Figure 5: Semantic Variance Impact
    print("\n" + "=" * 70)
    print("FIGURE 5: Semantic Variance Impact")
    print("=" * 70)
    success = run_script(
        "compare_experiments.py",
        [
            "--experiment1",
            args.alignment_results,
            "--experiment2",
            args.summary_results,
        ],
    )
    if success:
        comparison_dir = find_experiment_comparison_dir(
            Path(args.alignment_results), Path(args.summary_results)
        )
        if comparison_dir:
            # Regenerate with report-appropriate titles
            fig5 = regenerate_figure_5_with_report_titles(comparison_dir, report_dir)
            if fig5:
                figures_generated.append(("Figure 5", fig5))

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Figures generated: {len(figures_generated)}/5")
    for name, path in figures_generated:
        print(f"  ✓ {name}: {path.name}")
    print(f"\nAll figures saved to: {report_dir.absolute()}")
    print("\nNote: Original figures remain in their analysis directories.")
    print("      To improve figure titles/labels, modify the plotting functions")
    print("      in the individual analysis scripts.")
    print(f"{'='*70}\n")

    if len(figures_generated) < 5:
        print(
            "⚠ Warning: Some figures were not generated. Check the output above for errors."
        )
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
