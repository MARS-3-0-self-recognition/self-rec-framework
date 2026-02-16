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
from collections import Counter
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from matplotlib.legend_handler import HandlerTuple
from scipy import stats
from utils import (
    expand_model_names,
    get_model_provider,
    provider_to_model_name,
    calculate_binomial_ci,
    weighted_regression_with_ci,
    weighted_correlation,
    apply_fdr_correction,
    get_significance_marker,
    save_figure_minimal_version,
    save_figure_no_r_version,
    format_evaluator_model_display_name,
)


# Suffix for reasoning/thinking models; instruct name = thinking_name.removesuffix(THINKING_SUFFIX)
THINKING_SUFFIX = "-thinking"


def extract_dataset_name(full_path: str) -> str:
    """
    Extract short dataset name from full path.

    Examples:
        "wikisum/training_set_1-20+test_set_1-30" -> "wikisum"
        "sharegpt/english_26+english2_74" -> "sharegpt"
        "bigcodebench/instruct_1-50" -> "bigcodebench"
    """
    return full_path.split("/")[0]

def format_dataset_display_name(dataset_name: str) -> str:
    """Format dataset name for display in legends (neater capitalization)."""
    mapping = {
        "wikisum": "WikiSum",
        "pku_saferlhf": "PKU-SafeRLHF",
        "bigcodebench": "BigCodeBench",
        "sharegpt": "ShareGPT",
    }
    return mapping.get(dataset_name.lower(), dataset_name)


def _get_thinking_instruct_pairs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
) -> tuple[list[str], list[str]]:
    """
    Find thinking (exp1) <-> instruct (exp2) pairs. Exp1 model names must end with THINKING_SUFFIX;
    instruct name is the same name with that suffix removed. Errors if two thinking names
    map to the same instruct name.

    Returns:
        (thinking_names, instruct_names) parallel lists, both ordered; instruct_names unique.
    """
    thinking_models = [m for m in df1.index if str(m).endswith(THINKING_SUFFIX)]
    if not thinking_models:
        raise ValueError(
            f"No models in exp1 end with '{THINKING_SUFFIX}'. "
            "Cannot run reasoning-vs-instruct comparison."
        )
    instruct_names = []
    for m in thinking_models:
        base = str(m).removesuffix(THINKING_SUFFIX)
        if base not in df2.index:
            continue
        instruct_names.append(base)
    if not instruct_names:
        raise ValueError(
            f"No exp1 thinking-models had a matching instruct model in exp2. "
            f"Exp1 thinking models: {thinking_models}; exp2 index: {list(df2.index)}."
        )
    # Check one-to-one: no duplicate instruct name
    if len(instruct_names) != len(set(instruct_names)):
        counts = Counter(instruct_names)
        dups = [k for k, v in counts.items() if v > 1]
        raise ValueError(
            f"Duplicate instruct name(s) from thinking models: {dups}. "
            "Each instruct name must map from exactly one thinking model."
        )
    # Rebuild thinking list in same order (only those with a pair)
    thinking_names = [m for m in thinking_models if str(m).removesuffix(THINKING_SUFFIX) in df2.index]
    return thinking_names, instruct_names


def load_and_compare(
    exp1_file: Path,
    exp2_file: Path,
    model_order: list[str] | None = None,
    reasoning_vs_instruct: bool = False,
) -> pd.DataFrame:
    """
    Load both performance files and compute difference (exp1 - exp2).

    Args:
        exp1_file: Path to first experiment's aggregated_performance.csv
        exp2_file: Path to second experiment's aggregated_performance.csv
        model_order: Optional list of models to filter/order (by instruct name in reasoning_vs_instruct mode)
        reasoning_vs_instruct: If True, pair exp1 models ending with -thinking to exp2 instruct names.

    Returns:
        DataFrame with models as index, datasets as columns, differences as values
    """
    df1 = pd.read_csv(exp1_file, index_col=0)
    df2 = pd.read_csv(exp2_file, index_col=0)

    common_datasets = df1.columns.intersection(df2.columns)
    if len(common_datasets) == 0:
        raise ValueError("No common datasets found between the two experiments!")
    df1 = df1[common_datasets]
    df2 = df2[common_datasets]

    if reasoning_vs_instruct:
        thinking_names, instruct_names = _get_thinking_instruct_pairs(df1, df2)
        df1_paired = df1.loc[thinking_names].copy()
        df1_paired.index = instruct_names
        df2_paired = df2.loc[instruct_names]
        df_diff = df1_paired - df2_paired
    else:
        common_models = df1.index.intersection(df2.index)
        if len(common_models) == 0:
            raise ValueError("No common models found between the two experiments!")
        df1 = df1.loc[common_models]
        df2 = df2.loc[common_models]
        df_diff = df1 - df2

    if model_order:
        available_models = [m for m in model_order if m in df_diff.index]
        if available_models:
            df_diff = df_diff.reindex(available_models)

    return df_diff


def get_family_base_color(model_name: str) -> str:
    """
    Get the base color for a model's family (single color, no gradations).
    
    Args:
        model_name: Short model name (e.g., "gpt-4o", "haiku-3.5")
        
    Returns:
        Hex color code for the model's family
    """
    # Define model families and their base colors (matching logo colors)
    family_colors = {
        "openai": "#10a37f",  # OpenAI green
        "openai-oss": "#10a37f",  # OpenAI green
        "anthropic": "#ea580c",  # Claude red-orange
        "google": "#fbbf24",  # Google yellow
        "llama": "#3b82f6",  # Blue
        "qwen": "#7c3aed",  # Purple
        "deepseek": "#dc2626",  # Red
        "xai": "#1d4ed8",  # XAI blue
        "moonshot": "#0891b2",  # Cyan
    }
    
    model_lower = model_name.lower()
    
    if model_lower.startswith("gpt-"):
        return family_colors["openai"]
    elif (
        model_lower.startswith("haiku-")
        or model_lower.startswith("sonnet-")
        or model_lower.startswith("opus-")
    ):
        return family_colors["anthropic"]
    elif model_lower.startswith("gemini-"):
        return family_colors["google"]
    elif model_lower.startswith("ll-"):
        return family_colors["llama"]
    elif model_lower.startswith("qwen-"):
        return family_colors["qwen"]
    elif model_lower.startswith("deepseek-"):
        return family_colors["deepseek"]
    elif model_lower.startswith("grok-"):
        return family_colors["xai"]
    elif model_lower.startswith("kimi-"):
        return family_colors["moonshot"]
    else:
        return "#9ca3af"  # Default gray


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
    fig, ax = plt.subplots(figsize=(20, max(8, len(df) * 0.4)))

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
    
    # TODO: Add significance markers if counts data available
    # Test if differences are significantly different from 0
    # This would require loading counts from both experiments

    # Add reference line at 0
    ax.axvline(
        x=0,
        color="#333333",
        linestyle="--",
        linewidth=1.5,
        alpha=1.0,
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


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    exp1_name: str,
    exp2_name: str,
    df_counts_exp1: pd.DataFrame | None = None,
    df_counts_exp2: pd.DataFrame | None = None,
    df_perf_exp1: pd.DataFrame | None = None,
    df_perf_exp2: pd.DataFrame | None = None,
):
    """
    Create a grouped bar chart with models on x-axis and side-by-side dataset bars.
    Shows exp1 - exp2 differences with error bars and significance markers.

    Args:
        df: DataFrame with models as index, datasets as columns, differences as values
        output_path: Path to save the plot
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment
        df_counts_exp1: Optional counts DataFrame for exp1 (for error bars)
        df_counts_exp2: Optional counts DataFrame for exp2 (for error bars)
    """
    print("Generating grouped bar chart...")

    # Remove models where all values are 0
    df = df.loc[(df != 0).any(axis=1)]

    if df.empty:
        print("  ⚠ No data to plot (all differences are zero)")
        return

    # Shorten dataset names for legend
    short_names = [extract_dataset_name(name) for name in df.columns]
    df.columns = short_names

    # Sort models by total absolute difference
    df["_total_abs"] = df.abs().sum(axis=1)
    df = df.sort_values("_total_abs", ascending=False)
    df = df.drop(columns=["_total_abs"])

    # Process counts DataFrames: shorten column names to match df.columns
    if df_counts_exp1 is not None:
        counts_short_names_exp1 = [extract_dataset_name(name) for name in df_counts_exp1.columns]
        df_counts_exp1.columns = counts_short_names_exp1
        # Reindex to match df (models and columns)
        df_counts_exp1 = df_counts_exp1.reindex(index=df.index, columns=df.columns)
        print(f"  ✓ Processed counts for {exp1_name}: {df_counts_exp1.shape[0]} models × {df_counts_exp1.shape[1]} datasets")
    
    if df_counts_exp2 is not None:
        counts_short_names_exp2 = [extract_dataset_name(name) for name in df_counts_exp2.columns]
        df_counts_exp2.columns = counts_short_names_exp2
        # Reindex to match df (models and columns)
        df_counts_exp2 = df_counts_exp2.reindex(index=df.index, columns=df.columns)
        print(f"  ✓ Processed counts for {exp2_name}: {df_counts_exp2.shape[0]} models × {df_counts_exp2.shape[1]} datasets")

    # Create figure (wider for many bars)
    n_models = len(df)
    n_datasets = len(df.columns)
    fig_width = max(20, n_models * n_datasets * 0.22)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Define colors
    if n_datasets <= 4:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
    elif n_datasets <= 8:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

    # Calculate error bars if counts available
    error_bars = None
    error_bar_count = 0
    if df_counts_exp1 is not None and df_counts_exp2 is not None:
        # Calculate error propagation: SE_diff = sqrt(SE_exp1² + SE_exp2²)
        errors = []
        for model in df.index:
            model_errors = []
            for dataset in df.columns:
                try:
                    n1 = df_counts_exp1.loc[model, dataset] if model in df_counts_exp1.index and dataset in df_counts_exp1.columns else None
                    n2 = df_counts_exp2.loc[model, dataset] if model in df_counts_exp2.index and dataset in df_counts_exp2.columns else None
                    
                    if pd.notna(n1) and pd.notna(n2) and n1 > 0 and n2 > 0:
                        error_bar_count += 1
                        # Calculate SE for difference: SE_diff = sqrt(SE1² + SE2²)
                        # Get actual proportions if available
                        # Note: df_perf_exp1/exp2 have full column names, need to match
                        p1 = None
                        p2 = None
                        if df_perf_exp1 is not None and df_perf_exp2 is not None:
                            try:
                                # Find matching column in original data (full path)
                                matching_col1 = None
                                matching_col2 = None
                                for col in df_perf_exp1.columns:
                                    if extract_dataset_name(col) == dataset:
                                        matching_col1 = col
                                        break
                                for col in df_perf_exp2.columns:
                                    if extract_dataset_name(col) == dataset:
                                        matching_col2 = col
                                        break
                                
                                if matching_col1 and model in df_perf_exp1.index:
                                    p1 = df_perf_exp1.loc[model, matching_col1]
                                if matching_col2 and model in df_perf_exp2.index:
                                    p2 = df_perf_exp2.loc[model, matching_col2]
                                if pd.notna(p1) and pd.notna(p2):
                                    # Use actual proportions
                                    _, _, se1 = calculate_binomial_ci(p1, n1)
                                    _, _, se2 = calculate_binomial_ci(p2, n2)
                                    se_diff = np.sqrt(se1**2 + se2**2)
                                else:
                                    # Fallback: use conservative estimate
                                    p_approx = 0.5
                                    se1 = np.sqrt(p_approx * (1 - p_approx) / n1)
                                    se2 = np.sqrt(p_approx * (1 - p_approx) / n2)
                                    se_diff = np.sqrt(se1**2 + se2**2)
                            except (KeyError, IndexError, AttributeError):
                                # Fallback: use conservative estimate
                                p_approx = 0.5
                                se1 = np.sqrt(p_approx * (1 - p_approx) / n1)
                                se2 = np.sqrt(p_approx * (1 - p_approx) / n2)
                                se_diff = np.sqrt(se1**2 + se2**2)
                        else:
                            # Conservative estimate: assume both proportions are 0.5
                            p_approx = 0.5
                            se1 = np.sqrt(p_approx * (1 - p_approx) / n1)
                            se2 = np.sqrt(p_approx * (1 - p_approx) / n2)
                            se_diff = np.sqrt(se1**2 + se2**2)
                        
                        model_errors.append(1.96 * se_diff)  # 95% CI half-width
                    else:
                        model_errors.append(0)
                except (KeyError, IndexError):
                    model_errors.append(0)
            errors.append(model_errors)
        error_bars = np.array(errors).T  # Transpose to match pandas plot format
        print(f"  ✓ Calculated error bars for {error_bar_count} model-dataset combinations")

    # Calculate x positions for bars
    x = np.arange(len(df.index))
    width = 0.8 / n_datasets

    # Plot grouped bars with error bars if available
    if error_bars is not None and error_bars.size > 0:
        # Use matplotlib bar plot to add error bars
        for i, (dataset, color) in enumerate(zip(df.columns, colors)):
            offset = (i - n_datasets/2 + 0.5) * width
            values = df[dataset].values
            if error_bars.shape[0] > i:
                yerr = error_bars[i]
                # Check if we have any non-zero error bars
                if np.any(yerr > 0):
                    ax.bar(
                        x + offset,
                        values,
                        width=width * 0.9,
                        color=color,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        yerr=yerr,
                        error_kw={'capsize': 3, 'capthick': 1.0, 'elinewidth': 1.5, 'alpha': 0.7},
                        label=dataset
                    )
                else:
                    # No error bars, plot without them
                    ax.bar(
                        x + offset,
                        values,
                        width=width * 0.9,
                        color=color,
                        alpha=0.8,
                        edgecolor="black",
                        linewidth=0.5,
                        label=dataset
                    )
            else:
                # No error bars for this dataset
                ax.bar(
                    x + offset,
                    values,
                    width=width * 0.9,
                    color=color,
                    alpha=0.8,
                    edgecolor="black",
                    linewidth=0.5,
                    label=dataset
                )
        ax.set_xticks(x)
        ax.set_xticklabels(
            [format_evaluator_model_display_name(m) for m in df.index],
            rotation=45,
            ha="right",
            fontsize=18,
        )
    else:
        # Fallback to pandas plot (no error bars)
        df.plot(
            kind="bar",
            ax=ax,
            color=colors,
            width=0.8,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5
        )
        # Get x positions from pandas plot
        x = np.arange(len(df.index))
        ax.set_xticklabels(
            [format_evaluator_model_display_name(m) for m in df.index],
            rotation=45,
            ha="right",
            fontsize=18,
        )

    # Add significance markers if counts available
    significance_count = 0
    if df_counts_exp1 is not None and df_counts_exp2 is not None:
        # Test if differences are significantly different from 0
        from scipy.stats import binomtest
        p_values = []
        for model in df.index:
            for dataset in df.columns:
                try:
                    n1 = df_counts_exp1.loc[model, dataset] if model in df_counts_exp1.index and dataset in df_counts_exp1.columns else None
                    n2 = df_counts_exp2.loc[model, dataset] if model in df_counts_exp2.index and dataset in df_counts_exp2.columns else None
                    diff = df.loc[model, dataset]
                    
                    if pd.notna(n1) and pd.notna(n2) and n1 > 0 and n2 > 0 and pd.notna(diff):
                        # Two-sample proportion test: H0: p1 = p2
                        # Get actual proportions if available, otherwise use conservative estimate
                        # Note: df_perf_exp1/exp2 have full column names, need to match
                        p1 = None
                        p2 = None
                        if df_perf_exp1 is not None and df_perf_exp2 is not None:
                            try:
                                # Find matching column in original data (full path)
                                matching_col1 = None
                                matching_col2 = None
                                for col in df_perf_exp1.columns:
                                    if extract_dataset_name(col) == dataset:
                                        matching_col1 = col
                                        break
                                for col in df_perf_exp2.columns:
                                    if extract_dataset_name(col) == dataset:
                                        matching_col2 = col
                                        break
                                
                                if matching_col1 and model in df_perf_exp1.index:
                                    p1 = df_perf_exp1.loc[model, matching_col1]
                                if matching_col2 and model in df_perf_exp2.index:
                                    p2 = df_perf_exp2.loc[model, matching_col2]
                                if pd.notna(p1) and pd.notna(p2):
                                    # Use actual proportions
                                    se1 = np.sqrt(p1 * (1 - p1) / n1)
                                    se2 = np.sqrt(p2 * (1 - p2) / n2)
                                    se_diff = np.sqrt(se1**2 + se2**2)
                                else:
                                    # Fallback to pooled proportion
                                    p_pooled = 0.5  # Conservative estimate
                                    se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                            except (KeyError, IndexError):
                                # Fallback to pooled proportion
                                p_pooled = 0.5
                                se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                        else:
                            # Conservative estimate: assume both proportions are 0.5
                            p_pooled = 0.5
                            se_diff = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                        
                        if se_diff > 0:
                            z_score = abs(diff) / se_diff
                            # Two-tailed p-value from normal distribution
                            from scipy.stats import norm
                            p_val = 2 * (1 - norm.cdf(z_score))
                            p_values.append(p_val)
                        else:
                            p_values.append(np.nan)
                    else:
                        p_values.append(np.nan)
                except (KeyError, IndexError):
                    p_values.append(np.nan)
        
        # Apply FDR correction
        if len(p_values) > 0 and not np.all(np.isnan(p_values)):
            valid_p_values = [p for p in p_values if pd.notna(p)]
            if len(valid_p_values) > 0:
                rejected, p_corrected = apply_fdr_correction(p_values)
                # Add markers to bars
                p_idx = 0
                for model_idx, model in enumerate(df.index):
                    for dataset_idx, dataset in enumerate(df.columns):
                        if p_idx < len(rejected) and rejected[p_idx] and pd.notna(p_values[p_idx]):
                            p_val = p_corrected[p_idx] if p_idx < len(p_corrected) else p_values[p_idx]
                            marker = get_significance_marker(p_values[p_idx], p_val)
                            if marker:
                                significance_count += 1
                                offset = (dataset_idx - n_datasets/2 + 0.5) * width
                                bar_height = df.loc[model, dataset]
                                yerr_val = error_bars[dataset_idx, model_idx] if error_bars is not None and error_bars.shape[0] > dataset_idx and error_bars.shape[1] > model_idx else 0.02
                                # Place marker above bar if positive, below if negative
                                if bar_height < 0:
                                    marker_y = bar_height - yerr_val - 0.01
                                    va = 'top'
                                else:
                                    marker_y = bar_height + yerr_val + 0.01
                                    va = 'bottom'
                                ax.text(
                                    x[model_idx] + offset,
                                    marker_y,
                                    marker,
                                    ha='center',
                                    va=va,
                                    fontsize=10,
                                    fontweight='bold'
                                )
                        p_idx += 1
                print(f"  ✓ Added {significance_count} significance markers")
            else:
                print(f"  ⚠ No valid p-values for significance testing")

    # Add reference line at 0
    ax.axhline(y=0, color="#333333", linestyle="--", linewidth=1.5, alpha=1.0, label="No difference")

    # Labels and title
    ax.set_xlabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"Performance Difference ({exp1_name} - {exp2_name})", fontsize=12, fontweight="bold")
    
    title = f"Performance Contrast: {exp1_name} vs {exp2_name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    # Legend - add significance marker if any; No difference 2nd to last, * p last
    handles, labels = ax.get_legend_handles_labels()
    # Check if any significance markers were added
    has_significance = any('*' in str(text.get_text()) for text in ax.texts)
    if has_significance:
        # Add significance marker to legend
        from matplotlib.patches import Rectangle
        sig_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        handles.append(sig_handle)
        labels.append("* p < 0.05")
    # Reorder so "No difference" is 2nd to last, * p < 0.05 last
    others_h = [h for h, l in zip(handles, labels) if l not in ("No difference", "* p < 0.05")]
    others_l = [l for h, l in zip(handles, labels) if l not in ("No difference", "* p < 0.05")]
    ref_pair = next(((h, l) for h, l in zip(handles, labels) if l == "No difference"), (None, None))
    sig_pair = next(((h, l) for h, l in zip(handles, labels) if l == "* p < 0.05"), (None, None))
    handles = others_h + ([ref_pair[0]] if ref_pair[0] is not None else []) + ([sig_pair[0]] if sig_pair[0] is not None else [])
    labels = others_l + ([ref_pair[1]] if ref_pair[1] is not None else []) + ([sig_pair[1]] if sig_pair[1] is not None else [])
    ax.legend(handles=handles, labels=labels, title="Dataset", loc="center left", bbox_to_anchor=(1.02, 0.5))
    
    # Rotate x labels if many models
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved grouped bar chart to: {output_path}")
    save_figure_minimal_version(ax, output_path)
    plt.close()


def load_original_data(
    exp1_file: Path,
    exp2_file: Path,
    model_order: list[str] | None = None,
    reasoning_vs_instruct: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load both performance files and return original data (not differences).

    Args:
        exp1_file: Path to first experiment's aggregated_performance.csv
        exp2_file: Path to second experiment's aggregated_performance.csv
        model_order: Optional list of models to filter/order (by instruct name in reasoning_vs_instruct mode)
        reasoning_vs_instruct: If True, pair exp1 -thinking models to exp2 instruct; both returned with instruct index.

    Returns:
        Tuple of (df1, df2) - both DataFrames with same index (instruct names in reasoning mode), datasets as columns
    """
    df1 = pd.read_csv(exp1_file, index_col=0)
    df2 = pd.read_csv(exp2_file, index_col=0)

    common_datasets = df1.columns.intersection(df2.columns)
    if len(common_datasets) == 0:
        raise ValueError("No common datasets found between the two experiments!")
    df1 = df1[common_datasets]
    df2 = df2[common_datasets]

    if reasoning_vs_instruct:
        thinking_names, instruct_names = _get_thinking_instruct_pairs(df1, df2)
        df1_paired = df1.loc[thinking_names].copy()
        df1_paired.index = instruct_names
        df2_paired = df2.loc[instruct_names]
        df1, df2 = df1_paired, df2_paired
    else:
        common_models = df1.index.intersection(df2.index)
        if len(common_models) == 0:
            raise ValueError("No common models found between the two experiments!")
        df1 = df1.loc[common_models]
        df2 = df2.loc[common_models]

    if model_order:
        available_models = [m for m in model_order if m in df1.index]
        if available_models:
            df1 = df1.reindex(available_models)
            df2 = df2.reindex(available_models)

    return df1, df2


def plot_performance_scatter(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    output_path: Path,
    exp1_name: str,
    exp2_name: str,
    df_counts_exp1: pd.DataFrame | None = None,
    df_counts_exp2: pd.DataFrame | None = None,
):
    """
    Create a scatter plot with exp1 performance on y-axis and exp2 performance on x-axis.
    Colors indicate model families, shapes indicate datasets.
    Linear regression fitted to each dataset group.
    
    Args:
        df1: DataFrame with exp1 performance (models as index, datasets as columns)
        df2: DataFrame with exp2 performance (models as index, datasets as columns)
        output_path: Path to save the plot
        exp1_name: Name of first experiment
        exp2_name: Name of second experiment
    """
    print("Generating performance scatter plot...")

    # Process counts DataFrames: match columns by short name
    if df_counts_exp1 is not None:
        # Create mapping from counts columns to df1 columns (by short name)
        counts_col_map_exp1 = {}
        for df1_col in df1.columns:
            df1_short = extract_dataset_name(df1_col)
            # Try to find matching counts column
            for counts_col in df_counts_exp1.columns:
                counts_short = extract_dataset_name(counts_col)
                if counts_short == df1_short:
                    counts_col_map_exp1[df1_col] = counts_col
                    break
        # Rename counts columns to match df1
        if counts_col_map_exp1:
            df_counts_exp1 = df_counts_exp1.rename(columns={v: k for k, v in counts_col_map_exp1.items()})
        df_counts_exp1 = df_counts_exp1.reindex(index=df1.index, columns=df1.columns)
    
    if df_counts_exp2 is not None:
        # Create mapping from counts columns to df2 columns (by short name)
        counts_col_map_exp2 = {}
        for df2_col in df2.columns:
            df2_short = extract_dataset_name(df2_col)
            # Try to find matching counts column
            for counts_col in df_counts_exp2.columns:
                counts_short = extract_dataset_name(counts_col)
                if counts_short == df2_short:
                    counts_col_map_exp2[df2_col] = counts_col
                    break
        # Rename counts columns to match df2
        if counts_col_map_exp2:
            df_counts_exp2 = df_counts_exp2.rename(columns={v: k for k, v in counts_col_map_exp2.items()})
        df_counts_exp2 = df_counts_exp2.reindex(index=df2.index, columns=df2.columns)

    # Prepare data for plotting
    plot_data = []
    for model in df1.index:
        for dataset in df1.columns:
            exp1_val = df1.loc[model, dataset]
            exp2_val = df2.loc[model, dataset]
            
            # Skip if either value is NaN
            if pd.isna(exp1_val) or pd.isna(exp2_val):
                continue
            
            # Get counts for error bars
            n1 = None
            n2 = None
            if df_counts_exp1 is not None and model in df_counts_exp1.index and dataset in df_counts_exp1.columns:
                n1 = df_counts_exp1.loc[model, dataset]
            if df_counts_exp2 is not None and model in df_counts_exp2.index and dataset in df_counts_exp2.columns:
                n2 = df_counts_exp2.loc[model, dataset]
            
            # Calculate error bars if counts available
            exp1_err = None
            exp2_err = None
            if pd.notna(n1) and n1 > 0:
                _, _, se1 = calculate_binomial_ci(exp1_val, int(n1))
                exp1_err = 1.96 * se1  # 95% CI half-width
            if pd.notna(n2) and n2 > 0:
                _, _, se2 = calculate_binomial_ci(exp2_val, int(n2))
                exp2_err = 1.96 * se2  # 95% CI half-width
                
            plot_data.append({
                'model': model,
                'dataset': extract_dataset_name(dataset),
                'exp1': exp1_val,
                'exp2': exp2_val,
                'family': get_model_provider(model),
                'exp1_err': exp1_err,
                'exp2_err': exp2_err,
                'n1': int(n1) if pd.notna(n1) else None,
                'n2': int(n2) if pd.notna(n2) else None,
            })
    
    if not plot_data:
        print("  ⚠ No data to plot")
        return
    
    plot_df = pd.DataFrame(plot_data)
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Define shapes for datasets
    unique_datasets = plot_df['dataset'].unique()
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', 'X']
    dataset_markers = {ds: markers[i % len(markers)] for i, ds in enumerate(unique_datasets)}
    
    # Define colors for datasets (for regression lines)
    dataset_colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    dataset_line_colors = {ds: dataset_colors[i % len(dataset_colors)] for i, ds in enumerate(unique_datasets)}
    
    # Get unique families and assign colors
    unique_families = plot_df['family'].unique()
    family_colors_map = {}
    for family in unique_families:
        # Get color from first model in this family
        family_models = plot_df[plot_df['family'] == family]['model'].unique()
        if len(family_models) > 0:
            family_colors_map[family] = get_family_base_color(family_models[0])
        else:
            family_colors_map[family] = "#9ca3af"
    
    # Remove "Together-" prefix from family names for legend
    def clean_family_name(family):
        return family.replace("Together-", "") if family.startswith("Together-") else family
    
    # Determine extended range for lines (slightly before 0 and past 1, or data range)
    data_min = min(plot_df['exp1'].min(), plot_df['exp2'].min())
    data_max = max(plot_df['exp1'].max(), plot_df['exp2'].max())
    line_min = min(-0.05, data_min - 0.05)
    line_max = max(1.05, data_max + 0.05)
    
    # Track correlations per dataset for legend
    datasets_with_fit = {}
    
    # Plot points grouped by dataset
    for dataset in unique_datasets:
        dataset_data = plot_df[plot_df['dataset'] == dataset]
        marker = dataset_markers[dataset]
        
        # Group by family for this dataset
        for family in dataset_data['family'].unique():
            family_data = dataset_data[dataset_data['family'] == family]
            
            color = family_colors_map[family]
            
            # Get error bars for this family's data
            xerr = family_data['exp2_err'].values if 'exp2_err' in family_data.columns else None
            yerr = family_data['exp1_err'].values if 'exp1_err' in family_data.columns else None
            
            # Filter out None values for error bars
            if xerr is not None:
                xerr = np.array([e if pd.notna(e) and e is not None else 0 for e in xerr])
                if np.all(xerr == 0):
                    xerr = None
            if yerr is not None:
                yerr = np.array([e if pd.notna(e) and e is not None else 0 for e in yerr])
                if np.all(yerr == 0):
                    yerr = None
            
            ax.errorbar(
                family_data['exp2'],
                family_data['exp1'],
                xerr=xerr,
                yerr=yerr,
                fmt='none',
                ecolor=color,
                alpha=0.4,
                capsize=2,
                capthick=0.5,
                elinewidth=0.5,
                zorder=1
            )
            
            ax.scatter(
                family_data['exp2'],
                family_data['exp1'],
                c=color,
                marker=marker,
                s=100,
                alpha=0.7,
                edgecolors='black',
                linewidths=0.5,
                label=None,  # Custom legend will be created
                zorder=2
            )
        
        # Fit linear regression for this dataset with weighted regression if counts available
        x_vals = np.array(dataset_data['exp2'].values)
        y_vals = np.array(dataset_data['exp1'].values)
        
        if len(x_vals) > 1:
            # Calculate weights from counts (use geometric mean of n1 and n2, or individual if one missing)
            weights = None
            if 'n1' in dataset_data.columns and 'n2' in dataset_data.columns:
                n1_vals = dataset_data['n1'].values
                n2_vals = dataset_data['n2'].values
                # Use geometric mean of counts, or individual count if one is missing
                weights_list = []
                for n1, n2 in zip(n1_vals, n2_vals):
                    # Handle None values
                    n1_val = None if (pd.isna(n1) or n1 is None) else float(n1)
                    n2_val = None if (pd.isna(n2) or n2 is None) else float(n2)
                    
                    if n1_val is not None and n2_val is not None and n1_val > 0 and n2_val > 0:
                        # Geometric mean for combined weight
                        weights_list.append(np.sqrt(n1_val * n2_val))
                    elif n1_val is not None and n1_val > 0:
                        weights_list.append(n1_val)
                    elif n2_val is not None and n2_val > 0:
                        weights_list.append(n2_val)
                    else:
                        weights_list.append(np.nan)
                
                if len(weights_list) > 0 and not np.all(np.isnan(weights_list)):
                    weights = np.array(weights_list)
                    # Normalize weights (clip extreme values to prevent outliers from dominating)
                    valid_weights = weights[~np.isnan(weights)]
                    if len(valid_weights) > 0:
                        p5 = np.percentile(valid_weights, 5)
                        p95 = np.percentile(valid_weights, 95)
                        weights = np.clip(weights, p5, p95)
            
            # Use weighted regression if weights available
            if weights is not None and not np.all(np.isnan(weights)) and len(weights) == len(x_vals):
                # Filter out NaN values from all arrays
                valid_mask = ~(np.isnan(x_vals) | np.isnan(y_vals) | np.isnan(weights))
                if np.sum(valid_mask) >= 2:  # Need at least 2 points for regression
                    x_vals_filtered = x_vals[valid_mask]
                    y_vals_filtered = y_vals[valid_mask]
                    weights_filtered = weights[valid_mask]
                    
                    reg_result = weighted_regression_with_ci(x_vals_filtered, y_vals_filtered, weights=weights_filtered, x_min=line_min, x_max=line_max)
                    if reg_result:
                        correlation = weighted_correlation(x_vals_filtered, y_vals_filtered, weights_filtered)
                        datasets_with_fit[dataset] = correlation
                        # Use dataset color for the regression line
                        line_color = dataset_line_colors[dataset]
                        # Plot confidence band
                        ax.fill_between(
                            reg_result['x'],
                            reg_result['ci_lower'],
                            reg_result['ci_upper'],
                            color=line_color,
                            alpha=0.15,
                            zorder=0
                        )
                        # Plot regression line
                        ax.plot(
                            reg_result['x'],
                            reg_result['y_pred'],
                            color=line_color,
                            linestyle='--',
                            linewidth=1.5,
                            alpha=0.7,
                            label=None,  # Custom legend will be created
                            zorder=1
                        )
                    else:
                        # Fallback to unweighted if weighted regression failed
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals_filtered, y_vals_filtered)
                        datasets_with_fit[dataset] = r_value
                        x_line = np.linspace(line_min, line_max, 100)
                        y_line = slope * x_line + intercept
                        line_color = dataset_line_colors[dataset]
                        ax.plot(x_line, y_line, color=line_color, linestyle='--', linewidth=1.5, alpha=0.7, label=None)
                else:
                    # Not enough valid points for regression
                    pass
            else:
                # Unweighted regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_vals, y_vals)
                datasets_with_fit[dataset] = r_value
                
                # Plot regression line extending to full range
                x_line = np.linspace(line_min, line_max, 100)
                y_line = slope * x_line + intercept
                
                # Use dataset color for the regression line
                line_color = dataset_line_colors[dataset]
                
                ax.plot(
                    x_line,
                    y_line,
                    color=line_color,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=None  # Custom legend will be created
                )
    
    # Add 1:1 reference line (y = x) extending to full range
    # Use different color from chance lines (#888888 is lighter gray)
    ax.plot(
        [line_min, line_max],
        [line_min, line_max],
        color='#888888',
        linestyle=':',
        linewidth=2,
        alpha=0.7,
        label='1:1 line'
    )
    
    # Add horizontal and vertical lines at 0.5 (chance)
    ax.axhline(y=0.5, color='#555555', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.axvline(x=0.5, color='#555555', linestyle='--', linewidth=1.0, alpha=0.8)
    
    # Set axis labels
    ax.set_xlabel(f"{exp2_name} Performance", fontsize=12, fontweight="bold")
    ax.set_ylabel(f"{exp1_name} Performance", fontsize=12, fontweight="bold")
    
    # Set title
    title = f"Performance Comparison: {exp1_name} vs {exp2_name}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    
    # Set equal aspect ratio and limits (0 to 1 for performance)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Add grid
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=18)

    # Build 3-column legend: Model Name | Dataset | Misc. (column-major order)
    # Misc column includes Chance (0.5) and 1:1 line
    def _title_handle(text):
        return plt.Line2D([], [], linestyle="", marker="", label=text)

    def _empty_handle():
        return plt.Line2D([], [], linestyle="", marker="", label=" ")

    family_handles = []
    for family in sorted(unique_families):
        color = family_colors_map[family]
        display_name = provider_to_model_name(family)
        family_handles.append(
            plt.Line2D(
                [0], [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=display_name,
            )
        )

    dataset_handles = []
    dataset_labels = []
    for dataset in sorted(unique_datasets):
        marker = dataset_markers[dataset]
        line_color = dataset_line_colors[dataset]
        h_marker = plt.Line2D(
            [0], [0],
            marker=marker,
            color="w",
            markerfacecolor="gray",
            markersize=10,
            markeredgecolor="black",
            markeredgewidth=0.5,
        )
        if dataset in datasets_with_fit:
            h_line = plt.Line2D([0], [0], linestyle="--", color=line_color, linewidth=2)
            dataset_handles.append((h_marker, h_line))
            dataset_labels.append(f"{format_dataset_display_name(dataset)} (r={datasets_with_fit[dataset]:.2f})")
        else:
            dataset_handles.append(h_marker)
            dataset_labels.append(format_dataset_display_name(dataset))

    chance_handle = plt.Line2D(
        [0], [0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Chance (0.5)",
    )
    one_to_one_handle = plt.Line2D(
        [0], [0],
        color="#888888",
        linestyle=":",
        linewidth=2,
        alpha=0.7,
        label="1:1 line",
    )

    n_fam = len(family_handles)
    n_ds = len(dataset_handles)
    n_misc = 2  # Chance + 1:1
    max_data_rows = max(n_fam, n_ds, n_misc)

    col1_handles = [_title_handle("Model Name")]
    col1_labels = ["Model Name"]
    for i in range(max_data_rows):
        col1_handles.append(family_handles[i] if i < n_fam else _empty_handle())
        col1_labels.append(family_handles[i].get_label() if i < n_fam else " ")

    col2_handles = [_title_handle("Dataset")]
    col2_labels = ["Dataset"]
    for i in range(max_data_rows):
        col2_handles.append(dataset_handles[i] if i < n_ds else _empty_handle())
        col2_labels.append(dataset_labels[i] if i < n_ds else " ")

    col3_handles = [_title_handle("Misc.")]
    col3_labels = ["Misc."]
    col3_handles.append(chance_handle)
    col3_labels.append("Chance (0.5)")
    col3_handles.append(one_to_one_handle)
    col3_labels.append("1:1 line")
    for i in range(max_data_rows - n_misc):
        col3_handles.append(_empty_handle())
        col3_labels.append(" ")

    legend_handles = col1_handles + col2_handles + col3_handles
    legend_labels = col1_labels + col2_labels + col3_labels

    # Add legend (only if multiple families or datasets)
    if len(unique_families) > 1 or len(unique_datasets) > 1:
        ax.legend(
            handles=legend_handles,
            labels=legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.12),
            ncol=3,
            fontsize=9,
            framealpha=0.9,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            borderpad=1.5,
            labelspacing=1.2,
            handlelength=2.5,
            columnspacing=2.0,
        )
    
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved scatter plot to: {output_path}")
    save_figure_no_r_version(ax, output_path)
    save_figure_minimal_version(ax, output_path)
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
    parser.add_argument(
        "--reasoning_vs_instruct",
        action="store_true",
        help="Pair exp1 models ending with -thinking to exp2 instruct names (one row per base model).",
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

    reasoning_vs_instruct = getattr(args, "reasoning_vs_instruct", False)
    if reasoning_vs_instruct:
        print("Mode: reasoning vs instruct (pairing exp1 -thinking models to exp2 instruct names)\n")

    # Load and compute differences
    print("Loading and comparing performance data...")
    try:
        df_diff = load_and_compare(
            exp1_file, exp2_file,
            model_order=model_order,
            reasoning_vs_instruct=reasoning_vs_instruct,
        )
        print(
            f"  ✓ Computed differences: {df_diff.shape[0]} models × {df_diff.shape[1]} datasets"
        )

        # Also load original data for scatter plot
        df1_orig, df2_orig = load_original_data(
            exp1_file, exp2_file,
            model_order=model_order,
            reasoning_vs_instruct=reasoning_vs_instruct,
        )
        print(
            f"  ✓ Loaded original data: {df1_orig.shape[0]} models × {df1_orig.shape[1]} datasets\n"
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

    # Try to load counts data for error bars and significance testing
    counts_exp1_file = exp1_file.parent / "aggregated_counts.csv"
    counts_exp2_file = exp2_file.parent / "aggregated_counts.csv"
    df_counts_exp1 = None
    df_counts_exp2 = None
    
    if counts_exp1_file.exists():
        try:
            df_counts_exp1 = pd.read_csv(counts_exp1_file, index_col=0)
            print(f"  ✓ Loaded counts for {args.exp1_name}: {df_counts_exp1.shape[0]} models × {df_counts_exp1.shape[1]} datasets")
            print(f"    Models: {list(df_counts_exp1.index[:5])}..." if len(df_counts_exp1.index) > 5 else f"    Models: {list(df_counts_exp1.index)}")
            print(f"    Datasets: {list(df_counts_exp1.columns[:3])}..." if len(df_counts_exp1.columns) > 3 else f"    Datasets: {list(df_counts_exp1.columns)}")
        except Exception as e:
            print(f"  ⚠ Could not load counts for {args.exp1_name}: {e}")
    else:
        print(f"  ⚠ Counts file not found: {counts_exp1_file}")
    
    if counts_exp2_file.exists():
        try:
            df_counts_exp2 = pd.read_csv(counts_exp2_file, index_col=0)
            print(f"  ✓ Loaded counts for {args.exp2_name}: {df_counts_exp2.shape[0]} models × {df_counts_exp2.shape[1]} datasets")
            print(f"    Models: {list(df_counts_exp2.index[:5])}..." if len(df_counts_exp2.index) > 5 else f"    Models: {list(df_counts_exp2.index)}")
            print(f"    Datasets: {list(df_counts_exp2.columns[:3])}..." if len(df_counts_exp2.columns) > 3 else f"    Datasets: {list(df_counts_exp2.columns)}")
        except Exception as e:
            print(f"  ⚠ Could not load counts for {args.exp2_name}: {e}")
    else:
        print(f"  ⚠ Counts file not found: {counts_exp2_file}")
    
    if df_counts_exp1 is None or df_counts_exp2 is None:
        print(f"  ⚠ Warning: Missing counts data (error bars and significance will be omitted)\n")

    # In reasoning_vs_instruct mode, align count DFs to instruct-name index (for plots)
    if reasoning_vs_instruct and df_counts_exp1 is not None and df_counts_exp2 is not None:
        df1_temp = pd.read_csv(exp1_file, index_col=0)
        df2_temp = pd.read_csv(exp2_file, index_col=0)
        _thinking, _instruct = _get_thinking_instruct_pairs(df1_temp, df2_temp)
        instruct_to_thinking = dict(zip(_instruct, _thinking))
        reindex_instruct = [i for i in df1_orig.index if i in instruct_to_thinking]
        reindex_thinking = [instruct_to_thinking[i] for i in reindex_instruct]
        df_counts_exp1 = df_counts_exp1.reindex(reindex_thinking)
        df_counts_exp1.index = reindex_instruct
        df_counts_exp2 = df_counts_exp2.reindex(reindex_instruct)

    # Generate plots
    plot_path = output_dir / "performance_contrast.png"
    plot_diverging_stacked_bar_chart(df_diff, plot_path, args.exp1_name, args.exp2_name)
    print()
    
    # Generate grouped bar chart
    grouped_plot_path = output_dir / "performance_contrast_grouped.png"
    plot_grouped_bar_chart(
        df_diff, 
        grouped_plot_path, 
        args.exp1_name, 
        args.exp2_name,
        df_counts_exp1=df_counts_exp1,
        df_counts_exp2=df_counts_exp2,
        df_perf_exp1=df1_orig,
        df_perf_exp2=df2_orig
    )
    print()
    
    # Generate scatter plot
    scatter_path = output_dir / "performance_scatter.png"
    plot_performance_scatter(
        df1_orig, 
        df2_orig, 
        scatter_path, 
        args.exp1_name, 
        args.exp2_name,
        df_counts_exp1=df_counts_exp1,
        df_counts_exp2=df_counts_exp2
    )
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
    print("  • performance_contrast_grouped.png: Grouped bar chart with error bars and significance")
    print("  • performance_scatter.png: Scatter plot (exp1 vs exp2) with regression lines")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
