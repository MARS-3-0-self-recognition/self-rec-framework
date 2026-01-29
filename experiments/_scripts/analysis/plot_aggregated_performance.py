#!/usr/bin/env python3
"""
Plot aggregated evaluator performance charts.

This script loads aggregated_performance.csv (or deviation) and creates:
1. Stacked bar chart (models on y-axis, stacked datasets)
2. Grouped bar chart (models on x-axis, side-by-side datasets)

Usage:
    uv run experiments/_scripts/analysis/plot_aggregated_performance.py \
        --aggregated_file data/analysis/_aggregated_data/.../aggregated_performance.csv
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import (
    get_model_family_colors, 
    get_model_provider,
    calculate_binomial_ci,
    apply_fdr_correction,
    get_significance_marker
)


def extract_dataset_name(full_path: str) -> str:
    """
    Extract short dataset name from full path.
    """
    return full_path.split("/")[0]


def plot_stacked_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
    is_deviation: bool = False,
):
    """
    Create a stacked bar chart showing evaluator performance or deviation across datasets.
    """
    chart_type = "deviation" if is_deviation else "performance"
    print(f"Generating stacked bar chart ({chart_type})...")

    if df.empty:
        print(f"  ⚠ No data to plot (all models have zero {chart_type})")
        return

    # Work on a copy
    df_plot = df.copy()

    # Shorten dataset names for legend
    short_names = [extract_dataset_name(name) for name in df_plot.columns]
    df_plot.columns = short_names

    if is_deviation:
        # Diverging stacked bar chart logic
        df_positive = df_plot.copy()
        df_negative = df_plot.copy()
        df_positive[df_positive < 0] = 0
        df_negative[df_negative > 0] = 0

        # Sort by total absolute deviation
        df_plot["_total_abs"] = df_plot.abs().sum(axis=1)
        df_plot = df_plot.sort_values("_total_abs", ascending=False)
        df_plot = df_plot.drop(columns=["_total_abs"])
        df_positive = df_positive.reindex(df_plot.index)
        df_negative = df_negative.reindex(df_plot.index)

        fig, ax = plt.subplots(figsize=(14, max(8, len(df_plot) * 0.4)))

        # Colors
        n_datasets = len(df_plot.columns)
        if n_datasets <= 4:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
        elif n_datasets <= 8:
            colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

        # Stack negative
        bottom_neg = np.zeros(len(df_plot))
        for i, (dataset, color) in enumerate(zip(df_plot.columns, colors)):
            values = df_negative[dataset].values
            ax.barh(
                range(len(df_plot)),
                values,
                left=bottom_neg,
                label=dataset,
                color=color,
                alpha=0.7,
            )
            bottom_neg += values

        # Stack positive
        bottom_pos = np.zeros(len(df_plot))
        for i, (dataset, color) in enumerate(zip(df_plot.columns, colors)):
            values = df_positive[dataset].values
            ax.barh(
                range(len(df_plot)),
                values,
                left=bottom_pos,
                color=color,
                alpha=0.7,
            )
            bottom_pos += values

        # Reference line
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5, label="Chance (0.5)")
        
        # Limits
        max_pos = df_positive.sum(axis=1).max()
        max_neg = abs(df_negative.sum(axis=1).min())
        max_val = max(max_pos, max_neg) if not pd.isna(max(max_pos, max_neg)) else 1.0
        padding = max(max_val * 0.1, 0.05)
        ax.set_xlim(-max_val - padding, max_val + padding)
        
        xlabel = "Aggregated Deviation from Chance (Sum Across Datasets)"
        title = "Evaluator Deviation from Chance Across Datasets (Diverging Stacked)"

    else:
        # Regular stacked bar chart
        df_plot["_total"] = df_plot.sum(axis=1)
        df_plot = df_plot.sort_values("_total", ascending=False)
        df_plot = df_plot.drop(columns=["_total"])

        fig, ax = plt.subplots(figsize=(14, max(8, len(df_plot) * 0.4)))

        n_datasets = len(df_plot.columns)
        if n_datasets <= 4:
            colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
        elif n_datasets <= 8:
            colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
        else:
            colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

        bottom = np.zeros(len(df_plot))
        for i, (dataset, color) in enumerate(zip(df_plot.columns, colors)):
            values = df_plot[dataset].values
            ax.barh(range(len(df_plot)), values, left=bottom, label=dataset, color=color)
            bottom += values

        max_val = df_plot.sum(axis=1).max()
        padding = max(max_val * 0.1, 0.05)
        ax.set_xlim(0, max_val + padding)

        xlabel = "Aggregated Performance Score (Sum Across Datasets)"
        title = "Evaluator Performance Across Datasets (Stacked)"

    ax.set_yticks(range(len(df_plot)))
    ax.set_yticklabels(df_plot.index)
    ax.invert_yaxis()

    ax.set_xlabel(xlabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    ax.legend(loc="lower right" if not is_deviation else "upper right", fontsize=10)
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Add totals
    if is_deviation:
        totals = df_plot.sum(axis=1) # df_plot already sorted by absolute sum, but we want net sum for label position? 
        # Actually in original code we just used the values from the df passed in.
        # But here I modified df_plot (sorted it). Let's re-calculate sum.
        totals = df_plot.sum(axis=1)
        for i, (model, total) in enumerate(totals.items()):
            if total != 0:
                label_x = total + (padding * 0.02) if total >= 0 else total - (padding * 0.02)
                ha = "left" if total >= 0 else "right"
                ax.text(label_x, i, f"{total:.3f}", va="center", ha=ha, fontsize=9)
    else:
        totals = df_plot.sum(axis=1)
        for i, (model, total) in enumerate(totals.items()):
            if total > 0:
                ax.text(total + padding * 0.02, i, f"{total:.3f}", va="center", ha="left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved stacked bar chart to: {output_path}")
    plt.close()


def plot_grouped_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
    is_deviation: bool = False,
):
    """
    Create a grouped bar chart with models on x-axis and side-by-side dataset bars.
    """
    chart_type = "deviation" if is_deviation else "performance"
    print(f"Generating grouped bar chart ({chart_type})...")

    if df.empty:
        print(f"  ⚠ No data to plot")
        return

    # Work on a copy
    df_plot = df.copy()

    # Shorten dataset names for legend
    short_names = [extract_dataset_name(name) for name in df_plot.columns]
    df_plot.columns = short_names
    
    # Sort models
    if is_deviation:
        # Sort by total absolute deviation
        df_plot["_total_abs"] = df_plot.abs().sum(axis=1)
        df_plot = df_plot.sort_values("_total_abs", ascending=False)
        df_plot = df_plot.drop(columns=["_total_abs"])
    else:
        # Sort by total performance
        df_plot["_total"] = df_plot.sum(axis=1)
        df_plot = df_plot.sort_values("_total", ascending=False)
        df_plot = df_plot.drop(columns=["_total"])

    # Create figure
    # Width depends on number of models * number of datasets
    n_models = len(df_plot)
    n_datasets = len(df_plot.columns)
    fig_width = max(14, n_models * n_datasets * 0.15)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Define colors
    if n_datasets <= 4:
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"][:n_datasets]
    elif n_datasets <= 8:
        colors = plt.cm.tab10(np.linspace(0, 1, n_datasets))
    else:
        colors = plt.cm.Set3(np.linspace(0, 1, n_datasets))

    # Try to load counts data for error bars
    # First try aggregated_counts.csv (new format)
    counts_file = output_path.parent / "aggregated_counts.csv"
    if not counts_file.exists():
        # Fallback to old naming convention
        counts_file = output_path.parent / output_path.name.replace("_grouped.png", "_counts.csv").replace(".png", "_counts.csv")
    df_counts = None
    if counts_file.exists():
        try:
            df_counts = pd.read_csv(counts_file, index_col=0)
            # Shorten dataset names to match
            counts_short_names = [extract_dataset_name(name) for name in df_counts.columns]
            df_counts.columns = counts_short_names
            # Reindex to match df_plot
            df_counts = df_counts.reindex(df_plot.index)
            print(f"  ✓ Loaded sample counts from {counts_file.name}")
        except Exception as e:
            print(f"  ⚠ Could not load counts file: {e}")
    
    # Calculate error bars if counts available
    error_bars = None
    if df_counts is not None:
        # Calculate 95% CI for each bar
        errors = []
        for model in df_plot.index:
            model_errors = []
            for dataset in df_plot.columns:
                if model in df_counts.index and dataset in df_counts.columns:
                    perf = df_plot.loc[model, dataset]
                    n = df_counts.loc[model, dataset]
                    if pd.notna(n) and n > 0:
                        _, _, se = calculate_binomial_ci(perf, n)
                        model_errors.append(1.96 * se)  # 95% CI half-width
                    else:
                        model_errors.append(0)
                else:
                    model_errors.append(0)
            errors.append(model_errors)
        error_bars = np.array(errors).T  # Transpose to match pandas plot format
    
    # Calculate x positions for bars (used for both matplotlib and pandas plots)
    x = np.arange(len(df_plot.index))
    width = 0.8 / n_datasets
    
    # Plot grouped bars with error bars if available
    if error_bars is not None:
        # Use matplotlib bar plot to add error bars
        for i, (dataset, color) in enumerate(zip(df_plot.columns, colors)):
            offset = (i - n_datasets/2 + 0.5) * width
            values = df_plot[dataset].values
            yerr = error_bars[i] if error_bars.shape[0] > i else None
            ax.bar(
                x + offset,
                values,
                width=width * 0.9,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                yerr=yerr,
                error_kw={'capsize': 2, 'capthick': 0.5, 'elinewidth': 0.5, 'alpha': 0.6},
                label=dataset
            )
        ax.set_xticks(x)
        ax.set_xticklabels(df_plot.index, rotation=45, ha="right")
    else:
        # Fallback to pandas plot (no error bars)
        df_plot.plot(
            kind="bar",
            ax=ax,
            color=colors,
            width=0.8,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.5
        )
        # Get x positions from pandas plot for significance markers
        # Pandas uses 0, 1, 2, ... for x positions
        x = np.arange(len(df_plot.index))

    if is_deviation:
        # Add reference line at 0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.8)
        
        ylabel = "Deviation from Chance (0.5)"
        title = "Evaluator Deviation from Chance by Dataset"
    else:
        # Regular performance
        ylabel = "Performance Score"
        title = "Evaluator Performance by Dataset"
        
        # Add chance line at 0.5 (assuming individual performance is normalized 0-1?)
        # Actually in aggregated file, values are 0-1 accuracy.
        # But wait, stacked chart sums them up.
        # Here we are showing side-by-side, so each bar is the accuracy for that dataset.
        # So chance line at 0.5 makes sense for the bars.
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="Chance (0.5)")
        
        # Add significance markers if counts available
        if df_counts is not None:
            # Test each bar against chance (0.5) using binomial test
            from scipy import stats
            p_values = []
            for model in df_plot.index:
                for dataset in df_plot.columns:
                    if model in df_counts.index and dataset in df_counts.columns:
                        perf = df_plot.loc[model, dataset]
                        n = df_counts.loc[model, dataset]
                        if pd.notna(n) and n > 0 and pd.notna(perf):
                            # Binomial test: H0: p = 0.5
                            n_int = int(n)  # Ensure n is an integer
                            k = int(perf * n_int)  # Number of successes
                            from scipy.stats import binomtest
                            p_val = binomtest(k, n_int, p=0.5, alternative='two-sided').pvalue
                            p_values.append(p_val)
                        else:
                            p_values.append(np.nan)
                    else:
                        p_values.append(np.nan)
            
            # Apply FDR correction
            if len(p_values) > 0 and not np.all(np.isnan(p_values)):
                rejected, p_corrected = apply_fdr_correction(p_values)
                # Add markers to bars
                p_idx = 0
                # Use same x positions as bars (x is already defined above)
                if error_bars is None:
                    # For pandas plot, need to ensure width is defined
                    if 'width' not in locals():
                        width = 0.8 / n_datasets
                    
                for model_idx, model in enumerate(df_plot.index):
                    for dataset_idx, dataset in enumerate(df_plot.columns):
                        if p_idx < len(rejected) and rejected[p_idx]:
                            p_val = p_corrected[p_idx] if p_idx < len(p_corrected) else p_values[p_idx]
                            marker = get_significance_marker(p_values[p_idx], p_val)
                            if marker:
                                offset = (dataset_idx - n_datasets/2 + 0.5) * width
                                bar_height = df_plot.loc[model, dataset]
                                if error_bars is not None:
                                    yerr_val = error_bars[dataset_idx, model_idx] if error_bars.shape[0] > dataset_idx and error_bars.shape[1] > model_idx else 0.02
                                else:
                                    yerr_val = 0.02
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

    ax.set_xlabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    
    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Legend - add significance marker if any markers were added
    handles, labels = ax.get_legend_handles_labels()
    # Check if any significance markers were added
    has_significance = any('*' in str(text.get_text()) for text in ax.texts)
    if has_significance:
        # Add significance marker to legend
        from matplotlib.patches import Rectangle
        sig_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        handles.append(sig_handle)
        labels.append("* p < 0.05")
    ax.legend(handles=handles, labels=labels, title="Dataset", loc="center left", bbox_to_anchor=(1.02, 0.5))
    
    # Rotate x labels if many models
    plt.xticks(rotation=45, ha="right")
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved grouped bar chart to: {output_path}")
    plt.close()


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


def plot_dataset_grouped_bar_chart(
    df: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
    is_deviation: bool = False,
):
    """
    Create a bar chart grouped by dataset.
    Within each dataset group, all evaluator models are shown, sorted by performance (highest to lowest).
    Models are not labeled on the x-axis.
    Bars are colored by model family (OpenAI, Anthropic, Google, etc.) with a single color per family.
    """
    chart_type = "deviation" if is_deviation else "performance"
    print(f"Generating dataset-grouped bar chart ({chart_type})...")

    if df.empty:
        print(f"  ⚠ No data to plot")
        return

    # Work on a copy
    df_plot = df.copy()

    # Shorten dataset names for legend
    short_names = [extract_dataset_name(name) for name in df_plot.columns]
    df_plot.columns = short_names
    
    # Get all unique model names and create color mapping (single color per family)
    all_models = df_plot.index.tolist()
    model_color_map = {model: get_family_base_color(model) for model in all_models}
    
    # Sort each dataset column by performance (descending)
    # This will create a consistent ordering within each group
    n_models = len(df_plot)
    n_datasets = len(df_plot.columns)
    
    # Create figure
    # Width: space for 4 dataset groups, each with n_models bars
    bar_width = 0.8 / n_models  # Each bar width
    group_width = n_models * bar_width + 0.5  # Width of each group (with spacing)
    fig_width = max(14, n_datasets * group_width)
    fig, ax = plt.subplots(figsize=(fig_width, 8))

    # Get model families for legend (strip "Together-" prefix)
    model_families = {}
    for model in all_models:
        family = get_model_provider(model)
        # Remove "Together-" prefix from family names
        if family.startswith("Together-"):
            family = family.replace("Together-", "")
        model_families[model] = family
    
    # Create family to color mapping
    family_to_color = {}
    for model, family in model_families.items():
        if family not in family_to_color:
            family_to_color[family] = model_color_map.get(model, "#9ca3af")
    
    # Plot bars for each dataset group
    x_positions = []
    group_labels = []
    
    # Track which families we've seen for legend (to avoid duplicate entries)
    seen_families = set()
    
    for group_idx, dataset in enumerate(df_plot.columns):
        # Sort models by performance for this dataset (descending)
        dataset_data = df_plot[dataset].sort_values(ascending=False)
        
        # Calculate x positions for bars in this group
        group_start = group_idx * group_width
        bar_positions = [group_start + i * bar_width for i in range(len(dataset_data))]
        x_positions.extend(bar_positions)
        
        # Plot bars with model family colors (single color per family)
        for i, (model, value) in enumerate(dataset_data.items()):
            family = model_families.get(model, "Unknown")
            color = family_to_color.get(family, "#9ca3af")  # Use family color
            
            # Add label only for first occurrence of each family
            label = family if family not in seen_families else ""
            if family not in seen_families:
                seen_families.add(family)
            
            ax.bar(
                [bar_positions[i]],
                [value],
                width=bar_width * 0.9,
                color=color,
                alpha=0.8,
                edgecolor="black",
                linewidth=0.5,
                label=label
            )
        
        # Store group label position (center of group)
        group_center = group_start + (len(dataset_data) * bar_width) / 2
        group_labels.append((group_center, dataset))

    # Set x-axis: group labels at center of each group, no model labels
    group_centers = [pos for pos, _ in group_labels]
    group_names = [name for _, name in group_labels]
    ax.set_xticks(group_centers)
    ax.set_xticklabels(group_names, fontsize=11, fontweight="bold")
    
    # Remove minor ticks and labels
    ax.tick_params(axis='x', which='minor', length=0)

    if is_deviation:
        # Add reference line at 0
        ax.axhline(y=0, color="black", linestyle="-", linewidth=1, alpha=0.8)
        
        ylabel = "Deviation from Chance (0.5)"
        title = "Evaluator Deviation by Dataset\n(Models sorted by performance within each group)"
    else:
        # Regular performance
        ylabel = "Performance Score"
        title = "Evaluator Performance by Dataset\n(Models sorted by performance within each group)"
        
        # Add chance line at 0.5
        ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.8, label="Chance (0.5)")

    ax.set_xlabel("Dataset", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel, fontsize=12, fontweight="bold")
    
    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)
    
    # Grid
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Legend (model families, not datasets) - "Together-" prefix already removed
    ax.legend(title="Model Family", loc="best", fontsize=10)
    
    # Set x-axis limits with padding
    if x_positions:
        ax.set_xlim(min(x_positions) - bar_width, max(x_positions) + bar_width)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved dataset-grouped bar chart to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate charts from aggregated performance data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--aggregated_file",
        type=str,
        required=True,
        help="Path to aggregated_performance.csv or aggregated_deviation.csv",
    )
    
    args = parser.parse_args()
    
    file_path = Path(args.aggregated_file)
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"{'='*70}")
    print("GENERATING AGGREGATED CHARTS")
    print(f"{'='*70}")
    print(f"Input file: {file_path}\n")

    # Load data
    df = pd.read_csv(file_path, index_col=0)
    
    # Try to load aggregated counts if available (for error bars)
    # Check for aggregated_counts.csv (new format)
    counts_file = file_path.parent / "aggregated_counts.csv"
    if not counts_file.exists():
        # Fallback to old naming convention
        counts_file = file_path.parent / file_path.name.replace("performance.csv", "counts.csv").replace("deviation.csv", "counts.csv")
    
    if counts_file.exists():
        try:
            df_counts = pd.read_csv(counts_file, index_col=0)
            print(f"  ✓ Found counts file: {counts_file.name}")
            # Note: aggregated_counts.csv already has shortened names, so no need to process
        except Exception as e:
            print(f"  ⚠ Could not load counts file: {e}")
    
    # Determine type (performance or deviation) based on filename
    is_deviation = "deviation" in file_path.name
    
    # Extract experiment title from path
    experiment_title = ""
    path_parts = file_path.parts
    if len(path_parts) >= 2:
        exp_name = path_parts[-3] if len(path_parts) >= 3 else path_parts[-2]
        if "_" in exp_name:
            parts = exp_name.split("_", 1)
            if parts[0].isdigit():
                experiment_title = parts[1]
            else:
                experiment_title = exp_name
        else:
            experiment_title = exp_name

    output_dir = file_path.parent
    
    # 1. Stacked Bar Chart (Existing style)
    stacked_output = output_dir / file_path.name.replace(".csv", ".png")
    plot_stacked_bar_chart(
        df, 
        stacked_output, 
        experiment_title=experiment_title, 
        is_deviation=is_deviation
    )
    print()

    # 2. Grouped Bar Chart (New style: Models on X axis, side-by-side bars)
    grouped_output = output_dir / file_path.name.replace(".csv", "_grouped.png")
    plot_grouped_bar_chart(
        df, 
        grouped_output, 
        experiment_title=experiment_title, 
        is_deviation=is_deviation
    )
    print()

    # 3. Dataset-Grouped Bar Chart (New style: 4 groups by dataset, models sorted within each)
    dataset_grouped_output = output_dir / file_path.name.replace(".csv", "_dataset_grouped.png")
    plot_dataset_grouped_bar_chart(
        df, 
        dataset_grouped_output, 
        experiment_title=experiment_title, 
        is_deviation=is_deviation
    )
    print()

    print(f"{'='*70}")
    print("CHARTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
