#!/usr/bin/env python3
"""
Compare recognition performance vs the difference in LM Arena Ranking between Evaluator and Generator.

X-axis: Rank Distance = Evaluator Rank - Generator Rank
    - Positive value: Evaluator is worse ranked than Generator (e.g., Rank 10 vs Rank 1 -> Distance 9)
    - Negative value: Evaluator is better ranked than Generator (e.g., Rank 1 vs Rank 10 -> Distance -9)
    - Zero: Evaluator and Generator have same rank (or are same model)

Y-axis: Recognition Performance

Usage:
    uv run experiments/_scripts/analysis/rank_distance.py \
        --accuracy_files data/analysis/dataset1/.../accuracy_pivot.csv data/analysis/dataset2/.../accuracy_pivot.csv \
        --model_names -set dr

Output:
    - rank_distance.png: Scatter plot of performance vs rank distance
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.ticker import MultipleLocator
from utils import (
    expand_model_names,
    get_model_provider,
    provider_to_model_name,
    calculate_binomial_ci,
    weighted_regression_with_ci,
    weighted_correlation,
    apply_fdr_correction,
    get_significance_marker,
    format_evaluator_model_display_name,
    save_figure_minimal_version,
    save_figure_no_r_version,
)
from src.helpers.model_names import LM_ARENA_RANKINGS

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

def extract_dataset_name(full_path: str) -> str:
    """Extract short dataset name from full path."""
    # Assuming path structure: data/analysis/{dataset}/{subset}/...
    parts = Path(full_path).parts
    try:
        # Find 'analysis' and take the next part
        analysis_idx = parts.index("analysis")
        return parts[analysis_idx + 1]
    except ValueError:
        return parts[0]

def format_dataset_display_name(dataset_name: str) -> str:
    """Format dataset name for display in legends (neater capitalization)."""
    mapping = {
        "wikisum": "WikiSum",
        "pku_saferlhf": "PKU-SafeRLHF",
        "bigcodebench": "BigCodeBench",
        "sharegpt": "ShareGPT",
    }
    return mapping.get(dataset_name.lower(), dataset_name)

def adjust_data_points(data_points, self_scores):
    """
    Adjust data points for IND experiments by averaging with evaluator's self-score.
    
    Args:
        data_points: List of dicts with keys: 
                     'evaluator', 'generator', 'dataset', 'distance', 'performance'
        self_scores: Dict mapping evaluator -> averaged self-score across datasets
    
    Returns:
        List of adjusted data points with same structure
    """
    if not self_scores:
        return data_points
    
    adjusted_points = []
    for point in data_points:
        evaluator = point['evaluator']
        if evaluator in self_scores:
            self_score = self_scores[evaluator]
            cross_performance = point['performance']
            adjusted_performance = (cross_performance + self_score) / 2
            adjusted_point = point.copy()
            adjusted_point['performance'] = adjusted_performance
            adjusted_points.append(adjusted_point)
        else:
            # If no self-score available, skip this point
            continue
    
    return adjusted_points


def plot_rank_distance(data_points, output_path, experiment_title="", self_scores=None):
    """
    Plot performance vs rank distance.
    
    Args:
        data_points: List of dicts with keys: 
                     'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
        experiment_title: Optional title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    # Apply adjustment for IND experiments if self_scores provided
    adjusted_data = adjust_data_points(data_points, self_scores)
    if not adjusted_data:
        print("⚠ No data points after adjustment")
        return
    
    df = pd.DataFrame(adjusted_data)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define marker shapes for datasets
    dataset_markers = {
        "wikisum": "o",
        "sharegpt": "s",
        "pku_saferlhf": "^",
        "bigcodebench": "D",
    }
    
    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }

    # Assign markers to datasets found in data
    unique_datasets = sorted(df['dataset'].unique())
    dataset_to_marker = {}
    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]
    
    for i, ds in enumerate(unique_datasets):
        if ds in dataset_markers:
            dataset_to_marker[ds] = dataset_markers[ds]
        else:
            dataset_to_marker[ds] = marker_list[i % len(marker_list)]

    datasets_with_fit = {}  # Dict: dataset -> correlation
    plotted_datasets = set()
    
    # Calculate x-axis range from all data points for extending regression lines
    all_distances = df['distance'].values
    x_min = all_distances.min()
    x_max = all_distances.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_min -= padding
    x_max += padding

    # Plot points per dataset
    for dataset in unique_datasets:
        ds_data = df[df['dataset'] == dataset].copy()
        marker = dataset_to_marker[dataset]
        color = dataset_colors.get(dataset, "gray") # Use dataset color for points too? 
        # Or maybe color by family? The prompt implies looking at relative capability.
        # Let's stick to dataset colors for clarity of trends per dataset.
        
        # Calculate error bars if n_samples available
        has_error_bars = 'n_samples' in ds_data.columns and ds_data['n_samples'].notna().any()
        yerr = None
        if has_error_bars:
            # Calculate 95% CI for each point
            errors = []
            for _, row in ds_data.iterrows():
                if pd.notna(row['n_samples']) and row['n_samples'] > 0:
                    _, _, se = calculate_binomial_ci(row['performance'], row['n_samples'])
                    errors.append(1.96 * se)  # 95% CI half-width
                else:
                    errors.append(np.nan)
            yerr = np.array(errors)
        
        # Plot error bars first (if available)
        if has_error_bars and not np.all(np.isnan(yerr)):
            ax.errorbar(
                ds_data['distance'],
                ds_data['performance'],
                yerr=yerr,
                fmt='none',
                ecolor=color,
                alpha=0.4,
                capsize=2,
                capthick=0.5,
                linewidth=0.5,
                zorder=1
            )
        
        # Plot points on top
        ax.scatter(
            ds_data['distance'],
            ds_data['performance'],
            marker=marker,
            color=color, # Using dataset color for points
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label=None,
            zorder=2
        )
        
        plotted_datasets.add((dataset, marker, color))

        # Fit line with weighted regression
        if len(ds_data) >= 2:
            x_vals = ds_data['distance'].values
            y_vals = ds_data['performance'].values
            
            # Prepare weights for weighted regression
            weights = None
            if has_error_bars:
                n_vals = ds_data['n_samples'].values
                valid_n = ~np.isnan(n_vals) & (n_vals > 0)
                if np.any(valid_n):
                    # Weight by inverse variance (with clipping to prevent extreme weights)
                    p_clipped = np.clip(y_vals, 0.05, 0.95)
                    weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)
            
            # Use weighted regression if weights available
            if weights is not None and not np.all(np.isnan(weights)):
                reg_result = weighted_regression_with_ci(x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max)
                if reg_result:
                    correlation = weighted_correlation(x_vals, y_vals, weights)
                    # Plot confidence band
                    ax.fill_between(
                        reg_result['x'],
                        reg_result['ci_lower'],
                        reg_result['ci_upper'],
                        color=color,
                        alpha=0.2,
                        zorder=0
                    )
                    # Plot fit line
                    ax.plot(
                        reg_result['x'],
                        reg_result['y_pred'],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        color=color,
                        zorder=1
                    )
                    datasets_with_fit[dataset] = correlation if not np.isnan(correlation) else np.corrcoef(x_vals, y_vals)[0, 1]
                else:
                    # Fallback to unweighted
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_min, x_max, 100)
                    y_line = coeffs[0] * x_line + coeffs[1]
                    ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=color)
                    datasets_with_fit[dataset] = correlation
            else:
                # Unweighted regression (no confidence band)
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_min, x_max, 100)
                y_line = coeffs[0] * x_line + coeffs[1]
                ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=color)
                datasets_with_fit[dataset] = correlation

    # Custom Legend
    legend_handles = []
    legend_labels = []
    
    for dataset, marker, color in sorted(plotted_datasets, key=lambda x: x[0]):
        h_marker = plt.Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor=color,
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
        
        if dataset in datasets_with_fit:
            correlation = datasets_with_fit[dataset]
            h_line = plt.Line2D(
                [0], [0],
                linestyle="--",
                color=color,
                linewidth=2
            )
            legend_handles.append((h_marker, h_line))
            display_name = format_dataset_display_name(dataset)
            legend_labels.append(f"{display_name} (r={correlation:.2f})")
        else:
            legend_handles.append(h_marker)
            display_name = format_dataset_display_name(dataset)
            legend_labels.append(display_name)

    # Add chance line to legend at the end
    chance_handle = plt.Line2D(
        [0], [0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Chance (0.5)"
    )
    legend_handles.append(chance_handle)
    legend_labels.append("Chance (0.5)")

    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title="Datasets"
    )

    # Reference lines
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3, label="Equal Rank")

    # Labels
    ax.set_xlabel("Rank Distance (Evaluator Rank - Generator Rank)\nPositive = Evaluator is worse ranked", fontsize=12, fontweight="bold")
    y_label = "Adjusted Recognition Accuracy\n(Averaged with Evaluator Self-Score)" if self_scores else "Recognition Accuracy"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    
    full_title = "Recognition Performance vs Rank Distance"
    if self_scores:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)

    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved plot to: {output_path}")
    plt.close()

def plot_rank_distance_aggregated(data_points, output_path, experiment_title=""):
    """
    Plot aggregated performance vs rank distance across all datasets.
    Averages performance for each (Evaluator, Generator) pair over all datasets.
    
    Args:
        data_points: List of dicts with keys: 
                     'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    df = pd.DataFrame(data_points)
    
    # Aggregate over datasets for each (Evaluator, Generator) pair
    # For performance: average (weighted by n_samples if available)
    # For n_samples: sum (total samples across datasets)
    if 'n_samples' in df.columns:
        # Weighted average using sample counts
        df['weighted_perf'] = df['performance'] * df['n_samples'].fillna(0)
        aggregated = df.groupby(['evaluator', 'generator', 'distance']).agg({
            'weighted_perf': 'sum',
            'n_samples': 'sum'
        }).reset_index()
        aggregated['performance'] = aggregated['weighted_perf'] / aggregated['n_samples'].replace(0, np.nan)
        aggregated = aggregated.drop(columns=['weighted_perf'])
    else:
        # Simple average if no sample counts
        aggregated = df.groupby(['evaluator', 'generator', 'distance'])['performance'].mean().reset_index()
        aggregated['n_samples'] = None
    
    aggregated_df = aggregated
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate x-axis range for extending regression lines
    all_distances = aggregated_df['distance'].values
    x_min = all_distances.min()
    x_max = all_distances.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_min -= padding
    x_max += padding
    
    # Calculate error bars if n_samples available
    x_vals = aggregated_df['distance'].values
    y_vals = aggregated_df['performance'].values
    has_error_bars = 'n_samples' in aggregated_df.columns and aggregated_df['n_samples'].notna().any()
    
    yerr = None
    weights = None
    if has_error_bars:
        # Calculate 95% CI for each aggregated point
        errors = []
        n_vals = aggregated_df['n_samples'].values
        for i, (perf, n) in enumerate(zip(y_vals, n_vals)):
            if pd.notna(n) and n > 0:
                _, _, se = calculate_binomial_ci(perf, n)
                errors.append(1.96 * se)  # 95% CI half-width
            else:
                errors.append(np.nan)
        yerr = np.array(errors)
        
        # Prepare weights for weighted regression
        valid_n = ~np.isnan(n_vals) & (n_vals > 0)
        if np.any(valid_n):
            p_clipped = np.clip(y_vals, 0.05, 0.95)
            weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)
    
    # Plot error bars first (if available)
    if has_error_bars and yerr is not None and not np.all(np.isnan(yerr)):
        ax.errorbar(
            x_vals,
            y_vals,
            yerr=yerr,
            fmt='none',
            ecolor='#1f77b4',
            alpha=0.4,
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            zorder=1
        )
    
    # Plot Points
    ax.scatter(
        x_vals,
        y_vals,
        marker='o',
        color='#1f77b4', # Standard blue
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Avg Performance",
        zorder=2
    )
    
    # Calculate Global Fit with weighted regression if weights available
    if weights is not None and not np.all(np.isnan(weights)):
        reg_result = weighted_regression_with_ci(x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max)
        if reg_result:
            correlation = weighted_correlation(x_vals, y_vals, weights)
            # Plot confidence band
            ax.fill_between(
                reg_result['x'],
                reg_result['ci_lower'],
                reg_result['ci_upper'],
                color='black',
                alpha=0.15,
                zorder=0
            )
            # Plot fit line
            ax.plot(
                reg_result['x'],
                reg_result['y_pred'],
                linestyle="-",
                linewidth=3,
                alpha=0.9,
                color="black",
                label=f"Global Fit (r={correlation:.2f})",
                zorder=1
            )
        else:
            # Fallback to unweighted
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            coeffs = np.polyfit(x_vals, y_vals, 1)
            # Use the padded x_min and x_max calculated earlier
            x_line = np.linspace(x_min, x_max, 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, linestyle="-", linewidth=3, alpha=0.9, color="black", label=f"Global Fit (r={correlation:.2f})")
    else:
        # Unweighted regression
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        coeffs = np.polyfit(x_vals, y_vals, 1)
        # Use the padded x_min and x_max calculated earlier
        x_line = np.linspace(x_min, x_max, 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, linestyle="-", linewidth=3, alpha=0.9, color="black", label=f"Global Fit (r={correlation:.2f})")

    # Reference lines
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3, label="Equal Rank")
    
    # Get legend handles/labels and ensure Chance (0.5) is at the end
    handles, labels = ax.get_legend_handles_labels()
    # Remove Chance (0.5) if it exists, then add it at the end
    chance_idx = None
    for i, label in enumerate(labels):
        if label == "Chance (0.5)":
            chance_idx = i
            break
    if chance_idx is not None:
        handles.pop(chance_idx)
        labels.pop(chance_idx)
    # Add chance handle at the end
    chance_handle = plt.Line2D(
        [0], [0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Chance (0.5)"
    )
    handles.append(chance_handle)
    labels.append("Chance (0.5)")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9
    )

    # Labels
    ax.set_xlabel("Rank Distance (Evaluator Rank - Generator Rank)\nPositive = Evaluator is worse ranked", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Recognition Accuracy (across datasets)", fontsize=12, fontweight="bold")
    
    full_title = "Aggregated Performance vs Rank Distance"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)

    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Set x-axis limits to match the padded range used for regression lines
    ax.set_xlim(x_min, x_max)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    # Ensure tick label size 18 (figure 2e)
    ax.tick_params(axis="both", labelsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved aggregated plot to: {output_path}")
    save_figure_minimal_version(ax, output_path)
    plt.close()

def plot_rank_distance_adjusted(cross_model_points, self_scores, output_path, experiment_title="", self_score_n_samples=None):
    """
    Plot adjusted performance vs rank distance.
    Each cross-model comparison is adjusted by averaging with the evaluator's self-score.
    This corrects for bias where models that always answer "no" appear to perform well.
    
    Args:
        cross_model_points: List of dicts with keys:
                           'evaluator', 'generator', 'distance', 'performance', 'n_samples' (already aggregated)
        self_scores: Dict mapping evaluator -> averaged self-score across datasets
        output_path: Path to save plot
        self_score_n_samples: Optional dict mapping evaluator -> n_samples for self-scores
    """
    if not cross_model_points:
        print("⚠ No data points to plot")
        return

    # Adjust each cross-model point by averaging with evaluator's self-score
    adjusted_points = []
    for point in cross_model_points:
        evaluator = point['evaluator']
        if evaluator not in self_scores:
            continue  # Skip if we don't have self-score for this evaluator
        
        self_score = self_scores[evaluator]
        cross_performance = point['performance']
        adjusted_performance = (cross_performance + self_score) / 2
        
        # Calculate error propagation: SE_adjusted = sqrt(SE_cross² + SE_self²) / 2
        n_samples = None
        if 'n_samples' in point and pd.notna(point['n_samples']) and point['n_samples'] > 0:
            if self_score_n_samples and evaluator in self_score_n_samples and pd.notna(self_score_n_samples[evaluator]):
                # Both available: propagate errors
                _, _, se_cross = calculate_binomial_ci(cross_performance, point['n_samples'])
                _, _, se_self = calculate_binomial_ci(self_score, self_score_n_samples[evaluator])
                se_adjusted = np.sqrt(se_cross**2 + se_self**2) / 2
                # Convert back to effective n_samples for weighted regression
                # SE = sqrt(p(1-p)/n), so n = p(1-p)/SE²
                p_clipped = np.clip(adjusted_performance, 0.05, 0.95)
                n_samples = p_clipped * (1 - p_clipped) / (se_adjusted**2) if se_adjusted > 0 else None
            else:
                # Only cross available: use that
                n_samples = point['n_samples']
        
        adjusted_points.append({
            'evaluator': evaluator,
            'generator': point['generator'],
            'distance': point['distance'],
            'performance': adjusted_performance,
            'n_samples': n_samples
        })
    
    if not adjusted_points:
        print("⚠ No adjusted data points to plot (missing self-scores)")
        return
    
    df = pd.DataFrame(adjusted_points)
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate x-axis range for extending regression lines
    all_distances = df['distance'].values
    x_min = all_distances.min()
    x_max = all_distances.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_min -= padding
    x_max += padding
    
    # Calculate error bars if n_samples available
    x_vals = df['distance'].values
    y_vals = df['performance'].values
    has_error_bars = 'n_samples' in df.columns and df['n_samples'].notna().any()
    
    yerr = None
    weights = None
    if has_error_bars:
        # Calculate 95% CI for each adjusted point
        errors = []
        n_vals = df['n_samples'].values
        for i, (perf, n) in enumerate(zip(y_vals, n_vals)):
            if pd.notna(n) and n > 0:
                _, _, se = calculate_binomial_ci(perf, n)
                errors.append(1.96 * se)  # 95% CI half-width
            else:
                errors.append(np.nan)
        yerr = np.array(errors)
        
        # Prepare weights for weighted regression
        valid_n = ~np.isnan(n_vals) & (n_vals > 0)
        if np.any(valid_n):
            p_clipped = np.clip(y_vals, 0.05, 0.95)
            weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)
    
    # Plot error bars first (if available)
    if has_error_bars and yerr is not None and not np.all(np.isnan(yerr)):
        ax.errorbar(
            x_vals,
            y_vals,
            yerr=yerr,
            fmt='none',
            ecolor='#1f77b4',
            alpha=0.4,
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            zorder=1
        )
    
    # Plot Points
    ax.scatter(
        x_vals,
        y_vals,
        marker='o',
        color='#1f77b4', # Standard blue
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Adjusted Performance",
        zorder=2
    )
    
    # Calculate Global Fit with weighted regression if weights available
    if weights is not None and not np.all(np.isnan(weights)):
        reg_result = weighted_regression_with_ci(x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max)
        if reg_result:
            correlation = weighted_correlation(x_vals, y_vals, weights)
            # Plot confidence band
            ax.fill_between(
                reg_result['x'],
                reg_result['ci_lower'],
                reg_result['ci_upper'],
                color='black',
                alpha=0.15,
                zorder=0
            )
            # Plot fit line
            ax.plot(
                reg_result['x'],
                reg_result['y_pred'],
                linestyle="-",
                linewidth=3,
                alpha=0.9,
                color="black",
                label=f"Global Fit (r={correlation:.2f})",
                zorder=1
            )
        else:
            # Fallback to unweighted
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            coeffs = np.polyfit(x_vals, y_vals, 1)
            # Use the padded x_min and x_max calculated earlier
            x_line = np.linspace(x_min, x_max, 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            ax.plot(x_line, y_line, linestyle="-", linewidth=3, alpha=0.9, color="black", label=f"Global Fit (r={correlation:.2f})")
    else:
        # Unweighted regression
        correlation = np.corrcoef(x_vals, y_vals)[0, 1]
        coeffs = np.polyfit(x_vals, y_vals, 1)
        # Use the padded x_min and x_max calculated earlier
        x_line = np.linspace(x_min, x_max, 100)
        y_line = coeffs[0] * x_line + coeffs[1]
        ax.plot(x_line, y_line, linestyle="-", linewidth=3, alpha=0.9, color="black", label=f"Global Fit (r={correlation:.2f})")

    # Reference lines
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3, label="Equal Rank")
    
    # Get legend handles/labels and ensure Chance (0.5) is at the end
    handles, labels = ax.get_legend_handles_labels()
    # Remove Chance (0.5) if it exists, then add it at the end
    chance_idx = None
    for i, label in enumerate(labels):
        if label == "Chance (0.5)":
            chance_idx = i
            break
    if chance_idx is not None:
        handles.pop(chance_idx)
        labels.pop(chance_idx)
    # Add chance handle at the end
    chance_handle = plt.Line2D(
        [0], [0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Chance (0.5)"
    )
    handles.append(chance_handle)
    labels.append("Chance (0.5)")

    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9
    )

    # Labels
    ax.set_xlabel("Rank Distance (Evaluator Rank - Generator Rank)\nPositive = Evaluator is worse ranked", fontsize=12, fontweight="bold")
    ax.set_ylabel("Adjusted Recognition Accuracy\n(Averaged with Evaluator Self-Score)", fontsize=12, fontweight="bold")
    
    full_title = "Adjusted Performance vs Rank Distance\n(Controlled for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)

    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Set x-axis limits to match the padded range used for regression lines
    ax.set_xlim(x_min, x_max)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    # Ensure tick label size 18 (figure 2f)
    ax.tick_params(axis="both", labelsize=18)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved adjusted plot to: {output_path}")
    save_figure_minimal_version(ax, output_path)
    plt.close()

def plot_rank_distance_filtered_by_evaluator_rank(data_points, output_path, experiment_title="", self_scores=None):
    """
    Plot performance vs evaluator rank for model pairs with rank distance between -20 and 20.
    Shows all data points from all datasets for these filtered pairs, with fit lines per dataset.
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
        experiment_title: Title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    # Apply adjustment for IND experiments if self_scores provided
    adjusted_data = adjust_data_points(data_points, self_scores)
    if not adjusted_data:
        print("⚠ No data points after adjustment")
        return
    
    df = pd.DataFrame(adjusted_data)
    
    # Filter to only pairs with rank distance between -20 and 20
    filtered_df = df[(df['distance'] > -20) & (df['distance'] < 20)].copy()
    
    if len(filtered_df) == 0:
        print("⚠ No data points with rank distance between -20 and 20")
        return
    
    # Get evaluator ranks for filtered pairs
    evaluator_ranks = []
    for evaluator in filtered_df['evaluator'].unique():
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_ranks.append((evaluator, rank))
    
    if not evaluator_ranks:
        print("⚠ No evaluators with valid LM Arena rankings found")
        return
    
    # Create mapping from evaluator to rank
    evaluator_to_rank = dict(evaluator_ranks)
    
    # Add evaluator rank column
    filtered_df['evaluator_rank'] = filtered_df['evaluator'].map(evaluator_to_rank)
    
    # Remove rows where evaluator rank is missing
    filtered_df = filtered_df.dropna(subset=['evaluator_rank'])
    
    if len(filtered_df) == 0:
        print("⚠ No data points after adding evaluator ranks")
        return
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define marker shapes for datasets
    dataset_markers = {
        "wikisum": "o",
        "sharegpt": "s",
        "pku_saferlhf": "^",
        "bigcodebench": "D",
    }
    
    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }
    
    # Get family colors for evaluators
    unique_evaluators = filtered_df['evaluator'].unique()
    evaluator_family_colors = {}
    for evaluator in unique_evaluators:
        evaluator_family_colors[evaluator] = get_family_base_color(evaluator)
    
    # Add family color column
    filtered_df['family_color'] = filtered_df['evaluator'].map(evaluator_family_colors)
    
    unique_datasets = sorted(filtered_df['dataset'].unique())
    datasets_with_fit = {}  # Dict: dataset -> correlation
    plotted_datasets = set()
    
    # Calculate x-axis range for extending regression lines (based on evaluator ranks)
    # Start at 0 to match figure 2c/2d (performance_vs_arena_ranking)
    all_ranks = filtered_df['evaluator_rank'].values
    x_min = 0
    x_max = all_ranks.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_max += padding
    
    # Plot points and fit lines per dataset
    for dataset in unique_datasets:
        ds_data = filtered_df[filtered_df['dataset'] == dataset]
        
        if len(ds_data) == 0:
            continue
        
        marker = dataset_markers.get(dataset, "o")
        line_color = dataset_colors.get(dataset, "gray")
        
        # Calculate error bars if n_samples available
        has_error_bars = 'n_samples' in ds_data.columns and ds_data['n_samples'].notna().any()
        if has_error_bars:
            # Plot error bars first
            for _, row in ds_data.iterrows():
                if pd.notna(row['n_samples']) and row['n_samples'] > 0:
                    _, _, se = calculate_binomial_ci(row['performance'], row['n_samples'])
                    yerr = 1.96 * se  # 95% CI half-width
                    ax.errorbar(
                        row['evaluator_rank'],
                        row['performance'],
                        yerr=yerr,
                        fmt='none',
                        ecolor=row['family_color'],
                        alpha=0.3,
                        capsize=1.5,
                        capthick=0.5,
                        linewidth=0.5,
                        zorder=1
                    )
        
        # Plot points colored by evaluator family
        for _, row in ds_data.iterrows():
            ax.scatter(
                row['evaluator_rank'],
                row['performance'],
                marker=marker,
                color=row['family_color'],
                s=80,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
                label=None,
                zorder=2
            )
        
        plotted_datasets.add((dataset, marker, line_color))
        
        # Fit line per dataset with weighted regression
        if len(ds_data) >= 2:
            x_vals = ds_data['evaluator_rank'].values
            y_vals = ds_data['performance'].values
            
            # Prepare weights for weighted regression
            weights = None
            if has_error_bars:
                n_vals = ds_data['n_samples'].values
                valid_n = ~np.isnan(n_vals) & (n_vals > 0)
                if np.any(valid_n):
                    p_clipped = np.clip(y_vals, 0.05, 0.95)
                    weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)
            
            # Use weighted regression if weights available
            if weights is not None and not np.all(np.isnan(weights)):
                reg_result = weighted_regression_with_ci(x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max)
                if reg_result:
                    correlation = weighted_correlation(x_vals, y_vals, weights)
                    # Plot confidence band
                    ax.fill_between(
                        reg_result['x'],
                        reg_result['ci_lower'],
                        reg_result['ci_upper'],
                        color=line_color,
                        alpha=0.15,
                        zorder=0
                    )
                    # Plot fit line
                    ax.plot(
                        reg_result['x'],
                        reg_result['y_pred'],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        color=line_color,
                        zorder=1
                    )
                    datasets_with_fit[dataset] = correlation if not np.isnan(correlation) else np.corrcoef(x_vals, y_vals)[0, 1]
                else:
                    # Fallback to unweighted
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_min, x_max, 100)
                    y_line = coeffs[0] * x_line + coeffs[1]
                    ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=line_color)
                    datasets_with_fit[dataset] = correlation
            else:
                # Unweighted regression
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_min, x_max, 100)
                y_line = coeffs[0] * x_line + coeffs[1]
                ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=line_color)
                datasets_with_fit[dataset] = correlation
    
    # Reference line
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    
    # Build 3-column legend: Model Name | Dataset | Misc. (column-major order)
    def _title_handle(text):
        return plt.Line2D([], [], linestyle="", marker="", label=text)

    def _empty_handle():
        return plt.Line2D([], [], linestyle="", marker="", label=" ")

    unique_families = {}
    for evaluator in unique_evaluators:
        family = get_model_provider(evaluator)
        if family not in unique_families:
            unique_families[family] = evaluator_family_colors[evaluator]

    family_handles = []
    for family, color in sorted(unique_families.items()):
        display_name = provider_to_model_name(family)
        h = plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color,
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5,
            linestyle='None',
            label=display_name,
        )
        family_handles.append(h)

    dataset_handles = []
    dataset_labels = []
    for dataset, marker, line_color in sorted(plotted_datasets, key=lambda x: x[0]):
        h_marker = plt.Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor='gray',
            markersize=10,
            markeredgecolor='black',
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

    n_fam = len(family_handles)
    n_ds = len(dataset_handles)
    max_data_rows = max(n_fam, n_ds, 1)

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
    for i in range(max_data_rows - 1):
        col3_handles.append(_empty_handle())
        col3_labels.append(" ")

    legend_handles = col1_handles + col2_handles + col3_handles
    legend_labels = col1_labels + col2_labels + col3_labels

    # Labels
    ax.set_xlabel("Evaluator LM Arena Rank\n(Lower rank = Higher capability)", fontsize=12, fontweight="bold")
    y_label = "Adjusted Recognition Accuracy\n(Averaged with Evaluator Self-Score)" if self_scores else "Recognition Accuracy"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    
    full_title = "Performance vs Evaluator Rank\n(Rank Distance: -20 < distance < 20)"
    if self_scores:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)
    
    # Place legend below the chart (3-column format)
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
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    # X-axis: start at 0 and count up to maximum (0 on left, max on right)
    ax.set_xlim(0, x_max)
    # Ensure tick label size 18 (figures 3a, 3b)
    ax.tick_params(axis="both", labelsize=18)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved filtered evaluator rank plot to: {output_path}")
    save_figure_no_r_version(ax, output_path)
    save_figure_minimal_version(ax, output_path)
    plt.close()

def plot_rank_distance_vs_evaluator_rank(data_points, output_dir, experiment_title="", self_scores=None):
    """
    Plot evaluator rank vs generator rank with performance as color (red to green heatmap).
    Creates separate plots for each dataset.
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_dir: Directory to save plots (will create one file per dataset)
        experiment_title: Title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    # Apply adjustment for IND experiments if self_scores provided
    adjusted_data = adjust_data_points(data_points, self_scores)
    if not adjusted_data:
        print("⚠ No data points after adjustment")
        return
    
    df = pd.DataFrame(adjusted_data)
    
    # Get evaluator and generator ranks
    evaluator_ranks = {}
    generator_ranks = {}
    
    for evaluator in df['evaluator'].unique():
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_ranks[evaluator] = rank
    
    for generator in df['generator'].unique():
        rank = get_model_arena_ranking(generator)
        if rank is not None:
            generator_ranks[generator] = rank
    
    if not evaluator_ranks or not generator_ranks:
        print("⚠ No models with valid LM Arena rankings found")
        return
    
    # Add rank columns
    df['evaluator_rank'] = df['evaluator'].map(evaluator_ranks)
    df['generator_rank'] = df['generator'].map(generator_ranks)
    
    # Remove rows where ranks are missing
    df = df.dropna(subset=['evaluator_rank', 'generator_rank'])
    
    if len(df) == 0:
        print("⚠ No data points after adding ranks")
        return
    
    # Create colormap from red (0) to green (1)
    # Using RdYlGn (not reversed) gives us red for low values, green for high values
    cmap = plt.cm.RdYlGn
    
    # Fixed normalization from 0 to 1
    norm = plt.Normalize(vmin=0, vmax=1)
    
    # Get unique datasets
    unique_datasets = sorted(df['dataset'].unique())
    
    # Create a separate plot for each dataset
    for dataset in unique_datasets:
        ds_data = df[df['dataset'] == dataset]
        
        if len(ds_data) == 0:
            continue
        
        # Set up plot
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Plot points with color based on performance
        scatter = ax.scatter(
            ds_data['generator_rank'],
            ds_data['evaluator_rank'],
            c=ds_data['performance'],
            cmap=cmap,
            vmin=0,
            vmax=1,
            s=100,
            alpha=0.7,
            edgecolors="black",
            linewidths=0.5
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Performance Score', fontsize=11, fontweight="bold")
        
        # Reference line for equal ranks (diagonal)
        rank_min = min(df['evaluator_rank'].min(), df['generator_rank'].min())
        rank_max = max(df['evaluator_rank'].max(), df['generator_rank'].max())
        ax.plot([rank_min, rank_max], [rank_min, rank_max], 
                color="black", linestyle="--", linewidth=1, alpha=0.3, label="Equal Rank")
        
        # Labels
        ax.set_xlabel("Generator LM Arena Rank\n(Lower rank = Higher capability)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Evaluator LM Arena Rank\n(Lower rank = Higher capability)", fontsize=12, fontweight="bold")
        
        display_name = format_dataset_display_name(dataset)
        full_title = f"Evaluator Rank vs Generator Rank\n{display_name}\n(Colored by Performance)"
        if self_scores:
            full_title += "\n(Adjusted for Self-Recognition Bias)"
        if experiment_title:
            full_title += f"\n{experiment_title}"
        ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)
        
        # Grid
        ax.grid(alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)
        
        # Invert both axes so lower rank numbers (better models) are on the right/top
        ax.invert_xaxis()
        ax.invert_yaxis()
        
        # Set equal aspect ratio for better visualization
        ax.set_aspect('equal', adjustable='box')
        
        plt.tight_layout()
        
        # Save with dataset name in filename
        output_path = Path(output_dir) / f"evaluator_rank_vs_generator_rank_{dataset}.png"
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"  ✓ Saved {dataset} plot to: {output_path}")
        plt.close()

def plot_rank_distance_filtered_by_evaluator_rank_positive(data_points, output_path, experiment_title="", self_scores=None):
    """
    Plot performance vs evaluator rank for model pairs with rank distance > -20.
    Shows all data points from all datasets for these filtered pairs, with fit lines per dataset.
    This includes all positive distances and distances between -20 and 0.
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
        experiment_title: Title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    # Apply adjustment for IND experiments if self_scores provided
    adjusted_data = adjust_data_points(data_points, self_scores)
    if not adjusted_data:
        print("⚠ No data points after adjustment")
        return
    
    df = pd.DataFrame(adjusted_data)
    
    # Filter to only pairs with rank distance > -20 (includes all positive and up to -20)
    filtered_df = df[df['distance'] > -20].copy()
    
    if len(filtered_df) == 0:
        print("⚠ No data points with rank distance > -20")
        return
    
    # Get evaluator ranks for filtered pairs
    evaluator_ranks = []
    for evaluator in filtered_df['evaluator'].unique():
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_ranks.append((evaluator, rank))
    
    if not evaluator_ranks:
        print("⚠ No evaluators with valid LM Arena rankings found")
        return
    
    # Create mapping from evaluator to rank
    evaluator_to_rank = dict(evaluator_ranks)
    
    # Add evaluator rank column
    filtered_df['evaluator_rank'] = filtered_df['evaluator'].map(evaluator_to_rank)
    
    # Remove rows where evaluator rank is missing
    filtered_df = filtered_df.dropna(subset=['evaluator_rank'])
    
    if len(filtered_df) == 0:
        print("⚠ No data points after adding evaluator ranks")
        return
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define marker shapes for datasets
    dataset_markers = {
        "wikisum": "o",
        "sharegpt": "s",
        "pku_saferlhf": "^",
        "bigcodebench": "D",
    }
    
    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }
    
    # Get family colors for evaluators
    unique_evaluators = filtered_df['evaluator'].unique()
    evaluator_family_colors = {}
    for evaluator in unique_evaluators:
        evaluator_family_colors[evaluator] = get_family_base_color(evaluator)
    
    # Add family color column
    filtered_df['family_color'] = filtered_df['evaluator'].map(evaluator_family_colors)
    
    unique_datasets = sorted(filtered_df['dataset'].unique())
    datasets_with_fit = {}  # Dict: dataset -> correlation
    plotted_datasets = set()
    
    # Calculate x-axis range for extending regression lines (based on evaluator ranks)
    # Start at 0 to match figure 2c/2d (performance_vs_arena_ranking)
    all_ranks = filtered_df['evaluator_rank'].values
    x_min = 0
    x_max = all_ranks.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_max += padding
    
    # Plot points and fit lines per dataset
    for dataset in unique_datasets:
        ds_data = filtered_df[filtered_df['dataset'] == dataset]
        
        if len(ds_data) == 0:
            continue
        
        marker = dataset_markers.get(dataset, "o")
        line_color = dataset_colors.get(dataset, "gray")
        
        # Calculate error bars if n_samples available
        has_error_bars = 'n_samples' in ds_data.columns and ds_data['n_samples'].notna().any()
        if has_error_bars:
            # Plot error bars first
            for _, row in ds_data.iterrows():
                if pd.notna(row['n_samples']) and row['n_samples'] > 0:
                    _, _, se = calculate_binomial_ci(row['performance'], row['n_samples'])
                    yerr = 1.96 * se  # 95% CI half-width
                    ax.errorbar(
                        row['evaluator_rank'],
                        row['performance'],
                        yerr=yerr,
                        fmt='none',
                        ecolor=row['family_color'],
                        alpha=0.3,
                        capsize=1.5,
                        capthick=0.5,
                        linewidth=0.5,
                        zorder=1
                    )
        
        # Plot points colored by evaluator family
        for _, row in ds_data.iterrows():
            ax.scatter(
                row['evaluator_rank'],
                row['performance'],
                marker=marker,
                color=row['family_color'],
                s=80,
                alpha=0.6,
                edgecolors="black",
                linewidths=0.5,
                label=None,
                zorder=2
            )
        
        plotted_datasets.add((dataset, marker, line_color))
        
        # Fit line per dataset with weighted regression
        if len(ds_data) >= 2:
            x_vals = ds_data['evaluator_rank'].values
            y_vals = ds_data['performance'].values
            
            # Prepare weights for weighted regression
            weights = None
            if has_error_bars:
                n_vals = ds_data['n_samples'].values
                valid_n = ~np.isnan(n_vals) & (n_vals > 0)
                if np.any(valid_n):
                    p_clipped = np.clip(y_vals, 0.05, 0.95)
                    weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)
            
            # Use weighted regression if weights available
            if weights is not None and not np.all(np.isnan(weights)):
                reg_result = weighted_regression_with_ci(x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max)
                if reg_result:
                    correlation = weighted_correlation(x_vals, y_vals, weights)
                    # Plot confidence band
                    ax.fill_between(
                        reg_result['x'],
                        reg_result['ci_lower'],
                        reg_result['ci_upper'],
                        color=line_color,
                        alpha=0.15,
                        zorder=0
                    )
                    # Plot fit line
                    ax.plot(
                        reg_result['x'],
                        reg_result['y_pred'],
                        linestyle="--",
                        linewidth=2,
                        alpha=0.8,
                        color=line_color,
                        zorder=1
                    )
                    datasets_with_fit[dataset] = correlation if not np.isnan(correlation) else np.corrcoef(x_vals, y_vals)[0, 1]
                else:
                    # Fallback to unweighted
                    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                    coeffs = np.polyfit(x_vals, y_vals, 1)
                    x_line = np.linspace(x_min, x_max, 100)
                    y_line = coeffs[0] * x_line + coeffs[1]
                    ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=line_color)
                    datasets_with_fit[dataset] = correlation
            else:
                # Unweighted regression
                correlation = np.corrcoef(x_vals, y_vals)[0, 1]
                coeffs = np.polyfit(x_vals, y_vals, 1)
                x_line = np.linspace(x_min, x_max, 100)
                y_line = coeffs[0] * x_line + coeffs[1]
                ax.plot(x_line, y_line, linestyle="--", linewidth=2, alpha=0.8, color=line_color)
                datasets_with_fit[dataset] = correlation
    
    # Reference line
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    
    # Build 3-column legend: Model Name | Dataset | Misc. (column-major order)
    def _title_handle(text):
        return plt.Line2D([], [], linestyle="", marker="", label=text)

    def _empty_handle():
        return plt.Line2D([], [], linestyle="", marker="", label=" ")

    unique_families = {}
    for evaluator in unique_evaluators:
        family = get_model_provider(evaluator)
        if family not in unique_families:
            unique_families[family] = evaluator_family_colors[evaluator]

    family_handles = []
    for family, color in sorted(unique_families.items()):
        display_name = provider_to_model_name(family)
        h = plt.Line2D(
            [0], [0],
            marker='o',
            color='w',
            markerfacecolor=color,
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5,
            linestyle='None',
            label=display_name,
        )
        family_handles.append(h)

    dataset_handles = []
    dataset_labels = []
    for dataset, marker, line_color in sorted(plotted_datasets, key=lambda x: x[0]):
        h_marker = plt.Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor='gray',
            markersize=10,
            markeredgecolor='black',
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

    n_fam = len(family_handles)
    n_ds = len(dataset_handles)
    max_data_rows = max(n_fam, n_ds, 1)

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
    for i in range(max_data_rows - 1):
        col3_handles.append(_empty_handle())
        col3_labels.append(" ")

    legend_handles = col1_handles + col2_handles + col3_handles
    legend_labels = col1_labels + col2_labels + col3_labels

    # Labels
    ax.set_xlabel("Evaluator LM Arena Rank\n(Lower rank = Higher capability)", fontsize=12, fontweight="bold")
    y_label = "Adjusted Recognition Accuracy\n(Averaged with Evaluator Self-Score)" if self_scores else "Recognition Accuracy"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    
    full_title = "Performance vs Evaluator Rank\n(Rank Distance: distance > -20)"
    if self_scores:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)
    
    # Place legend below the chart (3-column format)
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
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    # X-axis: start at 0 and count up to maximum (0 on left, max on right)
    ax.set_xlim(0, x_max)
    # Ensure tick label size 18 (figures 3c, 3d)
    ax.tick_params(axis="both", labelsize=18)
    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved filtered evaluator rank plot (positive) to: {output_path}")
    save_figure_no_r_version(ax, output_path)
    save_figure_minimal_version(ax, output_path)
    plt.close()

def plot_rank_distance_filtered_by_evaluator_rank_positive_averaged(data_points, output_path, experiment_title="", self_scores=None):
    """
    Plot performance vs evaluator rank for model pairs with rank distance > -20.
    Averages performance for each evaluator over all generator models per dataset.
    Shows one point per (evaluator, dataset) pair, with fit lines per dataset.
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
        experiment_title: Title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    # Apply adjustment for IND experiments if self_scores provided
    adjusted_data = adjust_data_points(data_points, self_scores)
    if not adjusted_data:
        print("⚠ No data points after adjustment")
        return
    
    df = pd.DataFrame(adjusted_data)
    
    # Filter to only pairs with rank distance > -20 (includes all positive and up to -20)
    filtered_df = df[df['distance'] > -20].copy()
    
    if len(filtered_df) == 0:
        print("⚠ No data points with rank distance > -20")
        return
    
    # Get evaluator ranks for filtered pairs
    evaluator_ranks = []
    for evaluator in filtered_df['evaluator'].unique():
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_ranks.append((evaluator, rank))
    
    if not evaluator_ranks:
        print("⚠ No evaluators with valid LM Arena rankings found")
        return
    
    # Create mapping from evaluator to rank
    evaluator_to_rank = dict(evaluator_ranks)
    
    # Add evaluator rank column
    filtered_df['evaluator_rank'] = filtered_df['evaluator'].map(evaluator_to_rank)
    
    # Remove rows where evaluator rank is missing
    filtered_df = filtered_df.dropna(subset=['evaluator_rank'])
    
    if len(filtered_df) == 0:
        print("⚠ No data points after adding evaluator ranks")
        return
    
    # Average performance for each evaluator over all generators per dataset
    averaged_df = filtered_df.groupby(['evaluator', 'dataset', 'evaluator_rank']).agg({
        'performance': 'mean'
    }).reset_index()
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Define marker shapes for datasets
    dataset_markers = {
        "wikisum": "o",
        "sharegpt": "s",
        "pku_saferlhf": "^",
        "bigcodebench": "D",
    }
    
    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }
    
    # Get family colors for evaluators
    unique_evaluators = averaged_df['evaluator'].unique()
    evaluator_family_colors = {}
    for evaluator in unique_evaluators:
        evaluator_family_colors[evaluator] = get_family_base_color(evaluator)
    
    # Add family color column
    averaged_df['family_color'] = averaged_df['evaluator'].map(evaluator_family_colors)
    
    unique_datasets = sorted(averaged_df['dataset'].unique())
    datasets_with_fit = {}  # Dict: dataset -> correlation
    plotted_datasets = set()
    
    # Calculate x-axis range for extending regression lines (based on evaluator ranks)
    all_ranks = averaged_df['evaluator_rank'].values
    x_min = all_ranks.min()
    x_max = all_ranks.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_min -= padding
    x_max += padding
    
    # Plot points and fit lines per dataset
    for dataset in unique_datasets:
        ds_data = averaged_df[averaged_df['dataset'] == dataset]
        
        if len(ds_data) == 0:
            continue
        
        marker = dataset_markers.get(dataset, "o")
        line_color = dataset_colors.get(dataset, "gray")
        
        # Plot averaged points colored by evaluator family
        for _, row in ds_data.iterrows():
            ax.scatter(
                row['evaluator_rank'],
                row['performance'],
                marker=marker,
                color=row['family_color'],
                s=100,  # Slightly larger since these are averaged points
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=None
            )
        
        plotted_datasets.add((dataset, marker, line_color))
        
        # Fit line per dataset (using dataset color)
        if len(ds_data) >= 2:
            x_vals = ds_data['evaluator_rank'].values
            y_vals = ds_data['performance'].values
            
            # Calculate correlation
            correlation = np.corrcoef(x_vals, y_vals)[0, 1]
            
            # Linear fit
            coeffs = np.polyfit(x_vals, y_vals, 1)
            
            # Use pre-calculated x_min/x_max for extending line
            x_line = np.linspace(x_min, x_max, 100)
            y_line = coeffs[0] * x_line + coeffs[1]
            
            ax.plot(
                x_line,
                y_line,
                linestyle="--",
                linewidth=2,
                alpha=0.8,
                color=line_color
            )
            datasets_with_fit[dataset] = correlation
    
    # Create legend with two parts: model families and datasets
    legend_handles = []
    legend_labels = []
    
    # First, add model family entries (for point colors)
    unique_families = {}
    for evaluator in unique_evaluators:
        family = get_model_provider(evaluator)
        # Remove "Together-" prefix for cleaner legend
        if family.startswith("Together-"):
            family = family.replace("Together-", "")
        if family not in unique_families:
            unique_families[family] = evaluator_family_colors[evaluator]
    
    for family, color in sorted(unique_families.items()):
        h_family = plt.Line2D(
            [0], [0],
            marker='s',
            color='w',
            markerfacecolor=color,
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5,
            linestyle='None'
        )
        legend_handles.append(h_family)
        legend_labels.append(family)
    
    # Add separator (empty entry)
    legend_handles.append(plt.Line2D([0], [0], visible=False))
    legend_labels.append("")  # Empty label for spacing
    
    # Then, add dataset entries (for markers and lines)
    for dataset, marker, line_color in sorted(plotted_datasets, key=lambda x: x[0]):
        h_marker = plt.Line2D(
            [0], [0],
            marker=marker,
            color='w',
            markerfacecolor='gray',  # Neutral color for marker
            markersize=10,
            markeredgecolor='black',
            markeredgewidth=0.5
        )
        
        if dataset in datasets_with_fit:
            correlation = datasets_with_fit[dataset]
            h_line = plt.Line2D(
                [0], [0],
                linestyle="--",
                color=line_color,
                linewidth=2
            )
            legend_handles.append((h_marker, h_line))
            display_name = format_dataset_display_name(dataset)
            legend_labels.append(f"{display_name} (r={correlation:.2f})")
        else:
            legend_handles.append(h_marker)
            display_name = format_dataset_display_name(dataset)
            legend_labels.append(display_name)
    
    # Reference line
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    
    # Labels
    ax.set_xlabel("Evaluator LM Arena Rank\n(Lower rank = Higher capability)", fontsize=12, fontweight="bold")
    if self_scores:
        y_label = "Adjusted Recognition Accuracy\n(Averaged over Generators, Adjusted with Self-Score)"
    else:
        y_label = "Recognition Accuracy (Averaged over Generators)"
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    
    full_title = "Performance vs Evaluator Rank (Averaged)\n(Rank Distance: distance > -20)"
    if self_scores:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)
    
    # Place legend outside the chart area
    ax.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        title="Model Families | Datasets"
    )
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    # Invert x-axis so lower rank numbers (better models) are on the right
    ax.invert_xaxis()
    
    # Adjust layout to make room for legend outside the plot
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved filtered evaluator rank plot (positive, averaged) to: {output_path}")
    plt.close()

def plot_rank_distance_grouped_bar_chart(data_points, output_path, experiment_title="", self_scores=None):
    """
    Create a grouped bar chart showing performance by evaluator (ordered by arena rank)
    with one bar per dataset for each evaluator.
    Only includes model pairs with rank distance between -20 and 20.
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'dataset', 'distance', 'performance'
        output_path: Path to save plot
        experiment_title: Title for plot
        self_scores: Optional dict mapping evaluator -> self-score (for IND experiments)
                     If provided, performance will be adjusted: (cross_performance + self_score) / 2
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    df = pd.DataFrame(data_points)
    
    # Filter to only pairs with rank distance between -20 and 20
    filtered_df = df[(df['distance'] > -20) & (df['distance'] < 20)].copy()
    
    if len(filtered_df) == 0:
        print("⚠ No data points with rank distance between -20 and 20")
        return
    
    # Group by evaluator and dataset, calculate mean performance
    grouped = filtered_df.groupby(['evaluator', 'dataset'])['performance'].mean().reset_index()
    
    # Apply adjustment for IND experiments if self_scores provided
    if self_scores is not None:
        # Adjust performance: (cross_model_performance + self_score) / 2
        def adjust_performance(row):
            evaluator = row['evaluator']
            cross_performance = row['performance']
            if evaluator in self_scores:
                self_score = self_scores[evaluator]
                return (cross_performance + self_score) / 2
            return cross_performance
        
        grouped['performance'] = grouped.apply(adjust_performance, axis=1)
    
    # Get evaluator ranks for ordering
    evaluator_ranks = []
    for evaluator in grouped['evaluator'].unique():
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_ranks.append((evaluator, rank))
    
    if not evaluator_ranks:
        print("⚠ No evaluators with valid LM Arena rankings found")
        return
    
    # Create mapping from evaluator to rank
    evaluator_to_rank = dict(evaluator_ranks)
    
    # Add evaluator rank column
    grouped['evaluator_rank'] = grouped['evaluator'].map(evaluator_to_rank)
    
    # Remove rows where evaluator rank is missing
    grouped = grouped.dropna(subset=['evaluator_rank'])
    
    if len(grouped) == 0:
        print("⚠ No data points after adding evaluator ranks")
        return
    
    # Sort by evaluator rank (ascending - lower rank = better)
    grouped = grouped.sort_values('evaluator_rank')
    
    # Get unique evaluators in sorted order
    unique_evaluators_ordered = grouped['evaluator'].unique()
    
    # Pivot to create grouped bar structure
    pivot_df = grouped.pivot(index='evaluator', columns='dataset', values='performance')
    
    # Ensure the pivot_df index is ordered by arena rank
    # Get ranks for all evaluators in the pivot index
    pivot_evaluator_ranks = {eval: evaluator_to_rank.get(eval) for eval in pivot_df.index}
    # Sort index by rank
    pivot_df = pivot_df.reindex(
        sorted(pivot_df.index, key=lambda x: pivot_evaluator_ranks.get(x, float('inf')))
    )
    
    # Define dataset colors
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(max(16, len(pivot_df) * 0.5), 8))
    
    # Get dataset order (sorted)
    datasets = sorted([col for col in pivot_df.columns if col in dataset_colors])
    n_datasets = len(datasets)
    n_evaluators = len(pivot_df)
    
    # Calculate bar positions
    x = np.arange(n_evaluators)
    width = 0.8 / n_datasets  # Total width of 0.8 divided by number of datasets
    
    # Try to load counts for error bars and significance testing
    df_counts_pivot = None
    if 'n_samples' in df.columns:
        # Aggregate counts same way as performance
        grouped_counts = filtered_df.groupby(['evaluator', 'dataset'])['n_samples'].sum().reset_index()
        df_counts_pivot = grouped_counts.pivot(index='evaluator', columns='dataset', values='n_samples')
        # Reindex to match pivot_df
        df_counts_pivot = df_counts_pivot.reindex(pivot_df.index)
    
    # Calculate error bars and significance if counts available
    error_bars_dict = {}
    p_values_dict = {}
    if df_counts_pivot is not None:
        from scipy.stats import binomtest
        from utils import calculate_binomial_ci, apply_fdr_correction, get_significance_marker
        
        for dataset in datasets:
            if dataset in df_counts_pivot.columns:
                errors = []
                p_vals = []
                for evaluator in pivot_df.index:
                    if evaluator in df_counts_pivot.index:
                        perf = pivot_df.loc[evaluator, dataset]
                        n = df_counts_pivot.loc[evaluator, dataset]
                        if pd.notna(n) and n > 0 and pd.notna(perf):
                            n_int = int(n)  # Ensure n is an integer
                            _, _, se = calculate_binomial_ci(perf, n_int)
                            errors.append(1.96 * se)  # 95% CI half-width
                            # Test against chance (0.5)
                            k = int(perf * n_int)
                            p_val = binomtest(k, n_int, p=0.5, alternative='two-sided').pvalue
                            p_vals.append(p_val)
                        else:
                            errors.append(0)
                            p_vals.append(np.nan)
                    else:
                        errors.append(0)
                        p_vals.append(np.nan)
                error_bars_dict[dataset] = np.array(errors)
                p_values_dict[dataset] = np.array(p_vals)
        
        # Apply FDR correction across all p-values
        if p_values_dict:
            all_p_values = []
            for p_vals in p_values_dict.values():
                all_p_values.extend(p_vals[~np.isnan(p_vals)])
            if len(all_p_values) > 0:
                _, p_corrected_all = apply_fdr_correction(all_p_values)
                # Map back to original structure
                p_idx = 0
                for dataset in datasets:
                    if dataset in p_values_dict:
                        p_vals = p_values_dict[dataset]
                        valid_mask = ~np.isnan(p_vals)
                        p_corrected = np.full_like(p_vals, np.nan)
                        if np.any(valid_mask):
                            n_valid = np.sum(valid_mask)
                            p_corrected[valid_mask] = p_corrected_all[p_idx:p_idx+n_valid]
                            p_idx += n_valid
                        p_values_dict[dataset] = p_corrected
    
    # Plot bars for each dataset with error bars
    for i, dataset in enumerate(datasets):
        offset = (i - n_datasets / 2 + 0.5) * width
        values = pivot_df[dataset].values
        color = dataset_colors.get(dataset, "gray")
        yerr = error_bars_dict.get(dataset) if error_bars_dict else None
        
        ax.bar(
            x + offset,
            values,
            width,
            label=dataset,
            color=color,
            alpha=0.8,
            edgecolor='black',
            linewidth=0.5,
            yerr=yerr,
            error_kw={'capsize': 2, 'capthick': 0.5, 'elinewidth': 0.5, 'alpha': 0.6} if yerr is not None else None
        )
        
        # Add significance markers if available
        if dataset in p_values_dict:
            p_vals = p_values_dict[dataset]
            for j, (evaluator, perf_val) in enumerate(zip(pivot_df.index, values)):
                if j < len(p_vals) and pd.notna(p_vals[j]):
                    # p_vals already contains FDR-corrected values
                    marker = get_significance_marker(p_vals[j])
                    if marker:
                        y_offset = yerr[j] if yerr is not None and j < len(yerr) else 0.02
                        # Place marker above bar if positive, below if negative
                        if perf_val < 0:
                            marker_y = perf_val - y_offset - 0.01
                            va = 'top'
                        else:
                            marker_y = perf_val + y_offset + 0.01
                            va = 'bottom'
                        ax.text(
                            x[j] + offset,
                            marker_y,
                            marker,
                            ha='center',
                            va=va,
                            fontsize=10,
                            fontweight='bold'
                        )
    
    # Set x-axis labels (formatted for display)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [format_evaluator_model_display_name(m) for m in pivot_df.index],
        rotation=45,
        ha="right",
        fontsize=18,
    )
    
    # Labels and title
    ax.set_xlabel("Evaluator Model (ordered by LM Arena Rank)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recognition Accuracy", fontsize=12, fontweight="bold")
    
    full_title = "Performance by Evaluator and Dataset\n(Rank Distance: -20 < distance < 20)"
    if self_scores is not None:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)
    
    # Reference line
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    
    # Legend - add significance marker if any markers were added, then ensure Chance (0.5) is at the end
    handles, labels = ax.get_legend_handles_labels()
    # Remove Chance (0.5) if it exists, we'll add it at the end
    chance_idx = None
    for i, label in enumerate(labels):
        if label == "Chance (0.5)":
            chance_idx = i
            break
    if chance_idx is not None:
        handles.pop(chance_idx)
        labels.pop(chance_idx)
    # Check if any significance markers were added
    has_significance = any('*' in str(text.get_text()) for text in ax.texts)
    if has_significance:
        # Add significance marker to legend
        from matplotlib.patches import Rectangle
        sig_handle = Rectangle((0, 0), 1, 1, fill=False, edgecolor='none', visible=False)
        handles.append(sig_handle)
        labels.append("* p < 0.05")
    # Add chance handle at the very end
    chance_handle = plt.Line2D(
        [0], [0],
        color="#555555",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="Chance (0.5)"
    )
    handles.append(chance_handle)
    labels.append("Chance (0.5)")
    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9,
        title="Datasets"
    )
    
    # Grid
    ax.grid(alpha=0.3, linestyle="--", axis='y')
    ax.set_axisbelow(True)
    ax.tick_params(axis="both", labelsize=18)

    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved grouped bar chart to: {output_path}")
    save_figure_minimal_version(ax, output_path)
    plt.close()

def plot_rank_distance_top5_evaluators(data_points, output_path, experiment_title="", is_adjusted=False):
    """
    Plot performance vs rank distance for top 5 highest-ranked evaluator models only.
    Works with aggregated data (PW) or adjusted data (IND).
    
    Args:
        data_points: List of dicts with keys:
                    'evaluator', 'generator', 'distance', 'performance' (already aggregated/adjusted)
        output_path: Path to save plot
        experiment_title: Title for plot
        is_adjusted: Whether this is adjusted data (IND) or aggregated data (PW)
    """
    if not data_points:
        print("⚠ No data points to plot")
        return

    df = pd.DataFrame(data_points)
    
    # Get all unique evaluators and their rankings
    unique_evaluators = df['evaluator'].unique()
    evaluator_rankings = {}
    
    for evaluator in unique_evaluators:
        rank = get_model_arena_ranking(evaluator)
        if rank is not None:
            evaluator_rankings[evaluator] = rank
    
    if not evaluator_rankings:
        print("⚠ No evaluators with valid LM Arena rankings found")
        return
    
    # Sort by rank (ascending - lower rank number = better/higher ELO)
    # Take top 5 (lowest rank numbers)
    sorted_evaluators = sorted(evaluator_rankings.items(), key=lambda x: x[1])
    top5_evaluators = [eval_name for eval_name, _ in sorted_evaluators[:5]]
    top5_ranks = {eval_name: rank for eval_name, rank in sorted_evaluators[:5]}
    
    print(f"Top 5 evaluators by LM Arena rank: {', '.join(f'{e} (rank {top5_ranks[e]})' for e in top5_evaluators)}")
    
    # Filter data to only include top 5 evaluators
    filtered_df = df[df['evaluator'].isin(top5_evaluators)]
    
    if len(filtered_df) == 0:
        print("⚠ No data points after filtering to top 5 evaluators")
        return
    
    # Set up plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Calculate Global Fit (only on filtered data)
    x_vals = filtered_df['distance'].values
    y_vals = filtered_df['performance'].values
    
    # Calculate x-axis range for extending regression line
    x_min = x_vals.min()
    x_max = x_vals.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)  # 5% padding or at least 1.0
    x_min -= padding
    x_max += padding
    
    # Calculate Correlation
    correlation = np.corrcoef(x_vals, y_vals)[0, 1]
    
    # Linear fit
    coeffs = np.polyfit(x_vals, y_vals, 1)
    
    x_line = np.linspace(x_min, x_max, 100)
    y_line = coeffs[0] * x_line + coeffs[1]
    
    # Plot Points
    ax.scatter(
        x_vals,
        y_vals,
        marker='o',
        color='#1f77b4', # Standard blue
        s=100,
        alpha=0.6,
        edgecolors="black",
        linewidths=0.5,
        label="Top 5 Evaluators"
    )
    
    # Plot Global Fit Line
    ax.plot(
        x_line, 
        y_line, 
        linestyle="-", 
        linewidth=3, 
        alpha=0.9, 
        color="black",
        label=f"Global Fit (r={correlation:.2f})"
    )

    ax.legend(
        loc="upper right",
        fontsize=10,
        framealpha=0.9
    )

    # Reference lines
    ax.axhline(y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8, label="Chance (0.5)")
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3, label="Equal Rank")

    # Labels
    y_label = "Adjusted Recognition Accuracy" if is_adjusted else "Average Recognition Accuracy (across datasets)"
    ax.set_xlabel("Rank Distance (Evaluator Rank - Generator Rank)\nPositive = Evaluator is worse ranked", fontsize=12, fontweight="bold")
    ax.set_ylabel(y_label, fontsize=12, fontweight="bold")
    
    full_title = "Top 5 Evaluators: Performance vs Rank Distance"
    if is_adjusted:
        full_title += "\n(Adjusted for Self-Recognition Bias)"
    if experiment_title:
        full_title += f"\n{experiment_title}"
    ax.set_title(full_title, fontsize=14, fontweight="bold", pad=20)

    # Grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    
    # Y-axis limits
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved top 5 evaluators plot to: {output_path}")
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

    parser = argparse.ArgumentParser(description="Plot performance vs rank distance")
    parser.add_argument("--accuracy_files", type=str, nargs="+", required=True, help="List of accuracy_pivot.csv files")
    parser.add_argument("--model_names", type=str, nargs="+", help="Filter models")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--exclude_self", action="store_true", help="Exclude self-comparisons (Evaluator == Generator)")
    
    args = parser.parse_args()

    # Pre-process model names if set
    model_filter = None
    if args.model_names:
        processed_args = [arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names]
        model_filter = expand_model_names(processed_args)
        print(f"Filtering for {len(model_filter)} models")

    data_points = []
    self_comparison_points = []  # Collect self-scores separately for adjustment
    
    print(f"{'='*70}")
    print("RANK DISTANCE ANALYSIS")
    print(f"{'='*70}")

    for file_path in args.accuracy_files:
        path = Path(file_path)
        if not path.exists():
            print(f"⚠ File not found: {path}")
            continue
            
        dataset_name = extract_dataset_name(str(path))
        print(f"Processing {dataset_name} from {path.name}...")
        
        try:
            # Read pivot table: Index=Evaluator, Columns=Generator/Alternative
            df = pd.read_csv(path, index_col=0)
            
            # Try to load corresponding accuracy_counts.csv file
            counts_path = path.parent / "accuracy_counts.csv"
            df_counts = None
            if counts_path.exists():
                df_counts = pd.read_csv(counts_path, index_col=0)
                print(f"  ✓ Loaded sample counts from {counts_path.name}")
            else:
                print(f"  ⚠ Sample counts not found: {counts_path.name} (error bars will be omitted)")
            
            # Melt to long format
            # Reset index to make Evaluator a column
            df_reset = df.reset_index().rename(columns={'index': 'Evaluator', df.index.name: 'Evaluator'})
            
            # Melt
            melted = df_reset.melt(id_vars=['Evaluator'], var_name='Generator', value_name='Performance')
            
            for _, row in melted.iterrows():
                evaluator = row['Evaluator']
                generator = row['Generator']
                performance = row['Performance']
                
                # Filter by model list if provided
                if model_filter and evaluator not in model_filter:
                    continue
                
                if pd.isna(performance):
                    continue

                # Get sample count if available
                n_samples = None
                if df_counts is not None and evaluator in df_counts.index and generator in df_counts.columns:
                    n_samples = df_counts.loc[evaluator, generator]
                    if pd.isna(n_samples) or n_samples == 0:
                        n_samples = None

                # Collect self-comparisons separately (needed for adjustment)
                if evaluator == generator:
                    self_comparison_points.append({
                        'evaluator': evaluator,
                        'generator': generator,
                        'dataset': dataset_name,
                        'performance': performance,
                        'n_samples': n_samples
                    })
                    # Skip self-comparisons in main data if requested
                    if args.exclude_self:
                        continue

                eval_rank = get_model_arena_ranking(evaluator)
                gen_rank = get_model_arena_ranking(generator)
                
                if eval_rank is None or gen_rank is None:
                    continue
                    
                distance = eval_rank - gen_rank
                
                data_points.append({
                    'evaluator': evaluator,
                    'generator': generator,
                    'dataset': dataset_name,
                    'distance': distance,
                    'performance': performance,
                    'n_samples': n_samples
                })
                
        except Exception as e:
            print(f"Error processing {path}: {e}")

    print(f"Collected {len(data_points)} valid data points.")
    if args.exclude_self:
        print(f"Collected {len(self_comparison_points)} self-comparison points for adjustment.")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw data for inspection
    if data_points:
        pd.DataFrame(data_points).to_csv(output_dir / "rank_distance_data.csv", index=False)
        
    # Get experiment name from output dir path for title
    exp_name = output_dir.parent.parent.name  # Assuming data/analysis/_aggregated_data/{EXP}/{TIMESTAMP}
    if not exp_name.startswith("ICML"):
        # Try finding ICML string in path
        for part in output_dir.parts:
            if part.startswith("ICML"):
                exp_name = part
                break

    # Detect IND experiments by name (e.g. ICML_02_UT_IND-Q_..., ICML_08_UT_IND-Q_Rec_NPr_FA_Rsn-Inst)
    # If IND and --exclude_self was not passed, remove self-comparisons from data_points now
    is_ind_experiment = "_IND" in exp_name
    if is_ind_experiment and self_comparison_points and not args.exclude_self:
        data_points = [p for p in data_points if p["evaluator"] != p["generator"]]
        print(f"Auto-detected IND experiment from name; excluded self-comparisons ({len(data_points)} cross-model points).")
    use_ind = (args.exclude_self or is_ind_experiment) and bool(self_comparison_points)

    # Calculate self_scores for IND experiments (needed for all plots)
    self_scores = None
    self_score_n_samples = None  # Also track n_samples for error propagation
    if use_ind:
        # IND experiment: calculate adjusted data
        self_df = pd.DataFrame(self_comparison_points)
        # Weighted average if n_samples available
        if 'n_samples' in self_df.columns:
            self_df['weighted_perf'] = self_df['performance'] * self_df['n_samples'].fillna(0)
            self_agg = self_df.groupby('evaluator').agg({
                'weighted_perf': 'sum',
                'n_samples': 'sum'
            })
            self_scores = (self_agg['weighted_perf'] / self_agg['n_samples'].replace(0, np.nan)).to_dict()
            self_score_n_samples = self_agg['n_samples'].to_dict()
        else:
            self_scores = self_df.groupby('evaluator')['performance'].mean().to_dict()
            self_score_n_samples = None
        print(f"Calculated self-scores for {len(self_scores)} evaluators.")

    plot_rank_distance(data_points, output_dir / "rank_distance.png", experiment_title=exp_name, self_scores=self_scores)
    
    # Plot filtered by evaluator rank (rank distance between -20 and 20)
    plot_rank_distance_filtered_by_evaluator_rank(
        data_points,
        output_dir / "rank_distance_filtered_evaluator_rank.png",
        experiment_title=exp_name,
        self_scores=self_scores
    )
    
    # Plot filtered by evaluator rank (rank distance > -20, includes all positive distances)
    plot_rank_distance_filtered_by_evaluator_rank_positive(
        data_points,
        output_dir / "rank_distance_filtered_evaluator_rank_positive.png",
        experiment_title=exp_name,
        self_scores=self_scores
    )
    
    # Plot filtered by evaluator rank (rank distance > -20, averaged over generators per dataset)
    plot_rank_distance_filtered_by_evaluator_rank_positive_averaged(
        data_points,
        output_dir / "rank_distance_filtered_evaluator_rank_positive_averaged.png",
        experiment_title=exp_name,
        self_scores=self_scores
    )
    
    # Plot evaluator rank vs generator rank (colored by performance, separate plots per dataset)
    plot_rank_distance_vs_evaluator_rank(
        data_points,
        output_dir,
        experiment_title=exp_name,
        self_scores=self_scores
    )
    
    # Calculate aggregated cross-model points (for both PW and IND)
    cross_model_df = pd.DataFrame(data_points)
    # Aggregate performance (weighted average if n_samples available) and sum n_samples
    if 'n_samples' in cross_model_df.columns:
        cross_model_df['weighted_perf'] = cross_model_df['performance'] * cross_model_df['n_samples'].fillna(0)
        aggregated_cross = cross_model_df.groupby(['evaluator', 'generator', 'distance']).agg({
            'weighted_perf': 'sum',
            'n_samples': 'sum'
        }).reset_index()
        aggregated_cross['performance'] = aggregated_cross['weighted_perf'] / aggregated_cross['n_samples'].replace(0, np.nan)
        aggregated_cross = aggregated_cross.drop(columns=['weighted_perf'])
    else:
        aggregated_cross = cross_model_df.groupby(['evaluator', 'generator', 'distance'])['performance'].mean().reset_index()
        aggregated_cross['n_samples'] = None
    aggregated_cross_list = aggregated_cross.to_dict('records')
    
    # Plot grouped bar chart by evaluator and dataset (rank distance between -20 and 20)
    plot_rank_distance_grouped_bar_chart(
        data_points,
        output_dir / "rank_distance_grouped_bar_chart.png",
        experiment_title=exp_name,
        self_scores=self_scores
    )
    
    # Plot aggregated (for PW) or adjusted (for IND)
    if use_ind:
        
        # Still plot aggregated for reference
        plot_rank_distance_aggregated(data_points, output_dir / "rank_distance_aggregated.png", experiment_title=exp_name)
        
        plot_rank_distance_adjusted(
            aggregated_cross_list,
            self_scores,
            output_dir / "rank_distance_adjusted.png",
            experiment_title=exp_name,
            self_score_n_samples=self_score_n_samples
        )
        
        # Create adjusted data for top 5 plot
        adjusted_points = []
        for point in aggregated_cross_list:
            evaluator = point['evaluator']
            if evaluator in self_scores:
                self_score = self_scores[evaluator]
                cross_performance = point['performance']
                adjusted_performance = (cross_performance + self_score) / 2
                adjusted_points.append({
                    'evaluator': evaluator,
                    'generator': point['generator'],
                    'distance': point['distance'],
                    'performance': adjusted_performance
                })
        
        plot_rank_distance_top5_evaluators(
            adjusted_points,
            output_dir / "rank_distance_top5_evaluators.png",
            experiment_title=exp_name,
            is_adjusted=True
        )
    else:
        # PW experiment: use aggregated data
        plot_rank_distance_aggregated(data_points, output_dir / "rank_distance_aggregated.png", experiment_title=exp_name)
        
        plot_rank_distance_top5_evaluators(
            aggregated_cross_list,
            output_dir / "rank_distance_top5_evaluators.png",
            experiment_title=exp_name,
            is_adjusted=False
        )
    
    print(f"{'='*70}\n")

if __name__ == "__main__":
    main()
