#!/usr/bin/env python3
"""
Compare model performance vs model size (parameter count), release date, and capability tier.

This script creates scatter plots showing:
- X-axis: Model size (parameters in billions) OR Release date OR Capability tier
- Y-axis: Performance (recognition accuracy)
- Different marker shapes for different datasets
- Color coding by model family

Creates five plots:
1. Performance vs Size (known sizes only)
2. Performance vs Size (known + estimated sizes)
3. Performance vs Release Date (known dates only)
4. Performance vs Release Date (known + estimated dates)
5. Performance vs Capability Tier

Usage:
    uv run experiments/_scripts/analysis/performance_vs_size.py \
        --aggregated_file data/analysis/_aggregated_data/.../aggregated_performance.csv \
        --model_names -set dr

Output:
    - Same directory as input file:
        - performance_vs_size_known.png: Scatter plot (known sizes only)
        - performance_vs_size_all.png: Scatter plot (known + estimated)
        - performance_vs_date_known.png: Scatter plot (known dates only)
        - performance_vs_date_all.png: Scatter plot (known + estimated dates)
        - performance_vs_tier.png: Scatter plot (capability tiers)
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
from utils import expand_model_names, get_model_provider
from src.helpers.model_names import (
    MODEL_PARAMETER_COUNTS,
    MODEL_PARAMETER_COUNTS_ESTIMATED,
    MODEL_RELEASE_DATES,
    MODEL_RELEASE_DATES_ESTIMATED,
    MODEL_CAPABILITY_TIERS,
)


def parse_size_string(size_str: str) -> float | None:
    """
    Parse a size string (e.g., "8B", "1.8T", "unknown") to a numeric value in billions.

    Args:
        size_str: Size string (e.g., "8B", "1.8T", "unknown")

    Returns:
        Size in billions (float), or None if unknown
    """
    if size_str == "unknown" or not size_str:
        return None

    size_str = size_str.strip().upper()

    # Handle trillions (T)
    if size_str.endswith("T"):
        value = float(size_str[:-1])
        return value * 1000  # Convert trillions to billions

    # Handle billions (B)
    if size_str.endswith("B"):
        return float(size_str[:-1])

    # Try to parse as number (assume billions)
    try:
        return float(size_str)
    except ValueError:
        return None


def get_model_size(model_name: str, use_estimated: bool = False) -> float | None:
    """
    Get model size in billions from MODEL_PARAMETER_COUNTS.

    Args:
        model_name: Short model name (e.g., "gpt-4o", "haiku-3.5", "ll-3.1-8b_fw")
        use_estimated: If True, also check MODEL_PARAMETER_COUNTS_ESTIMATED

    Returns:
        Size in billions (float), or None if unknown
    """
    # Remove -thinking suffix if present
    base_name = model_name.replace("-thinking", "")

    # Check known sizes first (try exact match, then without _fw suffix)
    names_to_try = [base_name]
    if base_name.endswith("_fw"):
        names_to_try.append(base_name[:-3])  # Remove "_fw" suffix

    for name in names_to_try:
        if name in MODEL_PARAMETER_COUNTS:
            size_str = MODEL_PARAMETER_COUNTS[name]
            if size_str != "unknown":
                return parse_size_string(size_str)

    # Check estimated sizes if requested
    if use_estimated:
        for name in names_to_try:
            if name in MODEL_PARAMETER_COUNTS_ESTIMATED:
                size_str = MODEL_PARAMETER_COUNTS_ESTIMATED[name]
                if size_str != "unknown":
                    return parse_size_string(size_str)

    return None


def parse_date_string(date_str: str) -> datetime | None:
    """
    Parse a date string (e.g., "2024-07-18", "2024-05", "unknown") to a datetime object.

    Args:
        date_str: Date string (e.g., "2024-07-18", "2024-05", "unknown")

    Returns:
        datetime object, or None if unknown/invalid
    """
    if date_str == "unknown" or not date_str:
        return None

    date_str = date_str.strip()

    # Try YYYY-MM-DD format
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        pass

    # Try YYYY-MM format (use first day of month)
    try:
        return datetime.strptime(date_str, "%Y-%m")
    except ValueError:
        pass

    return None


def get_model_release_date(
    model_name: str, use_estimated: bool = False
) -> datetime | None:
    """
    Get model release date from MODEL_RELEASE_DATES.

    Args:
        model_name: Short model name (e.g., "gpt-4o", "haiku-3.5", "ll-3.1-8b_fw")
        use_estimated: If True, also check MODEL_RELEASE_DATES_ESTIMATED

    Returns:
        datetime object, or None if unknown
    """
    # Remove -thinking suffix if present
    base_name = model_name.replace("-thinking", "")

    # Check known dates first (try exact match, then without _fw suffix)
    names_to_try = [base_name]
    if base_name.endswith("_fw"):
        names_to_try.append(base_name[:-3])  # Remove "_fw" suffix

    for name in names_to_try:
        if name in MODEL_RELEASE_DATES:
            date_str = MODEL_RELEASE_DATES[name]
            if date_str != "unknown":
                return parse_date_string(date_str)

    # Check estimated dates if requested
    if use_estimated:
        for name in names_to_try:
            if name in MODEL_RELEASE_DATES_ESTIMATED:
                date_str = MODEL_RELEASE_DATES_ESTIMATED[name]
                if date_str != "unknown":
                    return parse_date_string(date_str)

    return None


def get_model_capability_tier(model_name: str) -> int | None:
    """
    Get model capability tier from MODEL_CAPABILITY_TIERS.

    Args:
        model_name: Short model name (e.g., "gpt-4o", "haiku-3.5", "ll-3.1-8b_fw")

    Returns:
        Capability tier (1-5), or None if unknown
    """
    # Remove -thinking suffix if present
    base_name = model_name.replace("-thinking", "")

    # Check capability tiers (try exact match, then without _fw suffix)
    names_to_try = [base_name]
    if base_name.endswith("_fw"):
        names_to_try.append(base_name[:-3])  # Remove "_fw" suffix

    for name in names_to_try:
        if name in MODEL_CAPABILITY_TIERS:
            return MODEL_CAPABILITY_TIERS[name]

    return None


def extract_dataset_name(full_path: str) -> str:
    """
    Extract short dataset name from full path.

    Examples:
        "wikisum/training_set_1-20+test_set_1-30" -> "wikisum"
        "sharegpt/english_26+english2_74" -> "sharegpt"
        "bigcodebench/instruct_1-50" -> "bigcodebench"
    """
    return full_path.split("/")[0]


def plot_performance_vs_size(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    experiment_title: str = "",
    use_estimated: bool = False,
):
    """
    Create a scatter plot of performance vs model size.

    Args:
        df: DataFrame with models as rows, datasets as columns, performance values
        output_path: Path to save the plot
        title: Chart title
        experiment_title: Optional experiment name for title
        use_estimated: Whether to include estimated sizes
    """
    print(f"Generating scatter plot: {title}...")

    # Get model sizes
    model_sizes = {}
    for model in df.index:
        size = get_model_size(model, use_estimated=use_estimated)
        if size is not None:
            model_sizes[model] = size

    if not model_sizes:
        print("  ⚠ No models with known sizes found")
        return

    # Filter to models with known sizes
    available_models = [m for m in df.index if m in model_sizes]
    if not available_models:
        print("  ⚠ No models with known sizes in the data")
        return

    df_filtered = df.loc[available_models]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define marker shapes for datasets (different shapes)
    dataset_markers = {
        "wikisum": "o",  # circle
        "sharegpt": "s",  # square
        "pku_saferlhf": "^",  # triangle
        "bigcodebench": "D",  # diamond
    }

    # Define colors for model families
    family_colors = {
        "OpenAI": "#10a37f",  # OpenAI green
        "Anthropic": "#ea580c",  # Claude red-orange
        "Google": "#fbbf24",  # Google yellow
        "Together-Llama": "#3b82f6",  # Blue
        "Together-Qwen": "#8b5cf6",  # Purple
        "Together-DeepSeek": "#06b6d4",  # Cyan
        "XAI": "#000000",  # Black
        "Moonshot": "#ec4899",  # Pink
        "Unknown": "#808080",  # Gray
    }

    # Get unique datasets and assign markers
    datasets = df_filtered.columns.tolist()
    dataset_to_marker = {}
    marker_idx = 0
    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        if short_name in dataset_markers:
            dataset_to_marker[dataset] = dataset_markers[short_name]
        else:
            # Assign a marker from the list
            dataset_to_marker[dataset] = marker_list[marker_idx % len(marker_list)]
            marker_idx += 1

    # Collect data points per dataset for fit lines
    dataset_data = {}  # dataset -> list of (x, y) tuples

    # Plot each model-dataset combination
    plotted_families = set()
    plotted_datasets = set()

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        marker = dataset_to_marker[dataset]
        dataset_data[dataset] = []

        for model in available_models:
            size = model_sizes[model]
            performance = df_filtered.loc[model, dataset]

            if pd.isna(performance):
                continue

            # Get model family and color
            family = get_model_provider(model)
            color = family_colors.get(family, family_colors["Unknown"])

            # Plot point
            ax.scatter(
                size,
                performance,
                marker=marker,
                color=color,
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=None,
            )  # We'll create custom legend

            # Collect data for fit line
            dataset_data[dataset].append((size, performance))

            # Track for legend
            plotted_families.add(family)
            plotted_datasets.add((dataset, marker))

    # Add fit lines for each dataset
    for dataset in datasets:
        if len(dataset_data[dataset]) < 2:
            continue  # Need at least 2 points for a line

        short_name = extract_dataset_name(dataset)
        x_vals = [x for x, y in dataset_data[dataset]]
        y_vals = [y for x, y in dataset_data[dataset]]

        # Fit line in log space (since x-axis is log scale)
        # Use log of x values for fitting
        log_x_vals = np.log10(x_vals)

        # Fit linear regression: y = a * log10(x) + b
        coeffs = np.polyfit(log_x_vals, y_vals, 1)

        # Generate points for the fit line
        x_min, x_max = min(x_vals), max(x_vals)
        x_line = np.logspace(np.log10(x_min), np.log10(x_max), 100)
        y_line = coeffs[0] * np.log10(x_line) + coeffs[1]

        # Get dataset color for fit line
        # line_color = dataset_colors.get(short_name, 'gray')

        # Plot fit line with dataset color
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            # color=line_color,
            label=short_name,
        )  # Don't add to legend

    # Create custom legend
    # Family colors (one entry per family)
    family_handles = []
    for family in sorted(plotted_families):
        color = family_colors.get(family, family_colors["Unknown"])
        family_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=family,
            )
        )

    # Dataset markers (one entry per dataset)
    dataset_handles = []
    for dataset, marker in sorted(plotted_datasets, key=lambda x: x[0]):
        short_name = extract_dataset_name(dataset)
        dataset_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=short_name,
            )
        )

    # Combine legends
    all_handles = family_handles + dataset_handles
    ax.legend(handles=all_handles, loc="upper left", fontsize=9, framealpha=0.9, ncol=2)

    # Add reference line at chance (0.5)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Chance (0.5)",
    )

    # Labels and title
    ax.set_xlabel("Model Size (Parameters in Billions)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recognition Accuracy", fontsize=12, fontweight="bold")

    # Use log scale for x-axis (model sizes vary widely)
    ax.set_xscale("log")

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    full_title = title
    if experiment_title:
        full_title = f"{title}\n{experiment_title}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=20)

    # Add grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved scatter plot to: {output_path}")
    plt.close()


def plot_performance_vs_date(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    experiment_title: str = "",
    use_estimated: bool = False,
):
    """
    Create a scatter plot of performance vs release date.

    Args:
        df: DataFrame with models as rows, datasets as columns, performance values
        output_path: Path to save the plot
        title: Chart title
        experiment_title: Optional experiment name for title
        use_estimated: Whether to include estimated dates
    """
    print(f"Generating scatter plot: {title}...")

    # Get model release dates
    model_dates = {}
    for model in df.index:
        date = get_model_release_date(model, use_estimated=use_estimated)
        if date is not None:
            model_dates[model] = date

    if not model_dates:
        print("  ⚠ No models with known release dates found")
        return

    # Filter to models with known dates
    available_models = [m for m in df.index if m in model_dates]
    if not available_models:
        print("  ⚠ No models with known release dates in the data")
        return

    df_filtered = df.loc[available_models]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define marker shapes for datasets (different shapes)
    dataset_markers = {
        "wikisum": "o",  # circle
        "sharegpt": "s",  # square
        "pku_saferlhf": "^",  # triangle
        "bigcodebench": "D",  # diamond
    }

    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }

    # Define colors for model families
    family_colors = {
        "OpenAI": "#10a37f",  # OpenAI green
        "Anthropic": "#ea580c",  # Claude red-orange
        "Google": "#fbbf24",  # Google yellow
        "Together-Llama": "#3b82f6",  # Blue
        "Together-Qwen": "#8b5cf6",  # Purple
        "Together-DeepSeek": "#06b6d4",  # Cyan
        "XAI": "#000000",  # Black
        "Moonshot": "#ec4899",  # Pink
        "Unknown": "#808080",  # Gray
    }

    # Get unique datasets and assign markers
    datasets = df_filtered.columns.tolist()
    dataset_to_marker = {}
    marker_idx = 0
    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        if short_name in dataset_markers:
            dataset_to_marker[dataset] = dataset_markers[short_name]
        else:
            # Assign a marker from the list
            dataset_to_marker[dataset] = marker_list[marker_idx % len(marker_list)]
            marker_idx += 1

    # Collect data points per dataset for fit lines
    dataset_data = {}  # dataset -> list of (x, y) tuples

    # Plot each model-dataset combination
    plotted_families = set()
    plotted_datasets = set()

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        marker = dataset_to_marker[dataset]
        dataset_data[dataset] = []

        for model in available_models:
            date = model_dates[model]
            performance = df_filtered.loc[model, dataset]

            if pd.isna(performance):
                continue

            # Get model family and color
            family = get_model_provider(model)
            color = family_colors.get(family, family_colors["Unknown"])

            # Plot point
            ax.scatter(
                date,
                performance,
                marker=marker,
                color=color,
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=None,
            )  # We'll create custom legend

            # Collect data for fit line (convert date to numeric for fitting)
            date_num = mdates.date2num(date)
            dataset_data[dataset].append((date_num, performance))

            # Track for legend
            plotted_families.add(family)
            plotted_datasets.add((dataset, marker))

    # Add fit lines for each dataset
    for dataset in datasets:
        if len(dataset_data[dataset]) < 2:
            continue  # Need at least 2 points for a line

        short_name = extract_dataset_name(dataset)
        x_vals = [x for x, y in dataset_data[dataset]]
        y_vals = [y for x, y in dataset_data[dataset]]

        # Fit linear regression
        coeffs = np.polyfit(x_vals, y_vals, 1)

        # Generate points for the fit line
        x_min, x_max = min(x_vals), max(x_vals)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = coeffs[0] * x_line + coeffs[1]

        # Convert back to dates for plotting
        x_line_dates = mdates.num2date(x_line)

        # Get dataset color for fit line
        line_color = dataset_colors.get(short_name, "gray")

        # Plot fit line with dataset color
        ax.plot(
            x_line_dates,
            y_line,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            color=line_color,
            label=None,
        )  # Don't add to legend

    # Create custom legend
    # Family colors (one entry per family)
    family_handles = []
    for family in sorted(plotted_families):
        color = family_colors.get(family, family_colors["Unknown"])
        family_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=family,
            )
        )

    # Dataset markers (one entry per dataset)
    dataset_handles = []
    for dataset, marker in sorted(plotted_datasets, key=lambda x: x[0]):
        short_name = extract_dataset_name(dataset)
        dataset_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=short_name,
            )
        )

    # Combine legends
    all_handles = family_handles + dataset_handles
    ax.legend(handles=all_handles, loc="upper left", fontsize=9, framealpha=0.9, ncol=2)

    # Add reference line at chance (0.5)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Chance (0.5)",
    )

    # Labels and title
    ax.set_xlabel("Release Date", fontsize=12, fontweight="bold")
    ax.set_ylabel("Recognition Accuracy", fontsize=12, fontweight="bold")

    # Format x-axis as dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    full_title = title
    if experiment_title:
        full_title = f"{title}\n{experiment_title}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=20)

    # Add grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved scatter plot to: {output_path}")
    plt.close()


def plot_performance_vs_tier(
    df: pd.DataFrame,
    output_path: Path,
    title: str,
    experiment_title: str = "",
):
    """
    Create a scatter plot of performance vs capability tier.

    Args:
        df: DataFrame with models as rows, datasets as columns, performance values
        output_path: Path to save the plot
        title: Chart title
        experiment_title: Optional experiment name for title
    """
    print(f"Generating scatter plot: {title}...")

    # Get model capability tiers
    model_tiers = {}
    for model in df.index:
        tier = get_model_capability_tier(model)
        if tier is not None:
            model_tiers[model] = tier

    if not model_tiers:
        print("  ⚠ No models with known capability tiers found")
        return

    # Filter to models with known tiers
    available_models = [m for m in df.index if m in model_tiers]
    if not available_models:
        print("  ⚠ No models with known capability tiers in the data")
        return

    df_filtered = df.loc[available_models]

    # Set up the plot
    fig, ax = plt.subplots(figsize=(14, 10))

    # Define marker shapes for datasets (different shapes)
    dataset_markers = {
        "wikisum": "o",  # circle
        "sharegpt": "s",  # square
        "pku_saferlhf": "^",  # triangle
        "bigcodebench": "D",  # diamond
    }

    # Define colors for datasets (for fit lines)
    dataset_colors = {
        "wikisum": "#1f77b4",  # blue
        "sharegpt": "#ff7f0e",  # orange
        "pku_saferlhf": "#2ca02c",  # green
        "bigcodebench": "#d62728",  # red
    }

    # Define colors for model families
    family_colors = {
        "OpenAI": "#10a37f",  # OpenAI green
        "Anthropic": "#ea580c",  # Claude red-orange
        "Google": "#fbbf24",  # Google yellow
        "Together-Llama": "#3b82f6",  # Blue
        "Together-Qwen": "#8b5cf6",  # Purple
        "Together-DeepSeek": "#06b6d4",  # Cyan
        "XAI": "#000000",  # Black
        "Moonshot": "#ec4899",  # Pink
        "Unknown": "#808080",  # Gray
    }

    # Get unique datasets and assign markers
    datasets = df_filtered.columns.tolist()
    dataset_to_marker = {}
    marker_idx = 0
    marker_list = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "H", "X"]

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        if short_name in dataset_markers:
            dataset_to_marker[dataset] = dataset_markers[short_name]
        else:
            # Assign a marker from the list
            dataset_to_marker[dataset] = marker_list[marker_idx % len(marker_list)]
            marker_idx += 1

    # Collect data points per dataset for fit lines
    dataset_data = {}  # dataset -> list of (x, y) tuples

    # Plot each model-dataset combination
    plotted_families = set()
    plotted_datasets = set()

    for dataset in datasets:
        short_name = extract_dataset_name(dataset)
        marker = dataset_to_marker[dataset]
        dataset_data[dataset] = []

        for model in available_models:
            tier = model_tiers[model]
            performance = df_filtered.loc[model, dataset]

            if pd.isna(performance):
                continue

            # Get model family and color
            family = get_model_provider(model)
            color = family_colors.get(family, family_colors["Unknown"])

            # Plot point
            ax.scatter(
                tier,
                performance,
                marker=marker,
                color=color,
                s=100,
                alpha=0.7,
                edgecolors="black",
                linewidths=0.5,
                label=None,
            )  # We'll create custom legend

            # Collect data for fit line
            dataset_data[dataset].append((tier, performance))

            # Track for legend
            plotted_families.add(family)
            plotted_datasets.add((dataset, marker))

    # Add fit lines for each dataset
    for dataset in datasets:
        if len(dataset_data[dataset]) < 2:
            continue  # Need at least 2 points for a line

        short_name = extract_dataset_name(dataset)
        x_vals = [x for x, y in dataset_data[dataset]]
        y_vals = [y for x, y in dataset_data[dataset]]

        # Fit linear regression
        coeffs = np.polyfit(x_vals, y_vals, 1)

        # Generate points for the fit line
        x_min, x_max = min(x_vals), max(x_vals)
        x_line = np.linspace(x_min, x_max, 100)
        y_line = coeffs[0] * x_line + coeffs[1]

        # Get dataset color for fit line
        line_color = dataset_colors.get(short_name, "gray")

        # Plot fit line with dataset color
        ax.plot(
            x_line,
            y_line,
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            color=line_color,
            label=None,
        )  # Don't add to legend

    # Create custom legend
    # Family colors (one entry per family)
    family_handles = []
    for family in sorted(plotted_families):
        color = family_colors.get(family, family_colors["Unknown"])
        family_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=family,
            )
        )

    # Dataset markers (one entry per dataset)
    dataset_handles = []
    for dataset, marker in sorted(plotted_datasets, key=lambda x: x[0]):
        short_name = extract_dataset_name(dataset)
        dataset_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=marker,
                color="w",
                markerfacecolor="gray",
                markersize=10,
                markeredgecolor="black",
                markeredgewidth=0.5,
                label=short_name,
            )
        )

    # Combine legends
    all_handles = family_handles + dataset_handles
    ax.legend(handles=all_handles, loc="upper left", fontsize=9, framealpha=0.9, ncol=2)

    # Add reference line at chance (0.5)
    ax.axhline(
        y=0.5,
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Chance (0.5)",
    )

    # Labels and title
    ax.set_xlabel(
        "Capability Tier (1 = Lowest, 5 = Highest)", fontsize=12, fontweight="bold"
    )
    ax.set_ylabel("Recognition Accuracy", fontsize=12, fontweight="bold")

    # Set x-axis to integer ticks (1-5)
    ax.set_xlim(0.5, 5.5)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1", "2", "3", "4", "5"])

    # Set y-axis limits
    ax.set_ylim(0, 1.05)

    full_title = title
    if experiment_title:
        full_title = f"{title}\n{experiment_title}"
    ax.set_title(full_title, fontsize=13, fontweight="bold", pad=20)

    # Add grid
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved scatter plot to: {output_path}")
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
        description="Compare model performance vs model size, release date, and capability tier",
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
    print("PERFORMANCE VS MODEL SIZE, RELEASE DATE & CAPABILITY TIER")
    print(f"{'='*70}")
    print(f"Input file: {aggregated_file}\n")

    # Load data
    print("Loading aggregated performance data...")
    df = pd.read_csv(aggregated_file, index_col=0)

    # Filter and order models if specified
    if model_order:
        available_models = [m for m in model_order if m in df.index]
        if available_models:
            df = df.reindex(available_models)
        else:
            print("  ⚠ Warning: No models from filter list found in data")

    print(f"  ✓ Loaded data: {df.shape[0]} models × {df.shape[1]} datasets\n")

    if df.empty:
        print("⚠ No data to analyze after filtering!")
        return

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

    # Generate plots
    known_plot_path = output_dir / "performance_vs_size_known.png"
    plot_performance_vs_size(
        df,
        known_plot_path,
        title="Performance vs Model Size (Known Sizes Only)",
        experiment_title=experiment_title,
        use_estimated=False,
    )
    print()

    all_plot_path = output_dir / "performance_vs_size_all.png"
    plot_performance_vs_size(
        df,
        all_plot_path,
        title="Performance vs Model Size (Known + Estimated)",
        experiment_title=experiment_title,
        use_estimated=True,
    )
    print()

    # Generate date plots
    known_date_plot_path = output_dir / "performance_vs_date_known.png"
    plot_performance_vs_date(
        df,
        known_date_plot_path,
        title="Performance vs Release Date (Known Dates Only)",
        experiment_title=experiment_title,
        use_estimated=False,
    )
    print()

    all_date_plot_path = output_dir / "performance_vs_date_all.png"
    plot_performance_vs_date(
        df,
        all_date_plot_path,
        title="Performance vs Release Date (Known + Estimated)",
        experiment_title=experiment_title,
        use_estimated=True,
    )
    print()

    # Generate tier plot
    tier_plot_path = output_dir / "performance_vs_tier.png"
    plot_performance_vs_tier(
        df,
        tier_plot_path,
        title="Performance vs Capability Tier",
        experiment_title=experiment_title,
    )
    print()

    # Display summary
    print(f"{'='*70}")
    print("MODEL SIZE SUMMARY")
    print(f"{'='*70}\n")

    # Count models with known/estimated sizes
    known_size_count = 0
    estimated_size_count = 0
    unknown_size_count = 0

    for model in df.index:
        if get_model_size(model, use_estimated=False) is not None:
            known_size_count += 1
        elif get_model_size(model, use_estimated=True) is not None:
            estimated_size_count += 1
        else:
            unknown_size_count += 1

    print(f"Models with known sizes: {known_size_count}")
    print(f"Models with estimated sizes only: {estimated_size_count}")
    print(f"Models with unknown sizes: {unknown_size_count}")
    print()

    print(f"{'='*70}")
    print("RELEASE DATE SUMMARY")
    print(f"{'='*70}\n")

    # Count models with known/estimated dates
    known_date_count = 0
    estimated_date_count = 0
    unknown_date_count = 0

    for model in df.index:
        if get_model_release_date(model, use_estimated=False) is not None:
            known_date_count += 1
        elif get_model_release_date(model, use_estimated=True) is not None:
            estimated_date_count += 1
        else:
            unknown_date_count += 1

    print(f"Models with known release dates: {known_date_count}")
    print(f"Models with estimated release dates only: {estimated_date_count}")
    print(f"Models with unknown release dates: {unknown_date_count}")
    print()

    print(f"{'='*70}")
    print("CAPABILITY TIER SUMMARY")
    print(f"{'='*70}\n")

    # Count models by tier
    tier_counts = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    unknown_tier_count = 0

    for model in df.index:
        tier = get_model_capability_tier(model)
        if tier is not None:
            tier_counts[tier] += 1
        else:
            unknown_tier_count += 1

    for tier in range(1, 6):
        print(f"Tier {tier}: {tier_counts[tier]} models")
    print(f"Unknown tier: {unknown_tier_count} models")
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • performance_vs_size_known.png: Scatter plot (known sizes only)")
    print("  • performance_vs_size_all.png: Scatter plot (known + estimated sizes)")
    print("  • performance_vs_date_known.png: Scatter plot (known dates only)")
    print("  • performance_vs_date_all.png: Scatter plot (known + estimated dates)")
    print("  • performance_vs_tier.png: Scatter plot (capability tiers)")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
