#!/usr/bin/env python3
"""
Shared utilities for pairwise analysis scripts.

This module contains common functions used across analysis scripts including:
- Model ordering and provider identification
- Heatmap styling and provider boundaries
- Color schemes for model families
- Model set expansion for command-line arguments
"""

import pandas as pd
from src.helpers.model_sets import get_model_set


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
    elif model_lower.startswith("grok-"):
        return "XAI"
    elif model_lower.startswith("kimi-"):
        return "Moonshot"
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
        "openai-oss": {
            "base": "#10a37f",  # OpenAI green
            "models": ["gpt-oss-20b-thinking", "gpt-oss-120b-thinking"],
            "shades": ["#7dd3b0", "#10a37f"],
        },
        "anthropic": {
            "base": "#ea580c",  # Claude red-orange
            "models": [
                "haiku-3.5",
                "sonnet-3.7",
                "sonnet-4.5",
                "opus-4.1",
                "sonnet-3.7-thinking",
            ],
            "shades": [
                "#fb923c",
                "#f97316",
                "#ea580c",
                "#c2410c",
                "#f97316",
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
            "models": [
                "ll-3.1-8b",
                "ll-3.1-70b",
                "ll-3.1-405b",
                "ll-3.3-70b-dsR1-thinking",
            ],
            "shades": [
                "#93c5fd",
                "#60a5fa",
                "#3b82f6",
                "#2563eb",
            ],  # Light to dark blue
        },
        "qwen": {
            "base": "#7c3aed",  # Purple
            "models": [
                "qwen-2.5-7b",
                "qwen-2.5-72b",
                "qwen-3.0-80b",
                "qwen-3.0-80b-thinking",
                "qwen-3.0-235b-thinking",
            ],
            "shades": [
                "#c4b5fd",
                "#a78bfa",
                "#7c3aed",
                "#7c3aed",
                "#6d28d9",
            ],  # Light to dark purple
        },
        "deepseek": {
            "base": "#dc2626",  # Red
            "models": ["deepseek-3.0", "deepseek-3.1", "deepseek-r1-thinking"],
            "shades": ["#fca5a5", "#dc2626", "#b91c1c"],  # Light to dark red
        },
        "xai": {
            "base": "#1d4ed8",  # XAI blue
            "models": ["grok-3-mini-thinking"],
            "shades": ["#3b82f6"],
        },
        "moonshot": {
            "base": "#0891b2",  # Cyan
            "models": ["kimi-k2-thinking"],
            "shades": ["#06b6d4"],
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


def detect_model_order(pivot: pd.DataFrame) -> list[str]:
    """
    Auto-detect the appropriate model order based on the models in the pivot table.

    Args:
        pivot: Pivot table with models as index

    Returns:
        Appropriate model order list (CoT or regular)
    """
    model_order_cot = get_model_set("gen_cot")
    model_order_regular = get_model_set("dr")

    # Check which order has more matches
    cot_matches = len([m for m in model_order_cot if m in pivot.index])
    regular_matches = len([m for m in model_order_regular if m in pivot.index])

    return model_order_cot if cot_matches > regular_matches else model_order_regular


def reorder_pivot(pivot: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
    """
    Reorder a pivot table according to canonical model order.

    Args:
        pivot: Pivot table with models as index and columns
        strict: If True, only include models in canonical order.
                If False, append unrecognized models at the end.

    Returns:
        Reordered pivot table
    """
    model_order = detect_model_order(pivot)

    # Filter to only models in the canonical order
    row_order = [m for m in model_order if m in pivot.index]
    col_order = [m for m in model_order if m in pivot.columns]

    if not strict:
        # Add any models not in canonical order at the end
        for m in pivot.index:
            if m not in row_order:
                row_order.append(m)
        for m in pivot.columns:
            if m not in col_order:
                col_order.append(m)

    # Reindex to apply ordering
    return pivot.reindex(index=row_order, columns=col_order)


def expand_model_names(model_names: list[str]) -> list[str]:
    """
    Expand model set references (e.g., '-set gen_cot') to actual model names.

    Supports patterns like:
    - Individual model names: 'haiku-3.5', 'gpt-4.1'
    - Set references: '-set gen_cot', '-set dr', '-set eval_cot'

    Args:
        model_names: List of model names and/or set references

    Returns:
        Expanded list of model names (sets replaced with actual model names)
    """
    expanded = []
    i = 0

    while i < len(model_names):
        if model_names[i] == "-set" and i + 1 < len(model_names):
            # Found a set reference
            set_name = model_names[i + 1]

            try:
                models_in_set = get_model_set(set_name)
                if models_in_set and len(models_in_set) > 0:
                    expanded.extend(models_in_set)
                    print(
                        f"  Expanded '-set {set_name}' -> {len(models_in_set)} models: {', '.join(models_in_set)}"
                    )
                else:
                    raise ValueError(f"Unknown or empty model set: {set_name}")
            except (AttributeError, KeyError, TypeError, ValueError) as e:
                raise ValueError(f"Unknown model set: {set_name}") from e

            i += 2  # Skip both '-set' and the set name
        else:
            # Regular model name
            expanded.append(model_names[i])
            i += 1

    return expanded


def parse_models_from_config(models_str: str | None) -> list[str] | None:
    """
    Parse models specification from config file.

    Supports two formats:
    1. Set reference: "-set gen_cot" or "-set dr"
    2. Space-separated list: "gpt-4o gpt-4.1-mini haiku-3.5"

    Args:
        models_str: String from config file, or None

    Returns:
        List of model names (expanded if set reference), or None if models_str is None/empty
    """
    if not models_str or not models_str.strip():
        return None

    models_str = models_str.strip()

    # Check if it's a set reference
    if models_str.startswith("-set "):
        set_name = models_str[5:].strip()  # Remove "-set " prefix
        models = get_model_set(set_name)
        if not models:
            raise ValueError(f"Unknown or empty model set: {set_name}")
        return models

    # Otherwise, treat as space-separated list of model names
    return models_str.split()
