#!/usr/bin/env python3
"""
Compare two pairwise self-recognition experiments.

Takes two experiment directories, loads their pre-computed accuracy pivot tables,
and generates a difference heatmap showing how accuracy changes between experiments.

Usage:
    uv run experiments/scripts/compare_experiments.py \
        --experiment1 data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \
        --experiment2 data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr

Output:
    - data/analysis/{dataset}/{subset}/comparisons/{exp1_name}_vs_{exp2_name}/
        - accuracy_difference.csv: Difference matrix (exp1 - exp2)
        - accuracy_difference_heatmap.png: Visualization
        - experiment_comparison_stats.txt: Comparison statistics
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
from inspect_ai.log import read_eval_log
import yaml


def get_experiment_name_mapping() -> dict[str, str]:
    """
    Map experiment code abbreviations to full descriptive names.

    Returns:
        Dictionary mapping abbreviated codes to full experiment names
    """
    return {
        # User Tags (UT)
        "UT_PW-Q_Rec_Pr": "User Tags Pairwise Query Primed",
        "UT_PW-Q_Rec_NPr": "User Tags Pairwise Query Unprimed",
        "UT_PW-C_Rec_Pr": "User Tags Pairwise Conversation Primed",
        "UT_PW-C_Rec_NPr": "User Tags Pairwise Conversation Unprimed",
        "UT_IND-Q_Rec_Pr": "User Tags Individual Query Primed",
        "UT_IND-Q_Rec_NPr": "User Tags Individual Query Unprimed",
        "UT_IND-C_Rec_Pr": "User Tags Individual Conversation Primed",
        "UT_IND-C_Rec_NPr": "User Tags Individual Conversation Unprimed",
        # Assistant Tags (AT)
        "AT_PW-Q_Rec_Pr": "Assistant Tags Pairwise Query Primed",
        "AT_PW-Q_Rec_NPr": "Assistant Tags Pairwise Query Unprimed",
        "AT_PW-C_Rec_Pr": "Assistant Tags Pairwise Conversation Primed",
        "AT_PW-C_Rec_NPr": "Assistant Tags Pairwise Conversation Unprimed",
        "AT_IND-Q_Rec_Pr": "Assistant Tags Individual Query Primed",
        "AT_IND-Q_Rec_NPr": "Assistant Tags Individual Query Unprimed",
        "AT_IND-C_Rec_Pr": "Assistant Tags Individual Conversation Primed",
        "AT_IND-C_Rec_NPr": "Assistant Tags Individual Conversation Unprimed",
    }


def get_experiment_number(results_dir: Path) -> str | None:
    """
    Extract experiment number from results directory path.

    Args:
        results_dir: Path to results directory

    Returns:
        Experiment number (e.g., "11") or None if not found
    """
    experiment_name = results_dir.name
    if "_" in experiment_name:
        parts = experiment_name.split("_", 1)
        if parts[0].isdigit():
            return parts[0]
    return None


def get_experiment_code(results_dir: Path) -> str:
    """
    Extract experiment code from results directory path.

    Args:
        results_dir: Path to results directory

    Returns:
        Experiment code (e.g., "UT_PW-Q_Rec_NPr")
    """
    experiment_code = results_dir.name
    # Remove leading numbers and underscore if present
    if "_" in experiment_code:
        parts = experiment_code.split("_", 1)
        if parts[0].isdigit():
            experiment_code = parts[1]
    return experiment_code


def get_experiment_title(results_dir: Path) -> str:
    """
    Get full experiment title from results directory path.

    Args:
        results_dir: Path to results directory

    Returns:
        Full experiment name or code if not in mapping
    """
    code = get_experiment_code(results_dir)
    mapping = get_experiment_name_mapping()
    return mapping.get(code, code)


def derive_config_path(results_dir: Path) -> Path | None:
    """
    Given a results directory, attempt to locate the corresponding experiment config.

    Expected layout:
      results_dir: data/results/.../{experiment_name}
      config:      experiments/{experiment_name}/config.yaml
    """
    experiment_name = results_dir.name
    candidate = Path("experiments") / experiment_name / "config.yaml"
    return candidate if candidate.exists() else None


def load_model_type(config_path: Path | None) -> str | None:
    """Load model_type from a config.yaml if available."""
    if config_path is None or not config_path.exists():
        return None
    try:
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("model_type")
    except Exception:
        return None


def get_model_order(model_type: str | None = None) -> list[str]:
    """
    Define the canonical order for models in the pivot table and heatmap.

    Models are organized by company/provider, then ordered from weakest to strongest.

    Returns:
        List of model names in display order
    """
    if model_type and model_type.lower() == "cot":
        return [
            "gpt-oss-20b-thinking",
            "gpt-oss-120b-thinking",
            "sonnet-3.7-thinking",
            "grok-3-mini-thinking",
            "ll-3.3-70b-dsR1-thinking",
            "qwen-3.0-80b-thinking",
            "qwen-3.0-235b-thinking",
            "deepseek-r1-thinking",
            "kimi-k2-thinking",
        ]

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


def parse_eval_filename(filename: str) -> tuple[str, str, str] | None:
    """
    Extract evaluator, control, and treatment from eval log filename.

    Expected format: TIMESTAMP_{evaluator}-eval-on-{control}-vs-{treatment}_{UUID}.eval

    Returns:
        (evaluator, control, treatment) or None if parsing fails
    """
    try:
        # Remove timestamp and UUID
        parts = filename.split("_", 1)
        if len(parts) < 2:
            return None

        # Get task name (remove UUID and .eval)
        task_part = parts[1]
        task_name = task_part.rsplit("_", 1)[0]  # Remove UUID

        # Parse: {evaluator}-eval-on-{control}-vs-{treatment}
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
    Load sample-level binary results (correct=1, incorrect=0, failed=None) from eval logs.

    If multiple eval files exist for the same (evaluator, treatment) pair,
    uses the most recent successful evaluation.

    Returns None for samples with 'F' (failed/malformed) so they can be aligned
    and skipped in paired comparisons.

    Args:
        results_dir: Path to results directory

    Returns:
        Dictionary mapping (evaluator, treatment) -> list of outcomes (1/0/None)
    """
    print(f"Loading sample-level results from {results_dir.name}...")

    # First pass: collect all files per (evaluator, treatment) pair
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

    # Second pass: for each pair, use the most recent successful eval
    results = {}
    duplicates_found = 0

    for key, file_list in files_by_pair.items():
        evaluator, treatment = key

        # If multiple files, sort by timestamp (most recent first)
        if len(file_list) > 1:
            duplicates_found += 1
            file_list = sorted(file_list, key=lambda f: f.name, reverse=True)

        # Try files in order (most recent first) until we find a successful one
        for eval_file in file_list:
            try:
                log = read_eval_log(eval_file)

                # Skip if not successful
                if log.status != "success":
                    continue

                # Extract outcomes from all samples (including None for 'F')
                outcomes = []
                if log.samples:
                    for sample in log.samples:
                        outcome = None  # Default to None (failed/invalid)

                        # Check if sample was scored
                        if sample.scores:
                            score_obj = sample.scores.get("logprob_scorer")
                            if score_obj is not None and hasattr(score_obj, "value"):
                                score_value = score_obj.value
                                # Handle dict format: {'acc': 'C'/'I'/'F'}
                                if (
                                    isinstance(score_value, dict)
                                    and "acc" in score_value
                                ):
                                    acc_val = score_value["acc"]
                                    if acc_val == "C":
                                        outcome = 1
                                    elif acc_val == "I":
                                        outcome = 0
                                    # acc_val == 'F' -> stays None
                                # Handle direct string: 'C' or 'I'
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
                    break  # Found a successful eval, stop trying other files

            except Exception:
                # Skip files that can't be read, try next one
                continue

    if duplicates_found > 0:
        print(
            f"  ⚠ Found {duplicates_found} (evaluator, treatment) pairs with multiple eval files"
        )
        print("    Using most recent successful evaluation for each")

    print(f"  ✓ Loaded {len(results)} (evaluator, treatment) pairs")
    return results


def load_pivot_table(results_dir: Path, model_order: list[str]) -> pd.DataFrame:
    """
    Load pre-computed accuracy pivot table from analysis directory.

    Args:
        results_dir: Path to results directory (e.g., data/results/.../11_UT_PW-Q_Rec_NPr)

    Returns:
        Pivot table DataFrame

    Raises:
        FileNotFoundError: If pivot table doesn't exist
    """
    # Convert results path to analysis path
    # data/results/... -> data/analysis/...
    parts = list(results_dir.parts)
    if "results" in parts:
        results_idx = parts.index("results")
        parts[results_idx] = "analysis"
        analysis_dir = Path(*parts)
    else:
        raise ValueError(f"Expected 'results' in path: {results_dir}")

    pivot_path = analysis_dir / "accuracy_pivot.csv"

    if not pivot_path.exists():
        raise FileNotFoundError(
            f"Pivot table not found: {pivot_path}\n"
            f"Run analyze_pairwise_results.py first to generate it."
        )

    # Load with index column
    pivot = pd.read_csv(pivot_path, index_col=0)

    # Filter to only models that exist in the data
    row_order = [m for m in model_order if m in pivot.index]
    col_order = [m for m in model_order if m in pivot.columns]

    # If no overlap, keep original ordering to avoid empty pivots
    if not row_order:
        row_order = list(pivot.index)
    if not col_order:
        col_order = list(pivot.columns)

    # Reindex to apply ordering
    pivot = pivot.reindex(index=row_order, columns=col_order)

    return pivot


def compute_difference(
    pivot1: pd.DataFrame, pivot2: pd.DataFrame, model_order: list[str]
) -> pd.DataFrame:
    """
    Compute difference between two pivot tables (pivot1 - pivot2).

    Args:
        pivot1: First pivot table
        pivot2: Second pivot table

    Returns:
        Difference matrix
    """
    print("Computing difference (experiment1 - experiment2)...")

    # Get union of all models from both pivots, ordered by canonical order
    all_rows = [m for m in model_order if m in pivot1.index or m in pivot2.index]
    all_cols = [m for m in model_order if m in pivot1.columns or m in pivot2.columns]

    # Reindex both to canonical order (this ensures consistent ordering)
    pivot1_ordered = pivot1.reindex(index=all_rows, columns=all_cols)
    pivot2_ordered = pivot2.reindex(index=all_rows, columns=all_cols)

    # Align the two dataframes (in case they have different models)
    diff = pivot1_ordered.subtract(pivot2_ordered, fill_value=np.nan)

    return diff


def compute_mcnemar_tests(
    results1: dict[tuple[str, str], list[int]],
    results2: dict[tuple[str, str], list[int]],
    pivot1: pd.DataFrame,
    pivot2: pd.DataFrame,
    model_order: list[str],
) -> tuple[pd.DataFrame, dict]:
    """
    Perform McNemar's tests comparing sample-level binary results between two experiments.

    McNemar's test is appropriate for paired binary data. It focuses on discordant pairs:
    - b = cases where exp1 correct, exp2 incorrect
    - c = cases where exp1 incorrect, exp2 correct

    The test statistic is: chi2 = (b - c)^2 / (b + c)
    For small samples (b + c < 25), uses exact binomial test instead.

    Args:
        results1: Sample-level results from experiment 1
        results2: Sample-level results from experiment 2
        pivot1: Pivot table from experiment 1 (for structure)
        pivot2: Pivot table from experiment 2 (for structure)
        model_order: Canonical model ordering

    Returns:
        Tuple of (p_values DataFrame, overall_stats dict)
    """
    print("Performing McNemar's tests...")

    # Get union of all models from both pivots, ordered by canonical order
    all_rows = [m for m in model_order if m in pivot1.index or m in pivot2.index]
    all_cols = [m for m in model_order if m in pivot1.columns or m in pivot2.columns]

    # Initialize p-values matrix with NaN, using canonical order
    p_values = pd.DataFrame(np.nan, index=all_rows, columns=all_cols)

    # Collect all discordant pairs for overall test
    total_b = 0  # exp1 correct, exp2 incorrect
    total_c = 0  # exp1 incorrect, exp2 correct

    # For each (evaluator, treatment) cell
    tested_cells = 0
    significant_cells = 0

    for evaluator in all_rows:
        for treatment in all_cols:
            # Skip diagonal
            if evaluator == treatment:
                continue

            key = (evaluator, treatment)

            # Check if we have data from both experiments
            if key in results1 and key in results2:
                samples1 = results1[key]
                samples2 = results2[key]

                # Check that we have the same number of samples
                if len(samples1) == len(samples2) and len(samples1) > 0:
                    # Filter out positions where either experiment has None (failed samples)
                    valid_pairs = [
                        (s1, s2)
                        for s1, s2 in zip(samples1, samples2)
                        if s1 is not None and s2 is not None
                    ]

                    if len(valid_pairs) == 0:
                        continue

                    # Count discordant pairs
                    # b = exp1 correct (1), exp2 incorrect (0)
                    # c = exp1 incorrect (0), exp2 correct (1)
                    b = sum(1 for s1, s2 in valid_pairs if s1 == 1 and s2 == 0)
                    c = sum(1 for s1, s2 in valid_pairs if s1 == 0 and s2 == 1)

                    # Accumulate for overall test
                    total_b += b
                    total_c += c

                    # McNemar's test
                    n_discordant = b + c

                    if n_discordant == 0:
                        # No discordant pairs - no difference
                        p_value = 1.0
                    elif n_discordant < 25:
                        # Use exact binomial test for small samples
                        # H0: P(b) = P(c) = 0.5
                        # Two-sided test: probability of observing b or more extreme
                        p_value = stats.binomtest(
                            b, n_discordant, 0.5, alternative="two-sided"
                        ).pvalue
                    else:
                        # Use chi-squared approximation (with continuity correction)
                        # McNemar's chi-squared: (|b - c| - 1)^2 / (b + c)
                        chi2 = (abs(b - c) - 1) ** 2 / n_discordant
                        p_value = 1 - stats.chi2.cdf(chi2, df=1)

                    p_values.loc[evaluator, treatment] = p_value

                    tested_cells += 1
                    if p_value < 0.05:
                        significant_cells += 1

    print(f"  ✓ Performed {tested_cells} McNemar's tests")
    print(f"  ✓ Found {significant_cells} significant differences (p < 0.05)")

    # Overall test: McNemar's test on aggregated discordant pairs
    # H0: P(exp1 correct, exp2 incorrect) = P(exp1 incorrect, exp2 correct)
    overall_stats = None
    n_discordant_total = total_b + total_c

    if n_discordant_total > 0:
        # Calculate effect size: proportion of discordant pairs favoring exp1
        prop_favoring_exp1 = (
            total_b / n_discordant_total if n_discordant_total > 0 else 0.5
        )

        if n_discordant_total < 25:
            # Exact binomial test
            p_value_overall = stats.binomtest(
                total_b, n_discordant_total, 0.5, alternative="two-sided"
            ).pvalue
            test_type = "exact binomial"
        else:
            # Chi-squared approximation with continuity correction
            chi2_overall = (abs(total_b - total_c) - 1) ** 2 / n_discordant_total
            p_value_overall = 1 - stats.chi2.cdf(chi2_overall, df=1)
            test_type = "chi-squared"

        overall_stats = {
            "n_discordant": n_discordant_total,
            "b_exp1_better": total_b,
            "c_exp2_better": total_c,
            "prop_favoring_exp1": prop_favoring_exp1,
            "test_type": test_type,
            "p_value": p_value_overall,
            "significant": p_value_overall < 0.05,
        }
        print(f"  ✓ Overall McNemar's test ({test_type}): p={p_value_overall:.4f}")
        print(f"    Discordant pairs: {total_b} favor exp1, {total_c} favor exp2")
    else:
        print("  ⚠ No discordant pairs found for overall test")

    return p_values, overall_stats


def plot_difference_heatmap(
    diff: pd.DataFrame,
    output_path: Path,
    exp1_title: str,
    exp2_title: str,
    p_values: pd.DataFrame | None = None,
):
    """
    Create and save heatmap of accuracy differences.

    Args:
        diff: Difference matrix (exp1 - exp2)
        output_path: Path to save the heatmap image
        exp1_title: Title of first experiment
        exp2_title: Title of second experiment
        p_values: Optional p-values matrix from paired t-tests (bold if p < 0.05)
    """
    print("Generating difference heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create mask for diagonal (evaluator == treatment) and NaN values
    mask = pd.DataFrame(False, index=diff.index, columns=diff.columns)
    for model in diff.index:
        if model in diff.columns:
            mask.loc[model, model] = True

    # Create heatmap with diverging colormap
    # Red = negative (exp2 better), White = no change, Green = positive (exp1 better)
    sns.heatmap(
        diff,
        annot=False,  # We'll add custom annotations
        cmap="RdYlGn",  # Diverging: red-yellow-green
        center=0.0,  # Center at zero difference
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Accuracy Difference (Exp1 - Exp2)"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Add custom annotations (bold if significant)
    for i, evaluator in enumerate(diff.index):
        for j, treatment in enumerate(diff.columns):
            if evaluator == treatment:
                continue  # Skip diagonal

            val = diff.loc[evaluator, treatment]
            if pd.notna(val):
                # Check if significant
                is_significant = False
                if p_values is not None and pd.notna(
                    p_values.loc[evaluator, treatment]
                ):
                    is_significant = p_values.loc[evaluator, treatment] < 0.05

                # Format text
                text = f"{val:.2f}"
                fontweight = "bold" if is_significant else "normal"
                fontsize = 9 if is_significant else 8

                ax.text(
                    j + 0.5,
                    i + 0.5,
                    text,
                    ha="center",
                    va="center",
                    fontsize=fontsize,
                    fontweight=fontweight,
                    color="black",
                )

    # Fill diagonal with gray
    for i, model in enumerate(diff.index):
        if model in diff.columns:
            j = list(diff.columns).index(model)
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
    add_provider_boundaries(ax, diff)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Multi-line title
    title = (
        f"Self-Recognition Accuracy Difference\n"
        f"({exp1_title})\n"
        f"MINUS\n"
        f"({exp2_title})"
    )

    ax.set_title(
        title,
        fontsize=13,
        fontweight="bold",
        pad=20,
    )

    # Add legend explaining bold values
    legend_text = (
        "Bold values indicate statistically significant differences (p < 0.05)"
    )
    fig.text(
        0.5,  # Center horizontally
        0.02,  # Near bottom
        legend_text,
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

    # Rotate labels for readability
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Tight layout
    plt.tight_layout()

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved difference heatmap to: {output_path}")

    plt.close()


def generate_summary_stats(
    diff: pd.DataFrame,
    output_path: Path,
    exp1_title: str,
    exp2_title: str,
    p_values: pd.DataFrame | None = None,
    overall_stats: dict | None = None,
):
    """Generate and save comparison summary statistics."""
    print("Generating summary statistics...")

    # Flatten and remove NaN/diagonal
    flat_diff = []
    for i, evaluator in enumerate(diff.index):
        for j, treatment in enumerate(diff.columns):
            if evaluator != treatment:  # Skip diagonal
                val = diff.iloc[i, j]
                if pd.notna(val):
                    flat_diff.append(val)

    flat_diff = np.array(flat_diff)

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("EXPERIMENT COMPARISON ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        f.write("EXPERIMENTS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Experiment 1: {exp1_title}\n")
        f.write(f"Experiment 2: {exp2_title}\n")
        f.write("Difference: Experiment 1 - Experiment 2\n\n")

        f.write("INTERPRETATION\n")
        f.write("-" * 70 + "\n")
        f.write("Positive values: Experiment 1 has HIGHER accuracy\n")
        f.write("Negative values: Experiment 2 has HIGHER accuracy\n")
        f.write("Values near 0: Similar performance\n")
        if p_values is not None:
            f.write("Bold values in heatmap: Statistically significant (p < 0.05)\n\n")
        else:
            f.write("\n")

        # Overall statistical test
        if overall_stats:
            f.write("OVERALL McNEMAR'S TEST\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total discordant pairs: {overall_stats['n_discordant']}\n")
            f.write(
                f"  Exp1 correct, Exp2 incorrect (b): {overall_stats['b_exp1_better']}\n"
            )
            f.write(
                f"  Exp1 incorrect, Exp2 correct (c): {overall_stats['c_exp2_better']}\n"
            )
            f.write(
                f"Proportion favoring Exp1: {overall_stats['prop_favoring_exp1']:.3f}\n"
            )
            f.write(f"Test type: {overall_stats['test_type']}\n")
            f.write(f"p-value: {overall_stats['p_value']:.6f}\n")
            f.write(
                f"Result: {'SIGNIFICANT' if overall_stats['significant'] else 'NOT SIGNIFICANT'} at α=0.05\n"
            )
            if overall_stats["significant"]:
                if overall_stats["b_exp1_better"] > overall_stats["c_exp2_better"]:
                    f.write("Conclusion: Experiment 1 has HIGHER accuracy overall\n")
                else:
                    f.write("Conclusion: Experiment 2 has HIGHER accuracy overall\n")
            else:
                f.write(
                    "Conclusion: No significant overall difference between experiments\n"
                )
            f.write("\n")

        # Individual cell statistics
        if p_values is not None:
            sig_count = 0
            total_tests = 0
            for evaluator in p_values.index:
                for treatment in p_values.columns:
                    if evaluator != treatment:  # Skip diagonal
                        p_val = p_values.loc[evaluator, treatment]
                        if pd.notna(p_val):
                            total_tests += 1
                            if p_val < 0.05:
                                sig_count += 1

            if total_tests > 0:
                f.write("CELL-WISE McNEMAR'S TESTS\n")
                f.write("-" * 70 + "\n")
                f.write(f"Total tests performed: {total_tests}\n")
                f.write(
                    f"Significant differences (p < 0.05): {sig_count} ({sig_count/total_tests*100:.1f}%)\n"
                )
                f.write(
                    f"Non-significant: {total_tests - sig_count} ({(total_tests-sig_count)/total_tests*100:.1f}%)\n\n"
                )
            else:
                f.write("CELL-WISE McNEMAR'S TESTS\n")
                f.write("-" * 70 + "\n")
                f.write(
                    "No valid McNemar's tests could be performed (insufficient overlapping data)\n\n"
                )

        if len(flat_diff) > 0:
            f.write("OVERALL DIFFERENCE STATISTICS (from aggregated accuracies)\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean difference: {np.mean(flat_diff):.3f}\n")
            f.write(f"Median difference: {np.median(flat_diff):.3f}\n")
            f.write(f"Std deviation: {np.std(flat_diff):.3f}\n")
            f.write(f"Min difference: {np.min(flat_diff):.3f}\n")
            f.write(f"Max difference: {np.max(flat_diff):.3f}\n\n")

            # Count improvements/degradations
            positive = (flat_diff > 0.05).sum()  # Exp1 better by >5%
            negative = (flat_diff < -0.05).sum()  # Exp2 better by >5%
            similar = ((flat_diff >= -0.05) & (flat_diff <= 0.05)).sum()

            f.write("PERFORMANCE SHIFTS (±5% threshold)\n")
            f.write("-" * 70 + "\n")
            f.write(
                f"Exp1 better: {positive} comparisons ({positive/len(flat_diff)*100:.1f}%)\n"
            )
            f.write(
                f"Exp2 better: {negative} comparisons ({negative/len(flat_diff)*100:.1f}%)\n"
            )
            f.write(
                f"Similar: {similar} comparisons ({similar/len(flat_diff)*100:.1f}%)\n\n"
            )

            # Evaluator-wise summary
            f.write("EVALUATOR-WISE MEAN DIFFERENCES\n")
            f.write("-" * 70 + "\n")
            evaluator_diffs = {}
            evaluator_significance = {}

            # Calculate significance for each evaluator using one-sample t-test
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

                    # Perform one-sample t-test: test if mean is significantly different from 0
                    if len(row_vals) > 1:
                        t_stat, p_val = stats.ttest_1samp(row_vals, 0.0)
                        evaluator_significance[evaluator] = p_val
                        sig_marker = (
                            "***"
                            if p_val < 0.001
                            else "**"
                            if p_val < 0.01
                            else "*"
                            if p_val < 0.05
                            else ""
                        )
                        f.write(f"{evaluator:25s}: {mean_diff:+.3f} {sig_marker}\n")
                    else:
                        evaluator_significance[evaluator] = None
                        f.write(f"{evaluator:25s}: {mean_diff:+.3f}\n")

            f.write("\nSignificance: *** p<0.001, ** p<0.01, * p<0.05\n")

            # Generate evaluator performance plot
            if evaluator_diffs:
                evaluator_plot_path = (
                    output_path.parent / "evaluator_difference_performance.png"
                )
                evaluator_series = pd.Series(evaluator_diffs).sort_values(
                    ascending=False
                )
                significance_series = pd.Series(evaluator_significance).reindex(
                    evaluator_series.index
                )
                plot_evaluator_performance(
                    evaluator_series,
                    evaluator_plot_path,
                    experiment_title=f"{exp1_title} - {exp2_title}",
                    ylabel="Mean Difference (Exp1 - Exp2)",
                    significance_pvalues=significance_series,
                )

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

    # Add vertical line at zero for difference plots
    if any(v < 0 for v in evaluator_avg.values) and any(
        v > 0 for v in evaluator_avg.values
    ):
        ax.axvline(x=0, color="black", linestyle="--", linewidth=1, alpha=0.5)

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
        x_min = min_val - padding
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
        description="Compare two pairwise self-recognition experiments"
    )
    parser.add_argument(
        "--experiment1",
        type=str,
        required=True,
        help="Path to first experiment results directory",
    )
    parser.add_argument(
        "--experiment2",
        type=str,
        required=True,
        help="Path to second experiment results directory",
    )
    parser.add_argument(
        "--config1",
        type=str,
        default=None,
        help="Optional path to config for experiment1 (to infer model_type)",
    )
    parser.add_argument(
        "--config2",
        type=str,
        default=None,
        help="Optional path to config for experiment2 (to infer model_type)",
    )
    parser.add_argument(
        "--model_type1",
        type=str,
        choices=["CoT", "DR"],
        default=None,
        help='Override model set for experiment1 ("CoT" uses thinking subset)',
    )
    parser.add_argument(
        "--model_type2",
        type=str,
        choices=["CoT", "DR"],
        default=None,
        help='Override model set for experiment2 ("CoT" uses thinking subset)',
    )

    args = parser.parse_args()

    # Convert to Path objects
    exp1_dir = Path(args.experiment1)
    exp2_dir = Path(args.experiment2)

    if not exp1_dir.exists():
        print(f"❌ Error: Experiment 1 directory not found: {exp1_dir}")
        return
    if not exp2_dir.exists():
        print(f"❌ Error: Experiment 2 directory not found: {exp2_dir}")
        return

    # Resolve config paths (provided or derived) and model types
    config1_path = Path(args.config1) if args.config1 else derive_config_path(exp1_dir)
    config2_path = Path(args.config2) if args.config2 else derive_config_path(exp2_dir)

    model_type1 = args.model_type1 or load_model_type(config1_path)
    model_type2 = args.model_type2 or load_model_type(config2_path)

    model_order1 = get_model_order(model_type1)
    model_order2 = get_model_order(model_type2)

    # Combined model order preserves ordering across both experiments
    combined_model_order: list[str] = []
    for lst in (model_order1, model_order2):
        for m in lst:
            if m not in combined_model_order:
                combined_model_order.append(m)
    if not combined_model_order:
        combined_model_order = model_order1 or model_order2

    # Get experiment codes and titles
    exp1_code = get_experiment_code(exp1_dir)
    exp2_code = get_experiment_code(exp2_dir)
    exp1_title = get_experiment_title(exp1_dir)
    exp2_title = get_experiment_title(exp2_dir)

    # Parse dataset and subset from both experiment paths
    # Expected format: data/results/{dataset}/{subset}/{experiment_code}
    def parse_dataset_info(results_dir: Path) -> tuple[str, str] | None:
        """Extract (dataset, subset) from results directory path."""
        parts = list(results_dir.parts)
        if "results" in parts:
            results_idx = parts.index("results")
            if (
                results_idx + 2 < len(parts) - 1
            ):  # Need dataset, subset, and experiment dir
                dataset = parts[results_idx + 1]
                subset = parts[results_idx + 2]
                return dataset, subset
        return None

    exp1_info = parse_dataset_info(exp1_dir)
    exp2_info = parse_dataset_info(exp2_dir)

    # Determine if this is a cross-dataset comparison
    is_cross_dataset = False
    if exp1_info and exp2_info:
        dataset1, subset1 = exp1_info
        dataset2, subset2 = exp2_info
        is_cross_dataset = (dataset1 != dataset2) or (subset1 != subset2)

    # Get experiment numbers for path naming
    exp1_num = get_experiment_number(exp1_dir)
    exp2_num = get_experiment_number(exp2_dir)

    # Setup output directory based on comparison type
    if is_cross_dataset:
        # Cross-dataset comparison
        # Path: data/analysis/cross-dataset_comparisons/{exp_num}_{experiment_code}/{dataset1_subset1}_vs_{dataset2_subset2}
        dataset1, subset1 = exp1_info
        dataset2, subset2 = exp2_info

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
        # Same dataset/subset comparison (original logic)
        # Extract dataset/subset path from experiment 1
        # e.g., data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr
        # -> data/analysis/pku_saferlhf/mismatch_1-20/comparisons/{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}
        parts = list(exp1_dir.parts)
        if "results" in parts:
            results_idx = parts.index("results")
            # Replace 'results' with 'analysis'
            parts[results_idx] = "analysis"
            # Remove the experiment-specific directory (last part)
            parts = parts[:-1]
            # Add 'comparisons' subdirectory and comparison name
            base_analysis_path = Path(*parts)
            # Build comparison name with experiment numbers
            if exp1_num and exp2_num:
                comparison_name = f"{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}"
            else:
                comparison_name = f"{exp1_code}_vs_{exp2_code}"
            output_dir = base_analysis_path / "comparisons" / comparison_name
        else:
            # Fallback if path doesn't contain 'results'
            if exp1_num and exp2_num:
                comparison_name = f"{exp1_num}_{exp1_code}_vs_{exp2_num}_{exp2_code}"
            else:
                comparison_name = f"{exp1_code}_vs_{exp2_code}"
            output_dir = Path("data/analysis/comparisons") / comparison_name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("EXPERIMENT COMPARISON")
    if is_cross_dataset:
        print("(Cross-Dataset Comparison)")
    print(f"{'='*70}")
    print(f"Experiment 1: {exp1_title}")
    print(f"              {exp1_dir}")
    if is_cross_dataset and exp1_info:
        print(f"              Dataset: {exp1_info[0]}/{exp1_info[1]}")
    if model_type1:
        print(f"              model_type: {model_type1}")
    print(f"Experiment 2: {exp2_title}")
    print(f"              {exp2_dir}")
    if is_cross_dataset and exp2_info:
        print(f"              Dataset: {exp2_info[0]}/{exp2_info[1]}")
    if model_type2:
        print(f"              model_type: {model_type2}")
    print(f"Output dir:   {output_dir}")
    print(f"{'='*70}\n")

    # Load pivot tables
    try:
        print("Loading pivot tables...")
        pivot1 = load_pivot_table(exp1_dir, model_order1)
        print(
            f"  ✓ Loaded experiment 1: {pivot1.shape[0]} evaluators × {pivot1.shape[1]} treatments"
        )
        pivot2 = load_pivot_table(exp2_dir, model_order2)
        print(
            f"  ✓ Loaded experiment 2: {pivot2.shape[0]} evaluators × {pivot2.shape[1]} treatments\n"
        )
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return

    # Compute difference
    diff = compute_difference(pivot1, pivot2, combined_model_order)
    print(
        f"  ✓ Computed differences: {diff.shape[0]} evaluators × {diff.shape[1]} treatments\n"
    )

    # Load sample-level results for statistical testing
    print("Loading sample-level data for statistical tests...\n")
    results1 = load_sample_level_results(exp1_dir)
    results2 = load_sample_level_results(exp2_dir)
    print()

    # Perform paired t-tests
    p_values, overall_stats = compute_mcnemar_tests(
        results1, results2, pivot1, pivot2, combined_model_order
    )
    print()

    # Save difference matrix
    diff_csv_path = output_dir / "accuracy_difference.csv"
    diff.to_csv(diff_csv_path)
    print(f"  ✓ Saved difference matrix to: {diff_csv_path}\n")

    # Save p-values matrix
    pvalues_csv_path = output_dir / "pvalues.csv"
    p_values.to_csv(pvalues_csv_path)
    print(f"  ✓ Saved p-values matrix to: {pvalues_csv_path}\n")

    # Generate difference heatmap (with significant values bolded)
    heatmap_path = output_dir / "accuracy_difference_heatmap.png"
    plot_difference_heatmap(diff, heatmap_path, exp1_title, exp2_title, p_values)
    print()

    # Generate summary stats
    summary_path = output_dir / "experiment_comparison_stats.txt"
    generate_summary_stats(
        diff, summary_path, exp1_title, exp2_title, p_values, overall_stats
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Accuracy Difference Matrix (Exp1 - Exp2)")
    print(f"{'='*70}\n")
    print(diff.round(3))
    print()

    print(f"{'='*70}")
    print("COMPARISON COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • accuracy_difference.csv: Difference matrix")
    print("  • pvalues.csv: P-values from paired t-tests")
    print("  • accuracy_difference_heatmap.png: Visualization (bold = significant)")
    print("  • evaluator_difference_performance.png: Evaluator performance bar chart")
    print("  • experiment_comparison_stats.txt: Comparison statistics & t-test results")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
