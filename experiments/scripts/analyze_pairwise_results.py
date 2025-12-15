#!/usr/bin/env python3
"""
Analyze pairwise self-recognition experiment results.

Takes a directory of eval logs, aggregates accuracy across all evaluations,
and generates pivot tables and heatmaps showing self-recognition performance.

Usage:
    uv run experiments/scripts/analyze_pairwise_results.py \
        --results_dir data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr

Output:
    - data/analysis/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr/
        - accuracy_pivot.csv: Raw accuracy data
        - accuracy_heatmap.png: Visualization
        - summary_stats.txt: Overall statistics
"""

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from inspect_ai.log import read_eval_log
import yaml
from scipy.stats import binomtest


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
            "sonnet-4.5-thinking",
            "opus-4.1-thinking",
            "gemini-2.5-flash-thinking",
            "gemini-2.5-pro-thinking",
            "ll-3.3-70b-dsR1-thinking",
            "qwen-3.0-80b-thinking",
            "deepseek-r1-thinking",
        ]

    return [
        # OpenAI (weakest to strongest)
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
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
        # Format: YYYY-MM-DDTHH-MM-SS+TZ_{task_name}_{uuid}.eval
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


def extract_accuracy(log) -> float | None:
    """
    Extract accuracy from an eval log.

    Handles partial failures gracefully:
    - Samples with "F" (failed/malformed) are skipped (e.g., due to token limits)
    - Only "C" (correct) and "I" (incorrect) samples are counted
    - Returns accuracy if at least one valid sample exists, otherwise None
    - This allows eval files with occasional failures to still be considered "scored"

    Returns:
        Accuracy as a float (0.0 to 1.0), or None if not available
    """
    try:
        # Check if log has results
        if log.status != "success":
            return None

        # Count correct answers from samples
        # Note: We count from samples directly rather than relying on log.results
        # because some eval logs may have sample scores but missing aggregated results
        if not log.samples:
            return None

        correct_count = 0
        total_count = 0

        for sample in log.samples:
            if not sample.scores:
                continue

            # Find acc score
            for score in sample.scores.values():
                if hasattr(score, "value") and isinstance(score.value, dict):
                    if "acc" in score.value:
                        acc_val = score.value["acc"]
                        # Only count C (correct) and I (incorrect) samples
                        # Skip F (failed/malformed) - can't assess attribution
                        # F samples occur when model hits token limit, produces invalid output, etc.
                        if acc_val == "C":
                            total_count += 1
                            correct_count += 1
                        elif acc_val == "I":
                            total_count += 1
                        # If acc_val == "F", skip entirely (partial failures are OK)
                        break

        # Return None only if ALL samples failed (no valid samples to score)
        # If at least one sample is C or I, return accuracy (handles partial failures)
        if total_count == 0:
            return None

        return correct_count / total_count

    except Exception as e:
        print(f"  Warning: Error extracting accuracy: {e}")
        return None


def load_all_evaluations(results_dir: Path) -> pd.DataFrame:
    """
    Load all evaluation logs from a directory and extract key metrics.

    Returns:
        DataFrame with columns: evaluator, control, treatment, accuracy, n_samples, status
    """
    eval_files = list(results_dir.glob("*.eval"))

    print(f"Found {len(eval_files)} eval log files")
    print("Processing...\n")

    data = []
    skipped = 0
    errors = 0

    for eval_file in eval_files:
        try:
            # Parse filename
            parsed = parse_eval_filename(eval_file.name)
            if not parsed:
                skipped += 1
                continue

            evaluator, control, treatment = parsed

            # Read log
            log = read_eval_log(eval_file)

            # Extract accuracy
            accuracy = extract_accuracy(log)
            n_samples = len(log.samples) if log.samples else 0

            data.append(
                {
                    "evaluator": evaluator,
                    "control": control,
                    "treatment": treatment,
                    "accuracy": accuracy,
                    "n_samples": n_samples,
                    "status": log.status,
                    "filename": eval_file.name,
                }
            )

        except Exception as e:
            errors += 1
            print(f"  ⚠ Error reading {eval_file.name}: {e}")

    print(f"\nLoaded {len(data)} evaluations")
    print(f"  Skipped: {skipped} (couldn't parse filename)")
    print(f"  Errors: {errors}\n")

    return pd.DataFrame(data)


def create_pivot_table(
    df: pd.DataFrame, model_order: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create pivot table of accuracy by evaluator and treatment, along with counts.

    For pairwise comparisons, the evaluator judges between control (self) and treatment.
    Accuracy = how often the evaluator correctly identifies its own output.

    Args:
        df: DataFrame with evaluator, control, treatment, accuracy, n_samples columns

    Returns:
        Tuple of:
          - accuracy pivot (rows=evaluator, cols=treatment)
          - count pivot (total samples contributing to each cell)
          - correct pivot (total correct predictions contributing to each cell)
    """
    # Filter to successful evaluations only and drop rows with missing accuracy
    df_success = df[(df["status"] == "success") & (df["accuracy"].notna())].copy()

    print(f"Creating pivot table from {len(df_success)} successful evaluations...")

    # Compute total correct per evaluation (accuracy * n_samples)
    df_success["correct"] = df_success["accuracy"] * df_success["n_samples"]

    # Aggregate by evaluator/treatment
    grouped = df_success.groupby(["evaluator", "treatment"]).agg(
        total_correct=("correct", "sum"),
        total_samples=("n_samples", "sum"),
    )

    # Accuracy pivot: total_correct / total_samples
    pivot_accuracy = (grouped["total_correct"] / grouped["total_samples"]).unstack(
        fill_value=pd.NA
    )
    pivot_counts = grouped["total_samples"].unstack(fill_value=0)
    pivot_correct = grouped["total_correct"].unstack(fill_value=0.0)

    # Filter to only models that exist in the data
    row_order = [m for m in model_order if m in pivot_accuracy.index]
    col_order = [m for m in model_order if m in pivot_accuracy.columns]

    # Reindex to apply ordering
    pivot_accuracy = pivot_accuracy.reindex(index=row_order, columns=col_order)
    pivot_counts = pivot_counts.reindex(index=row_order, columns=col_order)
    pivot_correct = pivot_correct.reindex(index=row_order, columns=col_order)

    return pivot_accuracy, pivot_counts, pivot_correct


def compute_asymmetry_analysis(
    pivot: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Compute asymmetry between evaluator and evaluatee roles.

    Returns:
        - asymmetry_matrix: Cell-wise differences (pivot - pivot.T)
        - row_col_comparison: DataFrame with row means, column means, and differences
        - asymmetry_scores: Series of asymmetry scores per model (row_mean - col_mean)
    """
    # Compute row means (when model is evaluator)
    row_means = pivot.mean(axis=1, skipna=True)

    # Compute column means (when model is treatment/evaluatee)
    col_means = pivot.mean(axis=0, skipna=True)

    # Create comparison DataFrame
    comparison = pd.DataFrame(
        {
            "row_mean": row_means,
            "col_mean": col_means,
        }
    )

    # Compute difference (row_mean - col_mean)
    comparison["difference"] = comparison["row_mean"] - comparison["col_mean"]
    comparison = comparison.sort_values("difference", ascending=False)

    # Compute cell-wise asymmetry: pivot - pivot.T
    # This shows for each pair (A, B): how much better A identifies itself vs B
    # compared to how well B identifies itself vs A
    asymmetry_matrix = pivot - pivot.T

    return asymmetry_matrix, comparison, comparison["difference"]


def plot_asymmetry_heatmap(
    asymmetry_matrix: pd.DataFrame, output_path: Path, experiment_title: str = ""
):
    """
    Create heatmap showing asymmetry between evaluator and evaluatee roles.

    Values show: pivot[A, B] - pivot[B, A]
    - Positive: Model A is better at identifying itself vs B than B is vs A
    - Negative: Model B is better at identifying itself vs A than A is vs B
    - Magnitude indicates strength of asymmetry

    Args:
        asymmetry_matrix: Matrix of differences (pivot - pivot.T)
        output_path: Path to save the heatmap
        experiment_title: Optional experiment name
    """
    print("Generating asymmetry heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create mask for diagonal and NaN values (missing data)
    mask = pd.DataFrame(
        False, index=asymmetry_matrix.index, columns=asymmetry_matrix.columns
    )
    for model in asymmetry_matrix.index:
        if model in asymmetry_matrix.columns:
            # Mask diagonal
            mask.loc[model, model] = True
        # Mask NaN values (missing data for that pair)
        for col in asymmetry_matrix.columns:
            if pd.isna(asymmetry_matrix.loc[model, col]):
                mask.loc[model, col] = True

    # Create heatmap with same colormap as accuracy heatmap
    # Use RdYlGn: red (negative) → yellow (zero) → green (positive)
    sns.heatmap(
        asymmetry_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Same as accuracy heatmap: Red-Yellow-Green
        center=0.0,
        vmin=-1.0,
        vmax=1.0,
        cbar_kws={"label": "Asymmetry (Row - Column)"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Fill diagonal and missing data cells with gray squares
    for i, model in enumerate(asymmetry_matrix.index):
        for j, col in enumerate(asymmetry_matrix.columns):
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
    add_provider_boundaries(ax, asymmetry_matrix)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Build title
    if experiment_title:
        title = f"Evaluator vs Evaluatee Asymmetry: {experiment_title}\n(Cell value = How well row model identifies itself vs column model\nminus how well column model identifies itself vs row model)"
    else:
        title = "Evaluator vs Evaluatee Asymmetry\n(Cell value = Row model performance - Column model performance)"

    ax.set_title(
        title,
        fontsize=14,
        fontweight="bold",
        pad=20,
    )

    # Rotate labels
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Tight layout
    plt.tight_layout()
    # Increase bottom margin significantly to make room for note and rotated x-axis labels
    plt.subplots_adjust(bottom=0.20)

    # Add interpretation note (moved further down to avoid overlap with x-axis labels)
    fig.text(
        0.5,
        0.01,
        "Positive (green): Row model better at self-identification | "
        "Negative (red): Column model better at self-identification | "
        "Zero (yellow): Balanced",
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

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved asymmetry heatmap to: {output_path}")

    plt.close()


def plot_row_vs_column_comparison(
    comparison: pd.DataFrame, output_path: Path, experiment_title: str = ""
):
    """
    Create bar plot comparing row means vs column means for each model.

    This shows:
    - Row mean: Average accuracy when model is evaluator (identifying itself)
    - Column mean: Average accuracy when model is treatment (others identifying it)
    - Difference: Row mean - Column mean

    Positive difference suggests model has distinctive style.
    Negative difference suggests model's outputs are easily identified by others.
    """
    print("Generating row vs column comparison plot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    models = comparison.index
    x_pos = range(len(models))

    # Plot bars
    width = 0.35
    ax.bar(
        [x - width / 2 for x in x_pos],
        comparison["row_mean"],
        width,
        label="As Evaluator (Row Mean)",
        color="steelblue",
        alpha=0.8,
    )
    ax.bar(
        [x + width / 2 for x in x_pos],
        comparison["col_mean"],
        width,
        label="As Treatment (Column Mean)",
        color="coral",
        alpha=0.8,
    )

    # Add difference line
    ax2 = ax.twinx()
    colors = ["green" if d > 0 else "red" for d in comparison["difference"]]
    ax2.bar(
        x_pos,
        comparison["difference"],
        width=0.2,
        label="Difference (Row - Col)",
        color=colors,
        alpha=0.6,
    )
    ax2.axhline(y=0, color="black", linestyle="--", linewidth=0.8)
    ax2.set_ylabel(
        "Difference (Row Mean - Column Mean)", fontsize=11, fontweight="bold"
    )

    # Labels
    ax.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Accuracy", fontsize=12, fontweight="bold")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.set_ylim(0, 1.0)

    # Title
    if experiment_title:
        title = f"Evaluator vs Evaluatee Performance: {experiment_title}\n(Row Mean: Model as evaluator | Column Mean: Model as treatment)"
    else:
        title = "Evaluator vs Evaluatee Performance\n(Row Mean: Model as evaluator | Column Mean: Model as treatment)"

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # Legend
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Add interpretation
    fig.text(
        0.5,
        0.02,
        "Positive difference (green): Model better at identifying itself than others are at identifying it (suggests distinctive style)\n"
        "Negative difference (red): Others identify it easily, but model struggles to identify itself (suggests quality-based bias)",
        ha="center",
        fontsize=9,
        style="italic",
        bbox=dict(
            boxstyle="round,pad=0.5",
            facecolor="lightyellow",
            edgecolor="gray",
            linewidth=1,
        ),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.12)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved row vs column comparison to: {output_path}")

    plt.close()


def plot_heatmap(
    pivot: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
    p_values: pd.DataFrame | None = None,
    alpha: float = 0.05,
):
    """
    Create and save heatmap of self-recognition accuracy.

    Args:
        pivot: Pivot table with evaluators as rows, treatments as columns
        output_path: Path to save the heatmap image
        experiment_title: Optional experiment name to include in the title
    """
    print("Generating heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create mask for diagonal (evaluator == treatment)
    mask = pd.DataFrame(False, index=pivot.index, columns=pivot.columns)
    for model in pivot.index:
        if model in pivot.columns:
            mask.loc[model, model] = True

    # Create heatmap
    hm = sns.heatmap(
        pivot,
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

    # Bold significant cells if p-values provided
    if p_values is not None:
        # Seaborn's hm.texts only includes unmasked cells; build a position->text map
        text_map: dict[tuple[int, int], any] = {}
        for text in hm.texts:
            x, y = text.get_position()
            # Heatmap text positions are at (col + 0.5, row + 0.5)
            j = int(round(x - 0.5))
            i = int(round(y - 0.5))
            text_map[(i, j)] = text

        for i, row in enumerate(pivot.index):
            for j, col in enumerate(pivot.columns):
                text = text_map.get((i, j))
                if text is None:
                    continue
                p_val = (
                    p_values.loc[row, col]
                    if (row in p_values.index and col in p_values.columns)
                    else None
                )
                if p_val is not None and pd.notna(p_val) and p_val < alpha:
                    text.set_fontweight("bold")

    # Fill diagonal with gray
    for i, model in enumerate(pivot.index):
        if model in pivot.columns:
            j = list(pivot.columns).index(model)
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
    add_provider_boundaries(ax, pivot)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Build title with optional experiment name
    if experiment_title:
        title = f"Self-Recognition Accuracy Matrix: {experiment_title}\n(How well each model identifies its own outputs vs. others)"
    else:
        title = "Self-Recognition Accuracy Matrix\n(How well each model identifies its own outputs vs. others)"

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
    print(f"  ✓ Saved heatmap to: {output_path}")

    plt.close()


def compute_significance_matrix(
    pivot_correct: pd.DataFrame, pivot_counts: pd.DataFrame, alpha: float = 0.05
) -> pd.DataFrame:
    """
    Compute p-values for accuracy vs chance (0.5) using exact binomial tests.

    Args:
        pivot_correct: Matrix of total correct counts
        pivot_counts: Matrix of total sample counts
        alpha: Significance threshold (unused here but kept for API symmetry)

    Returns:
        DataFrame of p-values (NaN where n=0)
    """
    p_values = pd.DataFrame(
        pd.NA, index=pivot_correct.index, columns=pivot_correct.columns
    )

    for r in pivot_correct.index:
        for c in pivot_correct.columns:
            n = pivot_counts.loc[r, c]
            k = pivot_correct.loc[r, c]
            if pd.notna(n) and n > 0 and pd.notna(k):
                try:
                    res = binomtest(
                        int(round(k)), int(round(n)), p=0.5, alternative="two-sided"
                    )
                    p_values.loc[r, c] = res.pvalue
                except Exception:
                    p_values.loc[r, c] = pd.NA

    return p_values


def generate_summary_stats(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    pivot_counts: pd.DataFrame,
    p_values: pd.DataFrame | None,
    comparison: pd.DataFrame,
    asymmetry_matrix: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
):
    """Generate and save summary statistics."""

    print("Generating summary statistics...")

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SELF-RECOGNITION EXPERIMENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total evaluations: {len(df)}\n")
        f.write(f"  Successful: {(df['status'] == 'success').sum()}\n")
        f.write(f"  Failed: {(df['status'] == 'error').sum()}\n")
        f.write(f"  Cancelled: {(df['status'] == 'cancelled').sum()}\n")
        f.write(f"  Started (incomplete): {(df['status'] == 'started').sum()}\n\n")

        # Accuracy stats
        df_success = df[df["status"] == "success"]
        if len(df_success) > 0 and df_success["accuracy"].notna().any():
            f.write("ACCURACY STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write(f"Mean accuracy: {df_success['accuracy'].mean():.3f}\n")
            f.write(f"Median accuracy: {df_success['accuracy'].median():.3f}\n")
            f.write(f"Std deviation: {df_success['accuracy'].std():.3f}\n")
            f.write(f"Min accuracy: {df_success['accuracy'].min():.3f}\n")
            f.write(f"Max accuracy: {df_success['accuracy'].max():.3f}\n\n")

        # Model performance
        f.write("EVALUATOR PERFORMANCE (Average Accuracy)\n")
        f.write("-" * 70 + "\n")
        evaluator_avg = (
            df_success.groupby("evaluator")["accuracy"]
            .mean()
            .sort_values(ascending=False)
        )
        for model, acc in evaluator_avg.items():
            f.write(f"  {model:30} {acc:.3f}\n")
        f.write("\n")

        # Generate evaluator performance plot
        evaluator_plot_path = output_path.parent / "evaluator_performance.png"
        plot_evaluator_performance(
            evaluator_avg, evaluator_plot_path, experiment_title=experiment_title
        )

        # Treatment difficulty
        f.write("TREATMENT DIFFICULTY (Average Accuracy Across All Evaluators)\n")
        f.write("-" * 70 + "\n")
        f.write("(Lower = harder to distinguish from self)\n\n")
        treatment_avg = (
            df_success.groupby("treatment")["accuracy"]
            .mean()
            .sort_values(ascending=True)
        )
        for model, acc in treatment_avg.items():
            f.write(f"  {model:30} {acc:.3f}\n")
        f.write("\n")

        # Pivot table dimensions
        f.write("COVERAGE\n")
        f.write("-" * 70 + "\n")
        f.write(f"Unique evaluators: {len(pivot.index)}\n")
        f.write(f"Unique treatments: {len(pivot.columns)}\n")
        f.write(
            f"Total possible comparisons: {len(pivot.index) * len(pivot.columns)}\n"
        )
        f.write(
            f"Diagonal (N/A): {len([m for m in pivot.index if m in pivot.columns])}\n"
        )
        valid_comparisons = pivot.notna().sum().sum()
        f.write(f"Valid evaluations: {int(valid_comparisons)}\n")
        f.write(
            f"Missing evaluations: {len(pivot.index) * len(pivot.columns) - int(valid_comparisons) - len([m for m in pivot.index if m in pivot.columns])}\n\n"
        )

        if p_values is not None:
            f.write("SIGNIFICANCE vs CHANCE (p=0.5)\n")
            f.write("-" * 70 + "\n")
            sig_mask = (p_values < 0.05) & p_values.notna()
            sig_count = int(sig_mask.sum().sum())
            f.write(f"Significant cells (p < 0.05): {sig_count}\n")
            # Identify strongest signals
            flattened = []
            for r in p_values.index:
                for c in p_values.columns:
                    pval = p_values.loc[r, c]
                    n = (
                        pivot_counts.loc[r, c]
                        if (r in pivot_counts.index and c in pivot_counts.columns)
                        else 0
                    )
                    acc = (
                        pivot.loc[r, c]
                        if (r in pivot.index and c in pivot.columns)
                        else pd.NA
                    )
                    if pd.notna(pval):
                        flattened.append((pval, r, c, acc, n))
            flattened.sort(key=lambda x: x[0])
            top = flattened[:5]
            f.write("Lowest p-values:\n")
            for pval, r, c, acc, n in top:
                acc_str = f"{acc:.3f}" if pd.notna(acc) else "NA"
                f.write(f"  {r} vs {c}: p={pval:.4g}, acc={acc_str}, n={int(n)}\n")
            f.write("\n")

        # Asymmetry analysis
        f.write("EVALUATOR vs EVALUATEE ASYMMETRY ANALYSIS\n")
        f.write("-" * 70 + "\n")
        f.write("This analysis distinguishes between:\n")
        f.write("  • Style-based recognition: Model has distinctive output style\n")
        f.write("  • Quality-based bias: Model biased by response quality/strength\n\n")

        f.write("ROW vs COLUMN MEANS (Per Model)\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Row Mean: Average accuracy when model is EVALUATOR (identifying itself)\n"
        )
        f.write(
            "Col Mean: Average accuracy when model is TREATMENT (others identifying it)\n"
        )
        f.write("Difference: Row Mean - Col Mean\n\n")
        f.write(
            f"{'Model':<30} {'Row Mean':<12} {'Col Mean':<12} {'Difference':<12} {'Interpretation':<50}\n"
        )
        f.write("-" * 116 + "\n")

        for model, row in comparison.iterrows():
            row_mean = row["row_mean"]
            col_mean = row["col_mean"]
            diff = row["difference"]

            if diff > 0.1:
                interpretation = "Distinctive style (better at self-ID)"
            elif diff < -0.1:
                interpretation = "Quality bias (others ID it easily)"
            else:
                interpretation = "Balanced"

            f.write(
                f"{model:<30} {row_mean:<12.3f} {col_mean:<12.3f} {diff:<12.3f} {interpretation:<50}\n"
            )
        f.write("\n")

        # Cell-wise asymmetry statistics
        f.write("CELL-WISE ASYMMETRY STATISTICS\n")
        f.write("-" * 70 + "\n")
        valid_asym = asymmetry_matrix[~asymmetry_matrix.isna()].values.flatten()
        if len(valid_asym) > 0:
            f.write(f"Mean absolute asymmetry: {abs(valid_asym).mean():.3f}\n")
            f.write(f"Max asymmetry: {valid_asym.max():.3f}\n")
            f.write(f"Min asymmetry: {valid_asym.min():.3f}\n")
            f.write(f"Std deviation: {valid_asym.std():.3f}\n\n")

            # Count significant asymmetries
            large_asym = abs(valid_asym) > 0.2
            f.write(
                f"Large asymmetries (|diff| > 0.2): {large_asym.sum()} / {len(valid_asym)} ({100*large_asym.sum()/len(valid_asym):.1f}%)\n"
            )
            f.write(
                f"Positive asymmetries: {(valid_asym > 0).sum()} ({100*(valid_asym > 0).sum()/len(valid_asym):.1f}%)\n"
            )
            f.write(
                f"Negative asymmetries: {(valid_asym < 0).sum()} ({100*(valid_asym < 0).sum()/len(valid_asym):.1f}%)\n\n"
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
    # OpenAI: Green/teal (OpenAI's brand color)
    # Anthropic: Orange (Claude's brand color)
    # Google: Yellow (Google's brand color)
    # Llama: Blue
    # Qwen: Purple
    # DeepSeek: Red

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
    ylabel: str = "Average Accuracy",
):
    """
    Create a bar plot showing evaluator performance averages.

    Args:
        evaluator_avg: Series with evaluator names as index and average values
        output_path: Path to save the plot
        experiment_title: Optional experiment title
        ylabel: Label for y-axis
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

    # Calculate padding for x-axis to prevent label overlap
    if len(evaluator_avg) > 0:
        min_val = evaluator_avg.min()
        max_val = evaluator_avg.max()
        val_range = max_val - min_val
        # Add 15% padding on each side
        padding = max(val_range * 0.15, 0.05)  # At least 0.05 padding
        x_min = max(0, min_val - padding)  # Don't go below 0 for accuracy values
        x_max = max_val + padding
        ax.set_xlim(x_min, x_max)

    # Labels and title
    ax.set_xlabel(ylabel, fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    title = "Evaluator Performance (Average Across All Treatments)"
    if experiment_title:
        title = f"{title}\n{experiment_title}"
    ax.set_title(title, fontsize=13, fontweight="bold", pad=20)

    # Add value labels on bars
    for i, (model, val) in enumerate(evaluator_avg.items()):
        ax.text(val, i, f" {val:.3f}", va="center", fontsize=9)

    # Add grid
    ax.grid(axis="x", alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  ✓ Saved evaluator performance plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analyze pairwise self-recognition experiment results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to directory containing eval logs (e.g., data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to experiment config.yaml (used to infer model_type if not explicitly provided)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["CoT", "DR"],
        default=None,
        help='Model set to use for ordering ("CoT" uses thinking subset; default DR uses standard list)',
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    model_type = args.model_type

    # If config_path not provided, attempt to derive it from results_dir
    derived_config_path = None
    if args.config_path is None:
        # Expect results_dir like: data/results/.../{experiment_name}
        experiment_name = results_dir.name
        candidate = Path("experiments") / experiment_name / "config.yaml"
        if candidate.exists():
            derived_config_path = candidate
    config_path = Path(args.config_path) if args.config_path else derived_config_path

    # If model_type not provided, try to infer from config_path (if resolved)
    if model_type is None and config_path:
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
            model_type = cfg.get("model_type")
        except Exception as e:
            print(
                f"Warning: Could not read model_type from config ({e}). Using default ordering."
            )

    if not results_dir.exists():
        print(f"Error: Directory not found: {results_dir}")
        return

    # Parse path to create matching analysis output path
    # Expected: data/results/{dataset_name}/{data_subset}/{experiment_name}
    # Output: data/analysis/{dataset_name}/{data_subset}/{experiment_name}
    parts = results_dir.parts
    if len(parts) >= 4 and parts[0] == "data" and parts[1] == "results":
        # Extract: dataset_name/data_subset/experiment_name
        relative_path = Path(*parts[2:])
        output_dir = Path("data/analysis") / relative_path
    else:
        # Fallback: use full path from results onwards
        output_dir = Path("data/analysis") / results_dir.name

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("PAIRWISE SELF-RECOGNITION ANALYSIS")
    print(f"{'='*70}")
    print(f"Results dir: {results_dir}")
    print(f"Output dir: {output_dir}")
    print(f"{'='*70}\n")

    # Load all evaluations
    df = load_all_evaluations(results_dir)

    if len(df) == 0:
        print("⚠ No evaluations found!")
        return

    # Create pivot table
    pivot, pivot_counts, pivot_correct = create_pivot_table(
        df, get_model_order(model_type)
    )

    if pivot.empty:
        print("⚠ No successful evaluations to analyze!")
        return

    # Compute significance vs chance (0.5)
    p_values = compute_significance_matrix(pivot_correct, pivot_counts)

    # Save pivot table
    pivot_path = output_dir / "accuracy_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"  ✓ Saved pivot table to: {pivot_path}\n")

    # Save counts and p-values
    counts_path = output_dir / "accuracy_counts.csv"
    pivot_counts.to_csv(counts_path)
    pvalues_path = output_dir / "pvalues_vs_chance.csv"
    p_values.to_csv(pvalues_path)
    print(f"  ✓ Saved counts to: {counts_path}")
    print(f"  ✓ Saved p-values to: {pvalues_path}\n")

    # Extract experiment name from path for title
    # Path format: .../dataset/subset/[NUM_]EXPERIMENT_CODE
    # e.g., .../pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr
    experiment_code = results_dir.name
    # Remove leading numbers and underscore if present (e.g., "11_UT_PW-Q_Rec_NPr" -> "UT_PW-Q_Rec_NPr")
    if "_" in experiment_code:
        parts = experiment_code.split("_", 1)
        if parts[0].isdigit():
            experiment_code = parts[1]

    # Look up full name from mapping
    experiment_mapping = get_experiment_name_mapping()
    experiment_title = experiment_mapping.get(experiment_code, experiment_code)

    # Generate heatmap
    heatmap_path = output_dir / "accuracy_heatmap.png"
    plot_heatmap(
        pivot,
        heatmap_path,
        experiment_title=experiment_title,
        p_values=p_values,
        alpha=0.05,
    )
    print()

    # Compute asymmetry analysis
    asymmetry_matrix, comparison, asymmetry_scores = compute_asymmetry_analysis(pivot)

    # Save asymmetry matrix
    asymmetry_path = output_dir / "asymmetry_matrix.csv"
    asymmetry_matrix.to_csv(asymmetry_path)
    print(f"  ✓ Saved asymmetry matrix to: {asymmetry_path}\n")

    # Save row vs column comparison
    comparison_path = output_dir / "row_vs_column_comparison.csv"
    comparison.to_csv(comparison_path)
    print(f"  ✓ Saved row vs column comparison to: {comparison_path}\n")

    # Generate asymmetry heatmap
    asymmetry_heatmap_path = output_dir / "asymmetry_heatmap.png"
    plot_asymmetry_heatmap(
        asymmetry_matrix, asymmetry_heatmap_path, experiment_title=experiment_title
    )
    print()

    # Generate row vs column comparison plot
    row_col_plot_path = output_dir / "row_vs_column_comparison.png"
    plot_row_vs_column_comparison(
        comparison, row_col_plot_path, experiment_title=experiment_title
    )
    print()

    # Generate summary stats
    summary_path = output_dir / "summary_stats.txt"
    generate_summary_stats(
        df,
        pivot,
        pivot_counts,
        p_values,
        comparison,
        asymmetry_matrix,
        summary_path,
        experiment_title=experiment_title,
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Accuracy Pivot Table")
    print(f"{'='*70}\n")
    print(pivot.round(3))
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • accuracy_pivot.csv: Raw accuracy data")
    print("  • accuracy_heatmap.png: Accuracy visualization")
    print("  • asymmetry_matrix.csv: Cell-wise asymmetry (pivot - pivot.T)")
    print("  • asymmetry_heatmap.png: Asymmetry visualization")
    print("  • row_vs_column_comparison.csv: Row vs column means per model")
    print("  • row_vs_column_comparison.png: Row vs column comparison plot")
    print("  • evaluator_performance.png: Evaluator performance bar chart")
    print("  • summary_stats.txt: Comprehensive statistics")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
