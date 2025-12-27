#!/usr/bin/env python3
"""
Analyze recognition disagreement between models in pairwise recognition experiments.

Takes a directory of eval logs from recognition tasks (e.g., Rec), aggregates
accuracy across all evaluations, and generates a disagreement heatmap showing
how much models disagree on self-recognition.

The disagreement score is computed as: abs(A_ij - (1 - A_ji))
where:
- A_ij = how often model i correctly recognizes its own output vs model j's
- A_ji = how often model j correctly recognizes its own output vs model i's
- 1 - A_ji = how often model j incorrectly recognizes (chooses model i's output)

Higher values indicate greater disagreement between the two models.

Usage:
    uv run experiments/scripts/analyze_recognition_disagreement.py \
        --results_dir data/results/pku_saferlhf/mismatch_1-20/17_UT_PW-Q_Rec_NPr_CoT

Output:
    - data/analysis/pku_saferlhf/mismatch_1-20/17_UT_PW-Q_Rec_NPr_CoT/
        - disagreement_matrix.csv: Disagreement scores
        - disagreement_heatmap.png: Visualization
        - summary_stats.txt: Overall statistics
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from inspect_ai.log import read_eval_log

from analysis_utils import (
    get_model_order,
    add_provider_boundaries,
    get_model_family_colors,
)


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

    For recognition tasks, accuracy represents how often the evaluator
    correctly recognizes its own output over the treatment output.

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
                        # Skip F (failed/malformed) - can't assess preference
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


def create_pivot_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table of accuracy by evaluator and treatment.

    For recognition tasks, accuracy = how often the evaluator correctly recognizes
    its own output over the treatment output.

    Args:
        df: DataFrame with evaluator, control, treatment, accuracy columns

    Returns:
        Pivot table with evaluators as rows, treatments as columns
    """
    # Filter to successful evaluations only
    df_success = df[df["status"] == "success"].copy()

    print(f"Creating pivot table from {len(df_success)} successful evaluations...")

    # Create pivot table
    # For recognition: evaluator identifies between control (self) and treatment
    # So the pivot should be: evaluator (rows) x treatment (columns)
    pivot = df_success.pivot_table(
        values="accuracy",
        index="evaluator",
        columns="treatment",
        aggfunc="mean",  # Average if multiple evals
    )

    # Reorder rows and columns according to canonical model order
    # Try CoT model order first, then fall back to regular order
    model_order_cot = get_model_order("cot")
    model_order_regular = get_model_order()

    # Check which order has more matches
    cot_matches = len([m for m in model_order_cot if m in pivot.index])
    regular_matches = len([m for m in model_order_regular if m in pivot.index])

    model_order = (
        model_order_cot if cot_matches > regular_matches else model_order_regular
    )

    # Filter to only models in the canonical order (strict filtering)
    row_order = [m for m in model_order if m in pivot.index]
    col_order = [m for m in model_order if m in pivot.columns]

    # Reindex to apply ordering (only includes models in canonical order)
    pivot = pivot.reindex(index=row_order, columns=col_order)

    return pivot


def compute_disagreement_matrix(pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Compute disagreement matrix between models on recognition.

    Disagreement score: abs(A_ij - (1 - A_ji))
    where:
    - A_ij = how often model i correctly recognizes its own output vs model j's
    - A_ji = how often model j correctly recognizes its own output vs model i's
    - 1 - A_ji = how often model j incorrectly recognizes (chooses model i's output)

    Higher values (closer to 1.0) indicate greater disagreement.

    Args:
        pivot: Pivot table with evaluators as rows, treatments as columns
               pivot[i, j] = how often model i recognizes its own output over model j's

    Returns:
        Disagreement matrix with same index and columns as pivot
    """
    disagreement = pd.DataFrame(index=pivot.index, columns=pivot.columns, dtype=float)

    for i, model_i in enumerate(pivot.index):
        for j, model_j in enumerate(pivot.columns):
            # Skip diagonal (model comparing with itself)
            if model_i == model_j:
                disagreement.loc[model_i, model_j] = float("nan")
                continue

            # Get A_ij: how often model i recognizes its own output over model j's
            A_ij = pivot.loc[model_i, model_j]

            # Get A_ji: how often model j recognizes its own output over model i's
            # Need to check if this exists (might be in transpose)
            if model_j in pivot.index and model_i in pivot.columns:
                A_ji = pivot.loc[model_j, model_i]
            else:
                # If reverse comparison doesn't exist, can't compute disagreement
                disagreement.loc[model_i, model_j] = float("nan")
                continue

            # Skip if either value is NaN
            if pd.isna(A_ij) or pd.isna(A_ji):
                disagreement.loc[model_i, model_j] = float("nan")
                continue

            # Compute disagreement: abs(A_ij - (1 - A_ji))
            # A_ij = P(model i recognizes its own output)
            # 1 - A_ji = P(model j chooses model i's output)
            # Disagreement measures how different these probabilities are
            disagreement_score = abs(A_ij - (1.0 - A_ji))
            disagreement.loc[model_i, model_j] = disagreement_score

    return disagreement


def plot_disagreement_heatmap(
    disagreement_matrix: pd.DataFrame, output_path: Path, experiment_title: str = ""
):
    """
    Create heatmap showing disagreement between models on recognition.

    Values show: abs(A_ij - (1 - A_ji))
    - Higher values (closer to 1.0): Models disagree on recognition
    - Lower values (closer to 0.0): Models agree on recognition
    - Range: 0.0 to 1.0

    Args:
        disagreement_matrix: Matrix of disagreement scores
        output_path: Path to save the heatmap
        experiment_title: Optional experiment name
    """
    print("Generating disagreement heatmap...")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))

    # Create mask for diagonal and NaN values (missing data)
    mask = pd.DataFrame(
        False, index=disagreement_matrix.index, columns=disagreement_matrix.columns
    )
    for model in disagreement_matrix.index:
        if model in disagreement_matrix.columns:
            # Mask diagonal
            mask.loc[model, model] = True
        # Mask NaN values (missing data for that pair)
        for col in disagreement_matrix.columns:
            if pd.isna(disagreement_matrix.loc[model, col]):
                mask.loc[model, col] = True

    # Create heatmap
    # Use RdYlGn: red (low disagreement) → yellow (medium) → green (high disagreement)
    sns.heatmap(
        disagreement_matrix,
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",  # Red (low) to green (high)
        center=0.5,
        vmin=0.0,
        vmax=1.0,
        cbar_kws={"label": "Disagreement Score"},
        mask=mask,
        linewidths=0.5,
        linecolor="gray",
        ax=ax,
    )

    # Fill diagonal and missing data cells with gray squares
    for i, model in enumerate(disagreement_matrix.index):
        for j, col in enumerate(disagreement_matrix.columns):
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
    add_provider_boundaries(ax, disagreement_matrix)

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")

    # Build title
    if experiment_title:
        title = f"Recognition Disagreement Matrix: {experiment_title}\n(Cell value = Disagreement score: |P(row recognizes self) - P(col chooses row)|)"
    else:
        title = "Recognition Disagreement Matrix\n(Cell value = Disagreement score: |P(row recognizes self) - P(col chooses row)|)"

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
    # Increase bottom margin to make room for note and rotated x-axis labels
    plt.subplots_adjust(bottom=0.20)

    # Add interpretation note
    fig.text(
        0.5,
        0.01,
        "Low (red): Models agree on recognition | "
        "High (green): Models disagree on recognition | "
        "Score = |P(row recognizes self) - P(col chooses row)|",
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
    print(f"  ✓ Saved disagreement heatmap to: {output_path}")

    plt.close()


def plot_evaluator_performance(
    evaluator_avg: pd.Series,
    output_path: Path,
    experiment_title: str = "",
    ylabel: str = "Average Score",
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
        x_min = max(0, min_val - padding)  # Don't go below 0 for agreement values
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


def generate_summary_stats(
    df: pd.DataFrame,
    pivot: pd.DataFrame,
    disagreement_matrix: pd.DataFrame,
    output_path: Path,
    experiment_title: str = "",
):
    """Generate and save summary statistics."""

    print("Generating summary statistics...")

    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("RECOGNITION DISAGREEMENT ANALYSIS\n")
        f.write("=" * 70 + "\n\n")

        # Overall stats
        f.write("OVERALL STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total evaluations: {len(df)}\n")
        f.write(f"  Successful: {(df['status'] == 'success').sum()}\n")
        f.write(f"  Failed: {(df['status'] == 'error').sum()}\n")
        f.write(f"  Cancelled: {(df['status'] == 'cancelled').sum()}\n")
        f.write(f"  Started (incomplete): {(df['status'] == 'started').sum()}\n\n")

        # Accuracy stats (recognition rates)
        df_success = df[df["status"] == "success"]
        if len(df_success) > 0 and df_success["accuracy"].notna().any():
            f.write("RECOGNITION STATISTICS\n")
            f.write("-" * 70 + "\n")
            f.write("(Accuracy = how often evaluator recognizes its own output)\n\n")
            f.write(f"Mean recognition rate: {df_success['accuracy'].mean():.3f}\n")
            f.write(f"Median recognition rate: {df_success['accuracy'].median():.3f}\n")
            f.write(f"Std deviation: {df_success['accuracy'].std():.3f}\n")
            f.write(f"Min recognition rate: {df_success['accuracy'].min():.3f}\n")
            f.write(f"Max recognition rate: {df_success['accuracy'].max():.3f}\n\n")

        # Disagreement stats
        f.write("DISAGREEMENT STATISTICS\n")
        f.write("-" * 70 + "\n")
        f.write("Disagreement score: |P(row recognizes self) - P(col chooses row)|\n")
        f.write("Higher values indicate greater disagreement on recognition.\n\n")

        valid_disagreement = disagreement_matrix[
            ~disagreement_matrix.isna()
        ].values.flatten()
        if len(valid_disagreement) > 0:
            f.write(f"Mean disagreement: {np.mean(valid_disagreement):.3f}\n")
            f.write(f"Median disagreement: {np.median(valid_disagreement):.3f}\n")
            f.write(f"Std deviation: {np.std(valid_disagreement):.3f}\n")
            f.write(f"Min disagreement: {np.min(valid_disagreement):.3f}\n")
            f.write(f"Max disagreement: {np.max(valid_disagreement):.3f}\n\n")

            # Count high/low disagreement
            high_disagreement = valid_disagreement > 0.5
            medium_disagreement = (valid_disagreement > 0.2) & (
                valid_disagreement <= 0.5
            )
            low_disagreement = valid_disagreement <= 0.2

            f.write(
                f"High disagreement (>0.5): {high_disagreement.sum()} / {len(valid_disagreement)} ({100*high_disagreement.sum()/len(valid_disagreement):.1f}%)\n"
            )
            f.write(
                f"Medium disagreement (0.2-0.5): {medium_disagreement.sum()} / {len(valid_disagreement)} ({100*medium_disagreement.sum()/len(valid_disagreement):.1f}%)\n"
            )
            f.write(
                f"Low disagreement (≤0.2): {low_disagreement.sum()} / {len(valid_disagreement)} ({100*low_disagreement.sum()/len(valid_disagreement):.1f}%)\n\n"
            )

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

        # Model-level disagreement summary
        f.write("MODEL-LEVEL DISAGREEMENT SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(
            "Average disagreement score for each model pair (row model vs column model)\n\n"
        )

        # Compute mean disagreement per row (how much each model disagrees with others)
        row_means = disagreement_matrix.mean(axis=1, skipna=True).sort_values(
            ascending=True  # Lower disagreement first
        )
        f.write("Average Disagreement When Model is Evaluator:\n")
        for model, mean_disagreement in row_means.items():
            f.write(f"  {model:<30} {mean_disagreement:.3f}\n")
        f.write("\n")

        # Generate evaluator performance plot
        evaluator_plot_path = (
            output_path.parent / "evaluator_disagreement_performance.png"
        )
        plot_evaluator_performance(
            row_means,
            evaluator_plot_path,
            experiment_title=experiment_title,
            ylabel="Average Disagreement Score (When Model is Evaluator)",
        )

        # Compute mean disagreement per column (how much others disagree with this model)
        col_means = disagreement_matrix.mean(axis=0, skipna=True).sort_values(
            ascending=True  # Lower disagreement first
        )
        f.write("Average Disagreement When Model is Treatment:\n")
        for model, mean_disagreement in col_means.items():
            f.write(f"  {model:<30} {mean_disagreement:.3f}\n")
        f.write("\n")

    print(f"  ✓ Saved summary to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Analyze recognition disagreement between models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        required=True,
        help="Path to directory containing eval logs (e.g., data/results/pku_saferlhf/mismatch_1-20/17_UT_PW-Q_Rec_NPr_CoT)",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)

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
    print("RECOGNITION DISAGREEMENT ANALYSIS")
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
    pivot = create_pivot_table(df)

    if pivot.empty:
        print("⚠ No successful evaluations to analyze!")
        return

    # Save pivot table
    pivot_path = output_dir / "recognition_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"  ✓ Saved pivot table to: {pivot_path}\n")

    # Extract experiment name from path for title
    # Path format: .../dataset/subset/[NUM_]EXPERIMENT_CODE
    experiment_code = results_dir.name
    # Remove leading numbers and underscore if present
    if "_" in experiment_code:
        parts = experiment_code.split("_", 1)
        if parts[0].isdigit():
            experiment_code = parts[1]

    experiment_title = experiment_code.replace("_", " ").title()

    # Compute disagreement matrix
    disagreement_matrix = compute_disagreement_matrix(pivot)

    # Save disagreement matrix
    disagreement_path = output_dir / "disagreement_matrix.csv"
    disagreement_matrix.to_csv(disagreement_path)
    print(f"  ✓ Saved disagreement matrix to: {disagreement_path}\n")

    # Generate disagreement heatmap
    disagreement_heatmap_path = output_dir / "disagreement_heatmap.png"
    plot_disagreement_heatmap(
        disagreement_matrix,
        disagreement_heatmap_path,
        experiment_title=experiment_title,
    )
    print()

    # Generate summary stats
    summary_path = output_dir / "disagreement_summary_stats.txt"
    generate_summary_stats(
        df, pivot, disagreement_matrix, summary_path, experiment_title=experiment_title
    )
    print()

    # Display preview
    print(f"{'='*70}")
    print("PREVIEW: Disagreement Matrix")
    print(f"{'='*70}\n")
    print(disagreement_matrix.round(3))
    print()

    print(f"{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    print("  • recognition_pivot.csv: Raw recognition data")
    print("  • disagreement_matrix.csv: Disagreement scores")
    print("  • disagreement_heatmap.png: Disagreement visualization")
    print("  • evaluator_disagreement_performance.png: Evaluator performance bar chart")
    print("  • disagreement_summary_stats.txt: Comprehensive statistics")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
