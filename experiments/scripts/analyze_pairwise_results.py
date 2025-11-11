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


def get_model_order() -> list[str]:
    """
    Define the canonical order for models in the pivot table and heatmap.

    Models are organized by company/provider, then ordered from weakest to strongest.

    Returns:
        List of model names in display order
    """
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
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b",
        "ll-3.1-70b",
        # Together AI - Qwen (weakest to strongest)
        "qwen-2.5-7b",
        "qwen-2.5-72b",
        "qwen-3.0-80b",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.0",
        "deepseek-3.1",
    ]


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

    Returns:
        Accuracy as a float (0.0 to 1.0), or None if not available
    """
    try:
        # Check if log has results
        if log.status != "success":
            return None

        if not log.results or not log.results.scores:
            return None

        # Look for accuracy metric
        for score in log.results.scores:
            if score.name == "acc":
                # The value is "C" or "I" per sample, need to aggregate
                # Actually, let's count from samples directly
                break

        # Count correct answers from samples
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
                        total_count += 1
                        if score.value["acc"] == "C":
                            correct_count += 1
                        break

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

    For pairwise comparisons, the evaluator judges between control (self) and treatment.
    Accuracy = how often the evaluator correctly identifies its own output.

    Args:
        df: DataFrame with evaluator, control, treatment, accuracy columns

    Returns:
        Pivot table with evaluators as rows, treatments as columns
    """
    # Filter to successful evaluations only
    df_success = df[df["status"] == "success"].copy()

    print(f"Creating pivot table from {len(df_success)} successful evaluations...")

    # Create pivot table
    # For pairwise: evaluator identifies between control (self) and treatment
    # So the pivot should be: evaluator (rows) x treatment (columns)
    pivot = df_success.pivot_table(
        values="accuracy",
        index="evaluator",
        columns="treatment",
        aggfunc="mean",  # Average if multiple evals
    )

    # Reorder rows and columns according to canonical model order
    model_order = get_model_order()

    # Filter to only models that exist in the data
    row_order = [m for m in model_order if m in pivot.index]
    col_order = [m for m in model_order if m in pivot.columns]

    # Reindex to apply ordering
    pivot = pivot.reindex(index=row_order, columns=col_order)

    return pivot


def plot_heatmap(pivot: pd.DataFrame, output_path: Path):
    """
    Create and save heatmap of self-recognition accuracy.

    Args:
        pivot: Pivot table with evaluators as rows, treatments as columns
        output_path: Path to save the heatmap image
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
    sns.heatmap(
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

    # Labels
    ax.set_xlabel("Comparison Model (Treatment)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Evaluator Model", fontsize=12, fontweight="bold")
    ax.set_title(
        "Self-Recognition Accuracy Matrix\n(How well each model identifies its own outputs vs. others)",
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


def generate_summary_stats(df: pd.DataFrame, pivot: pd.DataFrame, output_path: Path):
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

    print(f"  ✓ Saved summary to: {output_path}")


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
    pivot = create_pivot_table(df)

    if pivot.empty:
        print("⚠ No successful evaluations to analyze!")
        return

    # Save pivot table
    pivot_path = output_dir / "accuracy_pivot.csv"
    pivot.to_csv(pivot_path)
    print(f"  ✓ Saved pivot table to: {pivot_path}\n")

    # Generate heatmap
    heatmap_path = output_dir / "accuracy_heatmap.png"
    plot_heatmap(pivot, heatmap_path)
    print()

    # Generate summary stats
    summary_path = output_dir / "summary_stats.txt"
    generate_summary_stats(df, pivot, summary_path)
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
    print("  • accuracy_pivot.csv: Raw data")
    print("  • accuracy_heatmap.png: Visualization")
    print("  • summary_stats.txt: Statistics")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
