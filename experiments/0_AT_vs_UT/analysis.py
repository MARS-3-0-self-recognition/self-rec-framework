#!/usr/bin/env python3
"""
Analyze self-recognition experiment results and generate plots.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from inspect_ai.log import read_eval_log, EvalLog
from src.helpers.utils import data_dir


# ==============================================================================
# Data structures
# ==============================================================================


@dataclass
class PairwiseScore:
    """Scores for a single (generator, alternative) pair."""

    generator: str
    alternative: str
    at_score: Optional[float]  # Conversational (assistant tags)
    ut_score: Optional[float]  # Comparison (user tags)


# ==============================================================================
# Data loading
# ==============================================================================


def get_judge_log_dir(
    dataset_name: str,
    task_name: str,
    model_name: str,
    alt_model_name: str,
    generation_string: str,
) -> Path:
    """Get path to evaluation logs for a specific judging task."""
    return (
        data_dir()
        / dataset_name
        / "judge_logs"
        / task_name
        / f"{model_name}_vs_{alt_model_name}_{generation_string}"
    )


def read_eval_log_from_dir(log_dir: Path) -> Optional[EvalLog]:
    """Read eval log from directory. Asserts exactly one .eval file exists."""
    eval_files = list(log_dir.glob("*.eval"))

    if not eval_files:
        return None

    assert (
        len(eval_files) == 1
    ), f"Expected 1 eval file in {log_dir}, found {len(eval_files)}"

    try:
        return read_eval_log(eval_files[0])
    except Exception as e:
        print(f"Warning: Could not read {eval_files[0]}: {e}")
        return None


def extract_mean_accuracy(log: EvalLog) -> Optional[float]:
    """Extract mean accuracy from an EvalLog."""
    if not log.results or not log.results.scores:
        return None

    metrics = log.results.scores[0].metrics

    if "mean" in metrics:
        return metrics["mean"].value
    elif "accuracy" in metrics:
        return metrics["accuracy"].value

    return None


def load_all_pairwise_scores(
    model_names: list[str],
    generation_string: str,
    dataset_name: str,
) -> list[PairwiseScore]:
    """Load evaluation scores for all model pairs (N × N-1 ordered pairs)."""
    results = []
    pairs = [(m1, m2) for m1, m2 in permutations(model_names, 2)]

    for generator, alternative in pairs:
        # Load conversational (AT) score
        at_log_dir = get_judge_log_dir(
            dataset_name, "conversational", generator, alternative, generation_string
        )
        at_log = read_eval_log_from_dir(at_log_dir)
        at_score = extract_mean_accuracy(at_log) if at_log else None

        # Load comparison (UT) score
        ut_log_dir = get_judge_log_dir(
            dataset_name, "comparison", generator, alternative, generation_string
        )
        ut_log = read_eval_log_from_dir(ut_log_dir)
        ut_score = extract_mean_accuracy(ut_log) if ut_log else None

        if at_score is not None or ut_score is not None:
            results.append(
                PairwiseScore(
                    generator=generator,
                    alternative=alternative,
                    at_score=at_score,
                    ut_score=ut_score,
                )
            )

    return results


def scores_to_dataframe(scores: list[PairwiseScore]) -> pd.DataFrame:
    """Convert list of PairwiseScore objects to DataFrame."""
    return pd.DataFrame(
        [
            {
                "generator": s.generator,
                "alternative": s.alternative,
                "AT_score": s.at_score,
                "UT_score": s.ut_score,
            }
            for s in scores
        ]
    )


def assign_model_colors(model_names: list[str]) -> dict[str, tuple]:
    """Assign distinct colors using colorblind-friendly palette."""
    palette = sns.color_palette("colorblind", n_colors=len(model_names))
    return {model: palette[i] for i, model in enumerate(model_names)}


# ==============================================================================
# Plotting
# ==============================================================================


def create_scatter_plot(
    scores_df: pd.DataFrame,
    model_colors: dict[str, tuple],
    output_path: Path,
    add_labels: bool = True,
) -> None:
    """
    Scatter plot: AT (x) vs UT (y).
    Lines connect same generator (colored by generator).
    Points colored by alternative.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    # Lines connecting same generator
    for generator in scores_df["generator"].unique():
        gen_data = scores_df[scores_df["generator"] == generator].sort_values(
            "alternative"
        )
        ax.plot(
            gen_data["AT_score"],
            gen_data["UT_score"],
            color=model_colors[generator],
            alpha=0.5,
            linewidth=2,
            zorder=1,
        )

    # Points colored by alternative
    for _, row in scores_df.iterrows():
        ax.scatter(
            row["AT_score"],
            row["UT_score"],
            color=model_colors[row["alternative"]],
            s=100,
            edgecolors="black",
            linewidth=1,
            zorder=2,
        )

        if add_labels:
            label = f"{row['generator'][:3]}→{row['alternative'][:3]}"
            ax.text(
                row["AT_score"],
                row["UT_score"],
                label,
                fontsize=8,
                ha="right",
                va="bottom",
            )

    # Diagonal reference
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=1, zorder=0)

    ax.set_xlabel("Self-Recognition Score (AT - Conversational)", fontsize=12)
    ax.set_ylabel("Self-Recognition Score (UT - Comparison)", fontsize=12)
    ax.set_title("Self-Recognition: Conversational vs Comparison", fontsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Legend
    handles = [
        plt.Line2D([0], [0], color=c, linewidth=2, label=f"{m} (gen)")
        for m, c in model_colors.items()
    ]
    handles += [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=c,
            markersize=8,
            markeredgecolor="black",
            label=f"{m} (alt)",
        )
        for m, c in model_colors.items()
    ]

    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1), fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


def create_heatmap(
    scores_df: pd.DataFrame,
    score_column: str,
    title: str,
    output_path: Path,
) -> None:
    """
    Heatmap: generators (x) vs alternatives (y).
    Red=0, Blue=1, Gray=NaN/diagonal.
    """
    all_models = sorted(set(scores_df["generator"]) | set(scores_df["alternative"]))

    # Build matrix
    matrix = pd.DataFrame(index=all_models, columns=all_models, dtype=float)
    for _, row in scores_df.iterrows():
        if pd.notna(row[score_column]):
            matrix.loc[row["alternative"], row["generator"]] = row[score_column]

    # Blank diagonal
    for model in all_models:
        matrix.loc[model, model] = np.nan

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = sns.color_palette("RdBu", as_cmap=True)
    cmap.set_bad(color="lightgray")

    sns.heatmap(
        matrix.astype(float),
        annot=True,
        fmt=".3f",
        cmap=cmap,
        vmin=0,
        vmax=1,
        center=0.5,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Self-Recognition Score"},
        ax=ax,
    )

    ax.set_xlabel("Generator Model", fontsize=12)
    ax.set_ylabel("Alternative Model", fontsize=12)
    ax.set_title(title, fontsize=14)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"✓ Saved: {output_path}")


# ==============================================================================
# Main
# ==============================================================================


def main():
    parser = argparse.ArgumentParser(description="Analyze self-recognition results")
    parser.add_argument("--experiment_id", type=str, required=True)
    parser.add_argument("--generation_string", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True)
    parser.add_argument("--model_names", type=str, nargs="+", required=True)
    parser.add_argument("--no_labels", action="store_true")

    args = parser.parse_args()

    output_dir = (
        Path(__file__).parent
        / args.experiment_id
        / "plots"
        / args.dataset_name
        / args.generation_string
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Experiment: {args.experiment_id}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Generation: {args.generation_string}")
    print(f"Models: {', '.join(args.model_names)}")
    print(f"{'='*60}\n")

    # Load data
    pairwise_scores = load_all_pairwise_scores(
        args.model_names,
        args.generation_string,
        args.dataset_name,
    )

    if not pairwise_scores:
        print("✗ No results found!")
        return

    scores_df = scores_to_dataframe(pairwise_scores)
    complete_df = scores_df.dropna(subset=["AT_score", "UT_score"])

    if complete_df.empty:
        print("✗ No pairs have both AT and UT scores!")
        return

    print(f"Loaded {len(complete_df)} complete pairs\n")

    # Generate plots
    model_colors = assign_model_colors(args.model_names)

    create_scatter_plot(
        complete_df,
        model_colors,
        output_dir / "scatter_AT_vs_UT.png",
        add_labels=not args.no_labels,
    )

    create_heatmap(
        scores_df[scores_df["AT_score"].notna()],
        "AT_score",
        "Self-Recognition Scores (AT - Conversational)",
        output_dir / "heatmap_AT.png",
    )

    create_heatmap(
        scores_df[scores_df["UT_score"].notna()],
        "UT_score",
        "Self-Recognition Scores (UT - Comparison)",
        output_dir / "heatmap_UT.png",
    )

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
