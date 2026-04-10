"""
Compare rank-distance vs recognition accuracy between two experiments.

Overlays scatter + trendline from both experiments on a single plot:
- Experiment 1: dark foreground
- Experiment 2: translucent background

Reads pre-computed rank_distance_data.csv files (output of srf-rank-distance).
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from self_rec_framework.scripts.utils import (
    calculate_binomial_ci,
    expand_model_names,
    save_figure_minimal_version,
    weighted_correlation,
    weighted_regression_with_ci,
)


def aggregate_cross_dataset(df):
    """
    Aggregate performance across datasets for each (evaluator, generator, distance).

    Uses weighted average by n_samples when available, matching
    rank_distance.py:443-456.
    """
    if "n_samples" in df.columns and df["n_samples"].notna().any():
        df = df.copy()
        df["weighted_perf"] = df["performance"] * df["n_samples"].fillna(0)
        aggregated = (
            df.groupby(["evaluator", "generator", "distance"])
            .agg({"weighted_perf": "sum", "n_samples": "sum"})
            .reset_index()
        )
        aggregated["performance"] = aggregated["weighted_perf"] / aggregated[
            "n_samples"
        ].replace(0, np.nan)
        aggregated = aggregated.drop(columns=["weighted_perf"])
    else:
        aggregated = (
            df.groupby(["evaluator", "generator", "distance"])["performance"]
            .mean()
            .reset_index()
        )
        aggregated["n_samples"] = None

    return aggregated


def compute_regression(agg_df):
    """
    Compute weighted regression and correlation on aggregated data.

    Returns dict with reg_result, correlation, weights, x_vals, y_vals, yerr,
    or None if insufficient data.
    """
    x_vals = agg_df["distance"].values
    y_vals = agg_df["performance"].values

    if len(x_vals) < 2:
        return None

    # x-axis range with padding
    x_min = x_vals.min()
    x_max = x_vals.max()
    x_range = x_max - x_min
    padding = max(x_range * 0.05, 1.0)
    x_min -= padding
    x_max += padding

    # Error bars
    has_n = "n_samples" in agg_df.columns and agg_df["n_samples"].notna().any()
    yerr = None
    weights = None

    if has_n:
        n_vals = agg_df["n_samples"].values
        errors = []
        for perf, n in zip(y_vals, n_vals):
            if pd.notna(n) and n > 0:
                _, _, se = calculate_binomial_ci(perf, n)
                errors.append(1.96 * se)
            else:
                errors.append(np.nan)
        yerr = np.array(errors)

        valid_n = ~np.isnan(n_vals) & (n_vals > 0)
        if np.any(valid_n):
            p_clipped = np.clip(y_vals, 0.05, 0.95)
            weights = np.where(valid_n, n_vals / (p_clipped * (1 - p_clipped)), 1.0)

    # Regression
    if weights is not None and not np.all(np.isnan(weights)):
        reg_result = weighted_regression_with_ci(
            x_vals, y_vals, weights=weights, x_min=x_min, x_max=x_max
        )
        if reg_result:
            correlation = weighted_correlation(x_vals, y_vals, weights)
        else:
            correlation = np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0
            reg_result = None
    else:
        correlation = np.corrcoef(x_vals, y_vals)[0, 1] if len(x_vals) > 1 else 0
        reg_result = weighted_regression_with_ci(
            x_vals, y_vals, x_min=x_min, x_max=x_max
        )

    return {
        "reg_result": reg_result,
        "correlation": correlation,
        "weights": weights,
        "x_vals": x_vals,
        "y_vals": y_vals,
        "yerr": yerr,
        "x_min": x_min,
        "x_max": x_max,
    }


def plot_rank_distance_contrast(
    agg1,
    agg2,
    reg1,
    reg2,
    exp1_name,
    exp2_name,
    output_path,
    metric_name="Recognition Accuracy",
    distance_type="rank",
):
    """
    Overlay rank-distance scatter + trendline from two experiments.

    Exp1 is plotted in the foreground (dark), exp2 in the background (translucent).
    """
    fig, ax = plt.subplots(figsize=(14, 10))

    # Determine shared x-axis range
    x_min = min(reg1["x_min"], reg2["x_min"])
    x_max = max(reg1["x_max"], reg2["x_max"])

    # --- Exp2: background layer ---
    bg_color = "#aaaaaa"
    bg_line_color = "#888888"

    if reg2["yerr"] is not None and not np.all(np.isnan(reg2["yerr"])):
        ax.errorbar(
            reg2["x_vals"],
            reg2["y_vals"],
            yerr=reg2["yerr"],
            fmt="none",
            ecolor=bg_color,
            alpha=0.2,
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            zorder=1,
        )

    slope2 = reg2["reg_result"]["slope"] if reg2["reg_result"] else np.nan
    r2 = reg2["correlation"]
    ax.scatter(
        reg2["x_vals"],
        reg2["y_vals"],
        marker="o",
        color=bg_color,
        s=80,
        alpha=0.3,
        edgecolors="gray",
        linewidths=0.5,
        label=f"{exp2_name} (slope={slope2:.4f}, r={r2:.2f})",
        zorder=2,
    )

    if reg2["reg_result"]:
        ax.fill_between(
            reg2["reg_result"]["x"],
            reg2["reg_result"]["ci_lower"],
            reg2["reg_result"]["ci_upper"],
            color=bg_line_color,
            alpha=0.07,
            zorder=3,
        )
        ax.plot(
            reg2["reg_result"]["x"],
            reg2["reg_result"]["y_pred"],
            linestyle="--",
            linewidth=2.5,
            alpha=0.4,
            color=bg_line_color,
            zorder=4,
        )

    # --- Exp1: foreground layer ---
    fg_color = "#1f77b4"
    fg_line_color = "black"

    if reg1["yerr"] is not None and not np.all(np.isnan(reg1["yerr"])):
        ax.errorbar(
            reg1["x_vals"],
            reg1["y_vals"],
            yerr=reg1["yerr"],
            fmt="none",
            ecolor=fg_color,
            alpha=0.4,
            capsize=2,
            capthick=0.5,
            linewidth=0.5,
            zorder=5,
        )

    slope1 = reg1["reg_result"]["slope"] if reg1["reg_result"] else np.nan
    r1 = reg1["correlation"]
    ax.scatter(
        reg1["x_vals"],
        reg1["y_vals"],
        marker="o",
        color=fg_color,
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidths=0.5,
        label=f"{exp1_name} (slope={slope1:.4f}, r={r1:.2f})",
        zorder=6,
    )

    if reg1["reg_result"]:
        ax.fill_between(
            reg1["reg_result"]["x"],
            reg1["reg_result"]["ci_lower"],
            reg1["reg_result"]["ci_upper"],
            color=fg_line_color,
            alpha=0.15,
            zorder=7,
        )
        ax.plot(
            reg1["reg_result"]["x"],
            reg1["reg_result"]["y_pred"],
            linestyle="-",
            linewidth=3,
            alpha=0.9,
            color=fg_line_color,
            zorder=8,
        )

    # --- Reference lines ---
    ax.axhline(
        y=0.5, color="#555555", linestyle="--", linewidth=1.0, alpha=0.8,
        label="Chance (0.5)",
    )
    ax.axvline(x=0, color="black", linestyle="-", linewidth=1, alpha=0.3,
               label="Equal Rank")

    # --- Slope annotation ---
    delta = slope1 - slope2 if not (np.isnan(slope1) or np.isnan(slope2)) else np.nan
    annotation = (
        f"{exp1_name} slope: {slope1:.4f}\n"
        f"{exp2_name} slope: {slope2:.4f}\n"
        f"\u0394slope: {delta:+.4f}"
    )
    ax.text(
        0.02, 0.02, annotation,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="bottom",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="gray"),
    )

    # --- Legend ---
    handles, labels = ax.get_legend_handles_labels()
    # Move Chance (0.5) to end
    chance_idx = next((i for i, l in enumerate(labels) if l == "Chance (0.5)"), None)
    if chance_idx is not None:
        h = handles.pop(chance_idx)
        l = labels.pop(chance_idx)
        handles.append(h)
        labels.append(l)
    # Remove Equal Rank from legend
    eq_idx = next((i for i, l in enumerate(labels) if l == "Equal Rank"), None)
    if eq_idx is not None:
        handles.pop(eq_idx)
        labels.pop(eq_idx)

    ax.legend(
        handles=handles,
        labels=labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=10,
        framealpha=0.9,
    )

    # --- Formatting ---
    if distance_type == "score":
        x_label = ("Score Distance (Evaluator Score - Generator Score)\n"
                    "Positive = Evaluator has higher score")
        dist_label = "Score Distance"
    else:
        x_label = ("Rank Distance (Evaluator Rank - Generator Rank)\n"
                    "Positive = Evaluator is worse ranked")
        dist_label = "Rank Distance"

    ax.set_xlabel(x_label, fontsize=12, fontweight="bold")
    ax.set_ylabel(
        f"Average {metric_name} (across datasets)",
        fontsize=12, fontweight="bold",
    )
    ax.set_title(
        f"{metric_name} vs {dist_label}\n{exp1_name} vs {exp2_name}",
        fontsize=14, fontweight="bold", pad=20,
    )
    ax.grid(alpha=0.3, linestyle="--")
    ax.set_axisbelow(True)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.tick_params(axis="both", labelsize=18)

    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"  \u2713 Saved contrast plot to: {output_path}")
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
        description="Compare rank-distance vs accuracy between two experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--exp1_file", type=str, required=True,
        help="Path to first experiment's rank_distance_data.csv",
    )
    parser.add_argument(
        "--exp2_file", type=str, required=True,
        help="Path to second experiment's rank_distance_data.csv",
    )
    parser.add_argument(
        "--exp1_name", type=str, required=True,
        help="Name of first experiment (for display)",
    )
    parser.add_argument(
        "--exp2_name", type=str, required=True,
        help="Name of second experiment (for display)",
    )
    parser.add_argument(
        "--model_names", type=str, nargs="+",
        help="Model names to filter evaluators. Supports -set notation.",
    )
    parser.add_argument(
        "--metric_name", type=str, default="Recognition Accuracy",
        help="Label for the y-axis metric (default: Recognition Accuracy)",
    )
    parser.add_argument(
        "--distance_type", type=str, choices=["rank", "score"], default="rank",
        help="Type of distance: 'rank' (lower rank = better) or 'score' (higher score = better)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory (default: data/analysis/_aggregated_data/{exp1}-vs-{exp2}/{timestamp}/)",
    )

    args = parser.parse_args()

    # Restore -set from placeholder
    if args.model_names:
        args.model_names = [
            arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names
        ]

    exp1_file = Path(args.exp1_file)
    exp2_file = Path(args.exp2_file)

    if not exp1_file.exists():
        print(f"Error: File not found: {exp1_file}")
        return
    if not exp2_file.exists():
        print(f"Error: File not found: {exp2_file}")
        return

    dist_type = args.distance_type
    dist_label = "SCORE" if dist_type == "score" else "RANK"

    print(f"{'='*70}")
    print(f"{dist_label} DISTANCE CONTRAST")
    print(f"{'='*70}")
    print(f"Experiment 1: {args.exp1_name} ({exp1_file})")
    print(f"Experiment 2: {args.exp2_name} ({exp2_file})")

    # Load data
    df1 = pd.read_csv(exp1_file)
    df2 = pd.read_csv(exp2_file)
    print(f"  Loaded exp1: {len(df1)} data points")
    print(f"  Loaded exp2: {len(df2)} data points")

    # Filter by model names (evaluator only, matching rank_distance.py)
    if args.model_names:
        model_order = expand_model_names(args.model_names)
        print(f"Model filter: {', '.join(model_order)}")
        df1 = df1[df1["evaluator"].isin(model_order)]
        df2 = df2[df2["evaluator"].isin(model_order)]
        print(f"  After filter — exp1: {len(df1)}, exp2: {len(df2)}")

    if df1.empty:
        print("Error: No data points remaining for experiment 1 after filtering.")
        return
    if df2.empty:
        print("Error: No data points remaining for experiment 2 after filtering.")
        return

    # Aggregate cross-dataset
    agg1 = aggregate_cross_dataset(df1)
    agg2 = aggregate_cross_dataset(df2)
    print(f"  Aggregated — exp1: {len(agg1)} points, exp2: {len(agg2)} points")

    # Compute regressions
    reg1 = compute_regression(agg1)
    reg2 = compute_regression(agg2)

    if reg1 is None or reg2 is None:
        print("Error: Insufficient data for regression in one or both experiments.")
        return

    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        comparison_name = f"{args.exp1_name}-vs-{args.exp2_name}"
        output_base = Path("data/analysis/_aggregated_data") / comparison_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_base / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    file_prefix = "score_distance" if dist_type == "score" else "rank_distance"
    output_path = output_dir / f"{file_prefix}_contrast.png"

    # Plot
    plot_rank_distance_contrast(
        agg1, agg2, reg1, reg2,
        args.exp1_name, args.exp2_name,
        output_path, args.metric_name,
        distance_type=dist_type,
    )

    # Save combined data CSV
    agg1_out = agg1.copy()
    agg1_out["experiment"] = args.exp1_name
    agg2_out = agg2.copy()
    agg2_out["experiment"] = args.exp2_name
    combined = pd.concat([agg1_out, agg2_out], ignore_index=True)
    csv_path = output_dir / f"{file_prefix}_contrast_data.csv"
    combined.to_csv(csv_path, index=False)
    print(f"  \u2713 Saved combined data to: {csv_path}")

    # Summary
    s1 = reg1["reg_result"]["slope"] if reg1["reg_result"] else np.nan
    s2 = reg2["reg_result"]["slope"] if reg2["reg_result"] else np.nan
    delta = s1 - s2 if not (np.isnan(s1) or np.isnan(s2)) else np.nan
    print(f"\n{'='*70}")
    print("SLOPE COMPARISON")
    print(f"{'='*70}")
    print(f"  {args.exp1_name}: slope={s1:.4f}, r={reg1['correlation']:.2f}")
    print(f"  {args.exp2_name}: slope={s2:.4f}, r={reg2['correlation']:.2f}")
    print(f"  \u0394slope: {delta:+.4f}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
