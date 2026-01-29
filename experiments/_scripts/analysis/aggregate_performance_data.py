#!/usr/bin/env python3
"""
Aggregate evaluator performance across multiple datasets.

This script loads evaluator_performance.csv files from multiple datasets and saves
merged CSV files for performance and deviation.

Usage:
    uv run experiments/_scripts/analysis/aggregate_performance_data.py \
        --performance_files data/analysis/wikisum/.../evaluator_performance.csv \
                              data/analysis/sharegpt/.../evaluator_performance.csv \
        --dataset_names "wikisum/training_set_1-20+test_set_1-30" "sharegpt/english_26+english2_74" \
        --model_names -set dr

Output:
    - data/analysis/_aggregated_data/{datetime}/
        - aggregated_performance.csv: Merged performance data
        - aggregated_deviation.csv: Merged deviation data
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory to path to import utils
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from utils import expand_model_names


def load_performance_data(
    performance_files: list[Path],
    dataset_names: list[str],
    column_name: str = "performance",
) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    """
    Load evaluator performance or deviation data from multiple CSV files and merge into a single DataFrame.
    Also loads and aggregates counts if available.

    Args:
        performance_files: List of paths to evaluator_performance.csv or evaluator_deviation.csv files
        dataset_names: List of dataset names/labels (same order as files)
        column_name: Name of the column to extract ('performance' or 'deviation')

    Returns:
        Tuple of:
        - DataFrame with models as index, datasets as columns, scores as values
        - DataFrame with counts (models as index, datasets as columns) or None if not available
    """
    merged_data = {}
    merged_counts = {}

    for file_path, dataset_name in zip(performance_files, dataset_names):
        if not file_path.exists():
            print(f"  ⚠ Warning: File not found: {file_path}")
            continue

        df = pd.read_csv(file_path, index_col=0)

        # Extract specified column
        if column_name in df.columns:
            merged_data[dataset_name] = df[column_name]
            
            # Extract counts if available
            if "n_samples" in df.columns:
                merged_counts[dataset_name] = df["n_samples"]
        else:
            print(f"  ⚠ Warning: '{column_name}' column not found in {file_path}")
            continue

    if not merged_data:
        raise ValueError(f"No valid {column_name} data loaded!")

    # Create DataFrame with datasets as columns, models as index
    result_df = pd.DataFrame(merged_data)

    # Fill missing values with 0 (models not present in a dataset)
    result_df = result_df.fillna(0)

    # Create counts DataFrame if available
    counts_df = None
    if merged_counts:
        counts_df = pd.DataFrame(merged_counts)
        counts_df = counts_df.fillna(0)

    return result_df, counts_df


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
        description="Aggregate evaluator performance across multiple datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--performance_files",
        type=str,
        nargs="+",
        required=True,
        help="Paths to evaluator_performance.csv files from different datasets",
    )
    parser.add_argument(
        "--dataset_names",
        type=str,
        nargs="+",
        required=True,
        help="Names/labels for each dataset (same order as performance_files)",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to include (filters and orders results). "
        "Supports -set notation (e.g., --model_names -set dr) or explicit names",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Optional: Output directory for aggregated files. If not provided, a timestamped directory is created.",
    )

    args = parser.parse_args()

    # Restore -set from placeholder
    args.model_names = [
        arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names
    ]

    # Expand model set references
    model_order = expand_model_names(args.model_names)
    print(f"Model filter/order: {', '.join(model_order)}\n")

    # Validate inputs
    if len(args.performance_files) != len(args.dataset_names):
        print("Error: Number of performance files must match number of dataset names")
        return

    performance_files = [Path(f) for f in args.performance_files]
    dataset_names = args.dataset_names

    # Validate all files exist
    for file_path in performance_files:
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return

    print(f"{'='*70}")
    print("PERFORMANCE AGGREGATION")
    print(f"{'='*70}")
    print(f"Datasets: {len(dataset_names)}")
    for name in dataset_names:
        print(f"  • {name}")
    print()

    # Extract experiment name from first file path if available
    experiment_name_full = ""
    first_file_parts = performance_files[0].parts
    if len(first_file_parts) >= 3:
        # Path: data/analysis/{dataset}/{experiment}/evaluator_performance/evaluator_performance.csv
        experiment_name_full = first_file_parts[
            -3
        ]  # Get full experiment name (e.g., "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")

    # Create output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Create output directory with experiment name and timestamp
        output_base = Path("data/analysis/_aggregated_data")
        if experiment_name_full:
            output_base = (
                output_base / experiment_name_full
            )  # Use full name for directory (e.g., "ICML_01_UT_PW-Q_Rec_NPr_FA_Inst")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = output_base / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # ============================================================================
    # Process Performance Data
    # ============================================================================

    print("Loading performance data...")
    try:
        df_perf, df_perf_counts = load_performance_data(
            performance_files, dataset_names, column_name="performance"
        )
        print(
            f"  ✓ Loaded data: {df_perf.shape[0]} models × {df_perf.shape[1]} datasets\n"
        )
        if df_perf_counts is not None:
            print(f"  ✓ Loaded counts data: {df_perf_counts.shape[0]} models × {df_perf_counts.shape[1]} datasets\n")
    except ValueError as e:
        print(f"  ⚠ Error loading performance data: {e}")
        df_perf = pd.DataFrame()
        df_perf_counts = None

    # Filter and order models
    if not df_perf.empty and model_order:
        available_models = [m for m in model_order if m in df_perf.index]
        if available_models:
            df_perf = df_perf.reindex(available_models)
        else:
            print("  ⚠ Warning: No models from filter list found in data")

    # Remove models with all zeros
    if not df_perf.empty:
        df_perf = df_perf.loc[(df_perf != 0).any(axis=1)]

    if not df_perf.empty:
        # Compute weighted average if counts available, otherwise simple average
        if df_perf_counts is not None:
            # Weighted average: sum(performance * n_samples) / sum(n_samples) per model
            weighted_perf = (df_perf * df_perf_counts).sum(axis=1) / df_perf_counts.sum(axis=1)
            total_counts = df_perf_counts.sum(axis=1)
            # Create DataFrame with weighted performance and total counts
            df_perf_weighted = pd.DataFrame({
                'performance': weighted_perf,
                'n_samples': total_counts
            })
            # Also save the per-dataset data
            df_perf.to_csv(output_dir / "aggregated_performance.csv")
            df_perf_weighted.to_csv(output_dir / "aggregated_performance_weighted.csv")
            df_perf_counts.to_csv(output_dir / "aggregated_counts.csv")
            print(f"  ✓ Saved aggregated performance data to: {output_dir / 'aggregated_performance.csv'}")
            print(f"  ✓ Saved weighted performance (with counts) to: {output_dir / 'aggregated_performance_weighted.csv'}")
            print(f"  ✓ Saved aggregated counts to: {output_dir / 'aggregated_counts.csv'}\n")
        else:
            # Simple average (no counts available)
            csv_path = output_dir / "aggregated_performance.csv"
            df_perf.to_csv(csv_path)
            print(f"  ✓ Saved aggregated performance data to: {csv_path}")
            print(f"  ⚠ No counts data available (weighted average not computed)\n")
    else:
        print("  ⚠ No valid performance data to save.\n")

    # ============================================================================
    # Process Deviation Data
    # ============================================================================

    # Build paths to deviation files (same structure as performance files)
    deviation_files = []
    for perf_file in performance_files:
        # Replace 'evaluator_performance.csv' with 'evaluator_deviation.csv'
        dev_file = perf_file.parent / "evaluator_deviation.csv"
        deviation_files.append(dev_file)

    # Check which deviation files exist
    existing_deviation_files = []
    existing_deviation_names = []
    for dev_file, dataset_name in zip(deviation_files, dataset_names):
        if dev_file.exists():
            existing_deviation_files.append(dev_file)
            existing_deviation_names.append(dataset_name)
        else:
            print(f"  ⚠ Warning: Deviation file not found: {dev_file}")

    df_dev = pd.DataFrame()
    df_dev_counts = None
    if existing_deviation_files:
        print("Loading deviation data...")
        try:
            df_dev, df_dev_counts = load_performance_data(
                existing_deviation_files,
                existing_deviation_names,
                column_name="deviation",
            )
            print(
                f"  ✓ Loaded data: {df_dev.shape[0]} models × {df_dev.shape[1]} datasets\n"
            )
            if df_dev_counts is not None:
                print(f"  ✓ Loaded counts data: {df_dev_counts.shape[0]} models × {df_dev_counts.shape[1]} datasets\n")

            # Filter and order models (use same order as performance)
            if model_order:
                available_models = [m for m in model_order if m in df_dev.index]
                if available_models:
                    df_dev = df_dev.reindex(available_models)

            # Remove models with all zeros
            df_dev = df_dev.loc[(df_dev != 0).any(axis=1)]

            if not df_dev.empty:
                # Save merged CSV
                csv_path = output_dir / "aggregated_deviation.csv"
                df_dev.to_csv(csv_path)
                print(f"  ✓ Saved aggregated deviation data to: {csv_path}\n")
            else:
                print("  ⚠ No valid deviation data to save.\n")

        except ValueError as e:
            print(f"  ⚠ Error loading deviation data: {e}")

    # ============================================================================
    # Summary
    # ============================================================================

    print(f"{'='*70}")
    print("DATA AGGREGATION COMPLETE")
    print(f"{'='*70}")
    print(f"Output directory: {output_dir}")
    if not df_perf.empty:
        print("  • aggregated_performance.csv")
        if df_perf_counts is not None:
            print("  • aggregated_performance_weighted.csv (with counts)")
            print("  • aggregated_counts.csv")
    if not df_dev.empty:
        print("  • aggregated_deviation.csv")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
