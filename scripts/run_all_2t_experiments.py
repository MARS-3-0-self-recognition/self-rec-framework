#!/usr/bin/env python3
"""
Batch Runner for Multiple 2T Experiments

This script runs multiple 2T experiments in parallel or sequence based on a list
of configuration files. It's analogous to run_all_experiments_parallel.py from forced_recog.

Usage:
    # Run all experiments sequentially
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments

    # Run with parallel execution
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments --parallel --workers 4

    # Dry run to see what would be executed
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments --dry-run
"""

import argparse
import sys
from pathlib import Path
import subprocess
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import time


def find_config_files(config_dir: Path) -> List[Path]:
    """Find all YAML config files in the specified directory."""
    if not config_dir.exists():
        print(f"Error: Config directory not found: {config_dir}")
        return []

    # Find all .yaml and .yml files
    config_files = list(config_dir.glob("*.yaml")) + list(config_dir.glob("*.yml"))
    return sorted(config_files)


def run_single_experiment(config_file: Path, dry_run: bool = False) -> Dict:
    """
    Run a single experiment from a config file.

    Args:
        config_file: Path to the config file
        dry_run: If True, only print what would be executed

    Returns:
        Dictionary with experiment results
    """
    print(f"\n{'='*80}")
    print(f"Experiment: {config_file.name}")
    print(f"{'='*80}")

    if dry_run:
        print(
            f"[DRY RUN] Would execute: python scripts/run_2t_experiment.py --config {config_file}"
        )
        return {"config_file": str(config_file), "status": "dry_run", "return_code": 0}

    # Run the experiment
    cmd = [sys.executable, "scripts/run_2t_experiment.py", "--config", str(config_file)]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False, text=True)
        elapsed_time = time.time() - start_time

        return {
            "config_file": str(config_file),
            "status": "success",
            "return_code": result.returncode,
            "elapsed_time": elapsed_time,
        }
    except subprocess.CalledProcessError as e:
        elapsed_time = time.time() - start_time
        print(f"Error running experiment {config_file.name}: {e}")

        return {
            "config_file": str(config_file),
            "status": "failed",
            "return_code": e.returncode,
            "elapsed_time": elapsed_time,
            "error": str(e),
        }


def run_experiments_sequential(
    config_files: List[Path], dry_run: bool = False
) -> List[Dict]:
    """Run experiments one after another."""
    results = []

    for config_file in config_files:
        result = run_single_experiment(config_file, dry_run)
        results.append(result)

    return results


def run_experiments_parallel(
    config_files: List[Path], max_workers: int = 4, dry_run: bool = False
) -> List[Dict]:
    """Run experiments in parallel using ThreadPoolExecutor."""
    results = []

    print(
        f"\nRunning {len(config_files)} experiments with {max_workers} parallel workers..."
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_config = {
            executor.submit(run_single_experiment, config_file, dry_run): config_file
            for config_file in config_files
        }

        # Collect results as they complete
        for future in as_completed(future_to_config):
            config_file = future_to_config[future]
            try:
                result = future.result()
                results.append(result)

                status_emoji = "✓" if result["status"] == "success" else "✗"
                print(f"{status_emoji} Completed: {config_file.name}")
            except Exception as e:
                print(f"✗ Exception for {config_file.name}: {e}")
                results.append(
                    {
                        "config_file": str(config_file),
                        "status": "exception",
                        "error": str(e),
                    }
                )

    return results


def print_summary(results: List[Dict]) -> None:
    """Print a summary of all experiment results."""
    print(f"\n{'='*80}")
    print("BATCH EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    exceptions = [r for r in results if r["status"] == "exception"]
    dry_runs = [r for r in results if r["status"] == "dry_run"]

    print(f"Total experiments: {len(results)}")
    print(f"  ✓ Successful: {len(successful)}")
    print(f"  ✗ Failed: {len(failed)}")
    print(f"  ⚠ Exceptions: {len(exceptions)}")

    if dry_runs:
        print(f"  ℹ Dry runs: {len(dry_runs)}")

    if failed:
        print("\nFailed experiments:")
        for result in failed:
            print(
                f"  - {Path(result['config_file']).name}: return code {result['return_code']}"
            )

    if exceptions:
        print("\nExceptions:")
        for result in exceptions:
            print(
                f"  - {Path(result['config_file']).name}: {result.get('error', 'Unknown error')}"
            )

    if successful and not dry_runs:
        total_time = sum(r.get("elapsed_time", 0) for r in successful)
        avg_time = total_time / len(successful) if successful else 0
        print("\nTiming:")
        print(f"  Total time: {total_time:.1f}s")
        print(f"  Average per experiment: {avg_time:.1f}s")

    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run multiple 2T experiments from config files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run all configs in a directory sequentially
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments

    # Run in parallel with 4 workers
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments --parallel --workers 4

    # Dry run to see what would execute
    python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments --dry-run
        """,
    )

    parser.add_argument(
        "--config-dir",
        type=str,
        default="protocols/pairwise/config/two_turn",
        help="Directory containing experiment config files (default: protocols/pairwise/config/two_turn)",
    )

    parser.add_argument(
        "--parallel", action="store_true", help="Run experiments in parallel"
    )

    parser.add_argument(
        "--workers", type=int, default=4, help="Number of parallel workers (default: 4)"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    args = parser.parse_args()

    # Find config files
    config_dir = Path(args.config_dir)
    config_files = find_config_files(config_dir)

    if not config_files:
        print(f"No config files found in {config_dir}")
        print("Please create YAML config files in this directory.")
        sys.exit(1)

    print(f"Found {len(config_files)} config files in {config_dir}")
    for cf in config_files:
        print(f"  - {cf.name}")

    # Run experiments
    if args.parallel and not args.dry_run:
        results = run_experiments_parallel(config_files, args.workers, args.dry_run)
    else:
        results = run_experiments_sequential(config_files, args.dry_run)

    # Print summary
    print_summary(results)

    # Exit with error code if any experiments failed
    failed_count = len([r for r in results if r["status"] in ["failed", "exception"]])
    sys.exit(1 if failed_count > 0 else 0)


if __name__ == "__main__":
    main()
