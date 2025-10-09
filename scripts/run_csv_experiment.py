#!/usr/bin/env python3
"""
CSV-Based 2T Experiment Runner

Simple cross-platform script to run 2T experiments from CSV files.
Supports both IDE mode (hardcoded path) and CLI mode (path as argument).

Usage:
    # From IDE (edit HARDCODED_CSV_PATH below)
    python run_csv_experiment.py

    # From CLI
    python run_csv_experiment.py data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv

    # With article text
    python run_csv_experiment.py path/to/data.csv --article-text "Article text..."

    # With custom model
    python run_csv_experiment.py path/to/data.csv --model anthropic/claude-sonnet-4-20250514
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ============================================================================
# IDE MODE: Hardcoded path for running from IDE
# ============================================================================
HARDCODED_CSV_PATH = (
    "data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv"
)


def run_csv_experiment(
    csv_path: str,
    model: str = "anthropic/claude-3-5-haiku-20241022",
    task: str = "two_turn_summary_recognition_csv",
    article_text: str | None = None,
    log_dir: str = "./logs",
) -> int:
    """
    Run a 2T experiment from a CSV file.

    Args:
        csv_path: Path to the CSV file
        model: Model identifier for inspect-ai
        task: Task name to run
        article_text: Optional article text
        log_dir: Directory for logs

    Returns:
        Return code from inspect eval
    """
    # Validate CSV exists
    csv_file = Path(csv_path)
    if not csv_file.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return 1

    # Build inspect eval command
    cmd = [
        "inspect",
        "eval",
        f"protocols/pairwise/tasks.py@{task}",
        "--model",
        model,
        "-T",
        f"csv_path={csv_path}",
        "--log-dir",
        log_dir,
    ]

    # Add article text if provided
    if article_text:
        cmd.extend(["-T", f"article_text={article_text}"])

    # Print info
    print("=" * 80)
    print("Running 2T Experiment from CSV")
    print("=" * 80)
    print(f"CSV: {csv_path}")
    print(f"Model: {model}")
    print(f"Task: {task}")
    if article_text:
        print(
            f"Article: {article_text[:100]}..."
            if len(article_text) > 100
            else f"Article: {article_text}"
        )
    print(f"\nCommand: {' '.join(cmd)}")
    print("=" * 80)
    print()

    # Run
    try:
        result = subprocess.run(cmd, check=True)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80 + "\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print(f"Error running experiment: {e}")
        print("=" * 80 + "\n")
        return e.returncode


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run 2T experiments from CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # IDE mode (uses hardcoded path)
    python run_csv_experiment.py

    # CLI mode
    python run_csv_experiment.py data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv

    # With article text
    python run_csv_experiment.py path/to/data.csv --article-text "Article about..."

    # With custom model
    python run_csv_experiment.py path/to/data.csv --model anthropic/claude-sonnet-4-20250514

    # Using uv
    uv run python run_csv_experiment.py path/to/data.csv
        """,
    )

    parser.add_argument(
        "csv_path",
        nargs="?",
        default=HARDCODED_CSV_PATH,
        help=f"Path to CSV file (default: {HARDCODED_CSV_PATH})",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="anthropic/claude-3-5-haiku-20241022",
        help="Model identifier (default: anthropic/claude-3-5-haiku-20241022)",
    )

    parser.add_argument(
        "--task",
        type=str,
        default="two_turn_summary_recognition_csv",
        help="Task name (default: two_turn_summary_recognition_csv)",
    )

    parser.add_argument(
        "--article-text",
        type=str,
        default=None,
        help="Optional article text (if all trials use same article)",
    )

    parser.add_argument(
        "--log-dir",
        type=str,
        default="./logs",
        help="Directory for logs (default: ./logs)",
    )

    args = parser.parse_args()

    # Run experiment
    return_code = run_csv_experiment(
        csv_path=args.csv_path,
        model=args.model,
        task=args.task,
        article_text=args.article_text,
        log_dir=args.log_dir,
    )

    sys.exit(return_code)


if __name__ == "__main__":
    main()
