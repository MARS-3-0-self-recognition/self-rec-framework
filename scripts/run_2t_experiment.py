#!/usr/bin/env python3
"""
Two-Turn (2T) Experiment Runner

This script runs 2T self-recognition experiments using the inspect-ai framework.
It integrates the forced_recog 2T experimental design with the new pairwise protocol structure.

Usage:
    # From IDE (hardcoded config path)
    python scripts/run_2t_experiment.py

    # From CLI (specify config path)
    python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml

    # Using uv
    uv run python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml
"""

import argparse
import sys
from pathlib import Path
import subprocess

# Import the hierarchical config loader
from protocols.pairwise.config.config_loader import load_experiment_config, ConfigLoader


def run_2t_experiment(config: dict) -> None:
    """
    Run a 2T experiment using inspect eval.

    Args:
        config: Configuration dictionary with experiment parameters (already merged with prompts)
    """
    # Extract parameters from config
    task_name = config.get("task_name", "two_turn_summary_recognition")
    model = config.get("model")
    model_name = config.get("model_name")
    alternative_model_name = config.get("alternative_model_name")
    dataset_name = config.get("dataset_name")
    model_generation_string = config.get("model_generation_string", "control")
    alternative_model_generation_string = config.get(
        "alternative_model_generation_string", "treatment"
    )

    # Extract system prompt using the config loader
    loader = ConfigLoader()
    system_prompt = loader.get_system_prompt(config)

    log_dir = config.get("log_dir", "./logs")

    # Validate required parameters
    if not model:
        raise ValueError(
            "Config must specify 'model' (e.g., 'anthropic/claude-3-5-sonnet-20241022')"
        )
    if not model_name:
        raise ValueError("Config must specify 'model_name' (e.g., 'claude-3-5-sonnet')")
    if not alternative_model_name:
        raise ValueError("Config must specify 'alternative_model_name' (e.g., 'gpt-4')")
    if not dataset_name:
        raise ValueError("Config must specify 'dataset_name' (e.g., 'wikisum')")

    # Build inspect eval command
    cmd = [
        "inspect",
        "eval",
        f"protocols/pairwise/tasks.py@{task_name}",
        "--model",
        model,
        "-T",
        f"model_name={model_name}",
        "-T",
        f"alternative_model_name={alternative_model_name}",
        "-T",
        f"dataset_name={dataset_name}",
        "-T",
        f"model_generation_string={model_generation_string}",
        "-T",
        f"alternative_model_generation_string={alternative_model_generation_string}",
        "--log-dir",
        log_dir,
    ]

    # Add system prompt (always specified from config/prompts)
    cmd.extend(["-T", f"system_prompt={system_prompt}"])

    # Add any additional inspect eval options from config
    if "max_samples" in config:
        cmd.extend(["--limit", str(config["max_samples"])])

    if "max_connections" in config:
        cmd.extend(["--max-connections", str(config["max_connections"])])

    # Print command for debugging
    print(f"\n{'='*80}")
    print(f"Running 2T Experiment: {task_name}")
    print(f"{'='*80}")
    print(f"Model: {model}")
    print(f"Model Name: {model_name}")
    print(f"Alternative Model: {alternative_model_name}")
    print(f"Dataset: {dataset_name}")
    print(
        f"Generation Strings: {model_generation_string} vs {alternative_model_generation_string}"
    )
    print(f"\nCommand: {' '.join(cmd)}")
    print(f"{'='*80}\n")

    # Run the command
    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n{'='*80}")
        print("Experiment completed successfully!")
        print(f"{'='*80}\n")
        return result.returncode
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*80}")
        print(f"Error running experiment: {e}")
        print(f"{'='*80}\n")
        return e.returncode


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Run 2T self-recognition experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default config (IDE mode)
    python scripts/run_2t_experiment.py

    # Run with custom config (CLI mode)
    python scripts/run_2t_experiment.py --config configs/my_2t_config.yaml

    # Using uv
    uv run python scripts/run_2t_experiment.py --config configs/my_2t_config.yaml
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="protocols/pairwise/config/two_turn/rec_config.yaml",  # Hardcoded default for IDE mode
        help="Path to YAML configuration file (default: protocols/pairwise/config/two_turn/rec_config.yaml)",
    )

    args = parser.parse_args()

    # Check if config file exists
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        print("Please create a config file or specify a different path.")
        sys.exit(1)

    # Load and run experiment
    try:
        # Load config using hierarchical loader (merges prompts automatically)
        config = load_experiment_config(args.config)
        return_code = run_2t_experiment(config)
        sys.exit(return_code)
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
