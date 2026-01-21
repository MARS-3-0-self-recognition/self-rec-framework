#!/usr/bin/env python3
"""
Test script to verify Google Gemini batch mode works with updated inspect-ai and google-genai.

This script generates data for a Gemini model with batch mode enabled to test
if the previous bugs have been fixed.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval

from src.inspect.tasks import generation
from src.inspect.config import create_generation_config
from src.helpers.utils import data_dir


def test_google_batch_mode(
    model_name: str = "gemini-2.0-flash",
    dataset_path: str = "data/input/wikisum/debug/input.json",
    dataset_config: str = "experiments/00_data_gen/configs/config.yaml",
    batch_size: int = 10,
):
    """
    Test Google Gemini batch mode with a small dataset.

    Args:
        model_name: Gemini model to test (default: gemini-2.0-flash)
        dataset_path: Path to small test dataset
        dataset_config: Path to generation config YAML
        batch_size: Batch size to use for testing
    """
    print(f"\n{'=' * 70}")
    print("TESTING GOOGLE GEMINI BATCH MODE")
    print(f"{'=' * 70}")
    print(f"Model: {model_name}")
    print(f"Dataset: {dataset_path}")
    print(f"Batch size: {batch_size}")
    print(f"{'=' * 70}\n")

    # Verify model is a Gemini model
    if "gemini" not in model_name.lower():
        print(f"⚠ Warning: Model '{model_name}' doesn't appear to be a Gemini model")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != "y":
            print("Test cancelled.")
            return

    # Parse dataset path
    dataset_path_obj = Path(dataset_path)
    parts = dataset_path_obj.parts

    if parts[0] == "data" and parts[1] == "input":
        dataset_name = parts[2]
        data_subset = parts[3]
    elif parts[0] == "input":
        dataset_name = parts[1]
        data_subset = parts[2]
    else:
        raise ValueError(
            f"Invalid dataset path format: {dataset_path}. "
            f"Expected: data/input/dataset_name/data_subset/input.json"
        )

    # Load generation config
    from generate_data import load_generation_config

    gen_config = load_generation_config(dataset_config)

    # Create ExperimentConfig for generation
    exp_config = create_generation_config(
        dataset_name=dataset_name,
        temperature=gen_config.get("temperature"),
        max_final_answer_tokens=gen_config.get("max_final_answer_tokens")
        or gen_config.get("max_tokens"),
        seed=gen_config.get("seed"),
    )

    # Create generation task
    print(f"Creating generation task for {model_name}...")
    task = generation(
        model_name=model_name,
        dataset_name=dataset_name,
        data_subset=data_subset,
        exp_config=exp_config,
    )

    # Set up log directory
    log_dir = (
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / "_test_batch_logs"
        / model_name
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"Running generation with batch mode enabled (batch_size={batch_size})...")
    print(f"{'=' * 70}\n")

    try:
        # Run with batch mode enabled
        eval_logs = eval(
            task,
            log_dir=str(log_dir),
            batch=batch_size,  # Use batch size instead of True for explicit testing
        )

        if eval_logs and len(eval_logs) > 0:
            print(f"\n{'=' * 70}")
            print("✅ SUCCESS: Batch mode completed without errors!")
            print(f"{'=' * 70}")
            print(f"Logs saved to: {log_dir}")
            print(f"Eval logs: {len(eval_logs)}")

            # Check if results look correct
            if hasattr(eval_logs[0], "eval") and hasattr(eval_logs[0].eval, "status"):
                status = eval_logs[0].eval.status
                print(f"Status: {status}")
                if status == "success":
                    print("\n✅ Batch mode test PASSED - Gemini batch mode appears to work!")
                else:
                    print(f"\n⚠ Status is '{status}' - may indicate an issue")
            else:
                print("\n✅ Batch mode test completed - check logs manually")

        else:
            print("\n⚠ No eval logs returned - check for errors above")

    except Exception as e:
        print(f"\n{'=' * 70}")
        print("❌ FAILURE: Batch mode encountered an error")
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        print(f"\nError type: {type(e).__name__}")
        import traceback

        print("\nTraceback:")
        traceback.print_exc()
        print(f"\n{'=' * 70}")
        print(
            "❌ Batch mode test FAILED - Gemini batch mode may still have issues"
        )
        print(f"{'=' * 70}\n")
        raise


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Test Google Gemini batch mode with updated packages",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default Gemini model and dataset
  uv run experiments/scripts/test_google_batch.py

  # Test with specific Gemini model
  uv run experiments/scripts/test_google_batch.py --model_name gemini-2.5-flash

  # Test with custom batch size
  uv run experiments/scripts/test_google_batch.py --batch_size 5
        """,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gemini-2.0-flash",
        help="Gemini model to test (default: gemini-2.0-flash)",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/input/wikisum/debug/input.json",
        help="Path to test dataset (default: wikisum debug)",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        default="experiments/00_data_gen/configs/config.yaml",
        help="Path to generation config YAML",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10,
        help="Batch size to use for testing (default: 10)",
    )

    args = parser.parse_args()

    test_google_batch_mode(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        batch_size=args.batch_size,
    )
