"""
Batch data generation script for multiple models.

This script wraps generate_data.py's run_generation function to enable
batch processing across multiple model names.
"""

import argparse
from dotenv import load_dotenv

from generate_data import run_generation


def run_batch_generation(
    model_names: list[str],
    dataset_path: str,
    dataset_config: str,
    overwrite: bool = False,
):
    """
    Generate data for multiple models using the same dataset and config.

    Args:
        model_names: List of model names to use for generation
        dataset_path: Path to input.json (e.g., 'data/wikisum/debug/input.json')
        dataset_config: Path to generation config YAML with temperature, treatments, etc.
        overwrite: If True, regenerate/reapply even if data exists
    """
    total_models = len(model_names)

    print(f"\n{'=' * 70}")
    print("BATCH DATA GENERATION")
    print(f"{'=' * 70}")
    print(f"Models to process: {total_models}")
    print(f"  {', '.join(model_names)}")
    print(f"Dataset path: {dataset_path}")
    print(f"Config: {dataset_config}")
    if overwrite:
        print("Mode: OVERWRITE (regenerating existing data)")
    else:
        print("Mode: SKIP (skipping existing data)")
    print(f"{'=' * 70}\n")

    # Track results
    successful = []
    failed = []

    for i, model_name in enumerate(model_names, 1):
        print(f"\n{'#' * 70}")
        print(f"Processing model {i}/{total_models}: {model_name}")
        print(f"{'#' * 70}")

        try:
            run_generation(
                model_name=model_name,
                dataset_path=dataset_path,
                dataset_config=dataset_config,
                overwrite=overwrite,
            )
            successful.append(model_name)
        except Exception as e:
            print(f"\n✗ ERROR processing {model_name}: {e}")
            failed.append((model_name, str(e)))

    # Summary
    print(f"\n{'=' * 70}")
    print("BATCH GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total models: {total_models}")
    print(f"Successful: {len(successful)}")
    if successful:
        for model in successful:
            print(f"  ✓ {model}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for model, error in failed:
            print(f"  ✗ {model}: {error}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Batch generate data using multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python experiments/scripts/generate_data_batch.py \\
    --model_names haiku-3-5 sonnet-3-7 gpt-4 \\
    --dataset_path=data/wikisum/debug/input/input.json \\
    --dataset_config=experiments/00_data_gen/configs/config.yaml
        """,
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names (e.g., 'haiku-3-5 sonnet-3-5 gpt-4')",
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to input.json (e.g., 'data/wikisum/debug/input/input.json')",
    )
    parser.add_argument(
        "--dataset_config",
        type=str,
        required=True,
        help="Path to generation config YAML (e.g., 'experiments/00_data_gen/configs/config.yaml')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing data files (default: skip existing files)",
    )

    args = parser.parse_args()

    run_batch_generation(
        model_names=args.model_names,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        overwrite=args.overwrite,
    )
