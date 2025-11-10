"""
Sweep data generation script for multiple models.

This script uses Inspect AI's multi-model parallelism to generate data for
multiple models simultaneously, then post-processes the results into separate
model directories.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval

from src.inspect.tasks import generation
from src.inspect.config import create_generation_config
from src.helpers.utils import data_dir, save_json
from src.helpers.model_names import inspect_model_name, SHORT_MODEL_NAMES
from generate_data import load_generation_config, apply_treatments


def run_sweep_generation(
    model_names: list[str],
    dataset_path: str,
    dataset_config: str,
    overwrite: bool = False,
    batch: bool | int | str = False,
):
    """
    Generate data for multiple models using Inspect AI's multi-model parallelism.

    Uses a single eval() call with multiple models, then post-processes results
    to separate data by model and apply treatments.

    Args:
        model_names: List of model names to use for generation
        dataset_path: Path to input.json (e.g., 'data/wikisum/debug/input.json')
        dataset_config: Path to generation config YAML with temperature, treatments, etc.
        overwrite: If True, regenerate/reapply even if data exists
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
    """
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

    total_models = len(model_names)

    print(f"\n{'=' * 70}")
    print("SWEEP DATA GENERATION (Multi-Model Parallel)")
    print(f"{'=' * 70}")
    print(f"Dataset: {dataset_name}/{data_subset}")
    print(f"Models to process: {total_models}")
    print(f"  {', '.join(model_names)}")
    print(f"Config: {dataset_config}")
    if batch:
        print("Batch mode: ENABLED")
    if overwrite:
        print("Mode: OVERWRITE (regenerating existing data)")
    else:
        print("Mode: SKIP (checking existing data)")
    print(f"{'=' * 70}\n")

    # Load generation config
    gen_config = load_generation_config(dataset_config)

    # Create ExperimentConfig for generation
    exp_config = create_generation_config(
        dataset_name=dataset_name,
        temperature=gen_config.get("temperature"),
        max_tokens=gen_config.get("max_tokens"),
        seed=gen_config.get("seed"),
    )

    # Check which models need generation (skip if exists and not overwrite)
    models_to_generate = []
    skipped_models = []

    for model_name in model_names:
        output_path = (
            data_dir() / "input" / dataset_name / data_subset / model_name / "data.json"
        )
        if output_path.exists() and not overwrite:
            print(f"  ⊘ {model_name}: data already exists, skipping")
            skipped_models.append(model_name)
        else:
            models_to_generate.append(model_name)
            if output_path.exists():
                print(f"  → {model_name}: will overwrite existing data")
            else:
                print(f"  ✓ {model_name}: will generate")

    if not models_to_generate:
        print("\n⊘ All models already have data, nothing to generate")
        return

    print(f"\n{'=' * 70}")
    print(f"Generating data for {len(models_to_generate)} models in parallel...")
    print(f"{'=' * 70}\n")

    # Convert short names to inspect model names
    inspect_models = [inspect_model_name(name) for name in models_to_generate]

    # Create generation task (same for all models)
    task = generation(
        model_name=models_to_generate[0],  # Placeholder, will be overridden by eval
        dataset_name=dataset_name,
        data_subset=data_subset,
        exp_config=exp_config,
    )

    # Set up shared log directory for generation
    log_dir = (
        data_dir() / "input" / dataset_name / data_subset / "_generation_logs_sweep"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run generation with multiple models in parallel
    print("Running multi-model generation (this may take a while with batch mode)...\n")
    eval_logs = eval(task, model=inspect_models, log_dir=str(log_dir), batch=batch)

    print(f"\n✓ Generation complete! Processing {len(eval_logs)} model outputs...\n")

    # Post-process: Separate outputs by model and save to individual directories
    successful = []
    failed = []

    for eval_log in eval_logs:
        try:
            # Get short model name from eval log
            full_model_name = eval_log.eval.model
            model_name = SHORT_MODEL_NAMES.get(full_model_name, full_model_name)

            # If not in our mapping, try to extract from model_names
            if model_name == full_model_name:
                # Try to match by finding which short name maps to this full name
                for short_name in models_to_generate:
                    if inspect_model_name(short_name) == full_model_name:
                        model_name = short_name
                        break

            print(f"  Processing outputs for {model_name}...")

            # Extract outputs
            data_dict = {}
            for sample in eval_log.samples:
                data_dict[sample.id] = sample.output.completion

            # Save to model-specific directory
            output_path = (
                data_dir()
                / "input"
                / dataset_name
                / data_subset
                / model_name
                / "data.json"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            save_json(data_dict, output_path)

            print(f"  ✓ {model_name}: Saved {len(data_dict)} samples to {output_path}")
            successful.append(model_name)

        except Exception as e:
            print(f"  ✗ Error processing {model_name}: {e}")
            failed.append((model_name, str(e)))

    # Apply treatments to each model's data
    print(f"\n{'=' * 70}")
    print("Applying treatments...")
    print(f"{'=' * 70}\n")

    for model_name in successful:
        base_data_path = (
            data_dir() / "input" / dataset_name / data_subset / model_name / "data.json"
        )

        try:
            apply_treatments(
                base_data_path=base_data_path,
                dataset_name=dataset_name,
                data_subset=data_subset,
                model_name=model_name,
                gen_config=gen_config,
                overwrite=overwrite,
            )
        except Exception as e:
            print(f"  ✗ Error applying treatments for {model_name}: {e}")
            failed.append((model_name, f"Treatment error: {str(e)}"))

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP GENERATION SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total models requested: {total_models}")
    print(f"Skipped (already exist): {len(skipped_models)}")
    print(f"Generated: {len(successful)}")

    if successful:
        print("\nSuccessful:")
        for model in successful:
            print(f"  ✓ {model}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for model, error in failed[:10]:
            error_msg = error[:80] if len(error) > 80 else error
            print(f"  ✗ {model}: {error_msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Sweep generate data using multiple models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python experiments/scripts/generate_data_sweep.py \\
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
    parser.add_argument(
        "--batch",
        nargs="?",
        const=True,
        default=False,
        help="Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI). "
        "Usage: --batch (default config), --batch 1000 (batch size), --batch config.yaml (config file)",
    )

    args = parser.parse_args()

    # Parse batch argument
    batch_value = args.batch
    if batch_value is not False:
        # Try to convert to int if it's a number string
        if isinstance(batch_value, str) and batch_value.isdigit():
            batch_value = int(batch_value)

    run_sweep_generation(
        model_names=args.model_names,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        overwrite=args.overwrite,
        batch=batch_value,
    )
