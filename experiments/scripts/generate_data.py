from inspect_ai import eval
from inspect_ai.log import EvalLog
from pathlib import Path
import yaml

from src.inspect.tasks import generation
from src.inspect.config import create_generation_config, ExperimentConfig
from src.helpers.utils import data_dir, save_json
from src.data_generation.procedural_editing.treatment import apply_treatment


def load_generation_config(config_path: str) -> dict:
    """Load simple generation config (temperature, max_tokens, treatments, etc.)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def construct_data_dict(eval_log: EvalLog) -> dict[str, str]:
    """Parse the model outputs and UUIDs into a dict."""
    data_dict = {}
    for sample in eval_log.samples:
        data_dict[sample.id] = sample.output.completion
    return data_dict


def _generate_base_data(
    model_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    overwrite: bool = False,
) -> Path:
    """
    Generate base data using a model (no treatments applied).

    Uses the generation task from src.inspect.tasks.

    Args:
        model_name: Model to use for generation
        dataset_name: Dataset name
        data_subset: Data subset directory
        exp_config: ExperimentConfig with prompts and generation parameters
        overwrite: If True, regenerate even if data exists

    Returns:
        Path to the generated data.json file
    """
    treatment_name = model_name
    output_path = (
        data_dir() / "input" / dataset_name / data_subset / treatment_name / "data.json"
    )

    # Check if already exists
    if output_path.exists() and not overwrite:
        print(f"  ✓ {treatment_name}: data already exists, skipping generation")
        return output_path

    if output_path.exists() and overwrite:
        print(f"  → {treatment_name}: overwriting existing data...")

    print(f"  Generating base data for {treatment_name}...")

    # Use the generation task from tasks.py - pass exp_config directly
    # Note: generation task expects config_path, but we'll modify it to accept exp_config
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
        / treatment_name
        / "generation_logs"
    )
    log_dir.mkdir(parents=True, exist_ok=True)

    # Run generation
    eval_logs = eval(task, log_dir=str(log_dir))
    assert len(eval_logs) == 1, "Expected only one eval log"
    eval_log = eval_logs[0]

    # Extract outputs
    data_dict = construct_data_dict(eval_log)

    # Save to data.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_json(data_dict, output_path)

    print(f"  ✓ {treatment_name}: Saved {len(data_dict)} samples to {output_path}")
    return output_path


def apply_treatments(
    base_data_path: Path,
    dataset_name: str,
    data_subset: str,
    model_name: str,
    gen_config: dict,
    overwrite: bool = False,
):
    """
    Apply all treatment combinations specified in the config.

    Args:
        base_data_path: Path to base data.json
        dataset_name: Dataset name
        data_subset: Data subset directory
        model_name: Model name (used as base treatment name)
        gen_config: Generation config with treatment specifications
        overwrite: If True, reapply treatments even if data exists

    Config format:
        treatments:
          caps: [S2, S4]
          typos: [S2, S4]

    If 'treatments' is not present or empty, no treatments are applied.
    """
    treatments = gen_config.get("treatments")
    if not treatments:
        print("  No treatments specified - base data only")
        return

    # Get seed from config
    seed = gen_config.get("seed")

    # Apply each treatment type and strength combination
    for treatment_type, strengths in treatments.items():
        if not strengths:
            continue

        for strength in strengths:
            treatment_name = f"{model_name}_{treatment_type}_{strength}"
            output_path = (
                data_dir()
                / "input"
                / dataset_name
                / data_subset
                / treatment_name
                / "data.json"
            )

            # Check if already exists
            if output_path.exists() and not overwrite:
                print(f"  ✓ {treatment_name}: data already exists, skipping treatment")
                continue

            if output_path.exists() and overwrite:
                print(f"  → {treatment_name}: overwriting existing treatment...")

            print(
                f"  Applying {treatment_type} treatment ({strength}) to {model_name}..."
            )

            # Apply treatment
            apply_treatment(
                treatment_type=treatment_type,
                strength=strength,
                input_path=str(base_data_path),
                output_path=str(output_path),
                seed=seed,
            )

            print(f"  ✓ {treatment_name}: treatment applied, saved to {output_path}")


def run_generation(
    model_name: str,
    dataset_path: str,
    dataset_config: str,
    overwrite: bool = False,
):
    """
    Generate data using a model and apply treatments.

    Args:
        model_name: Model to use for generation (e.g., 'haiku-3-5')
        dataset_path: Path to input.json (e.g., 'data/wikisum/debug/input.json')
        dataset_config: Path to generation config YAML with temperature, treatments, etc.
        overwrite: If True, regenerate/reapply even if data exists
    """
    # Parse dataset path to determine output location
    dataset_path_obj = Path(dataset_path)
    parts = dataset_path_obj.parts

    # Expected: data/input/dataset_name/data_subset/input.json
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

    # Load generation config (simple config with temperature, treatments, etc.)
    gen_config = load_generation_config(dataset_config)

    # Create ExperimentConfig for generation (reuses config.py pipeline)
    exp_config = create_generation_config(
        dataset_name=dataset_name,
        temperature=gen_config.get("temperature"),
        max_tokens=gen_config.get("max_tokens"),
        seed=gen_config.get("seed"),
    )

    print(f"\n{'=' * 60}")
    print(f"Generating data for {dataset_name}/{data_subset}")
    print(f"Model: {model_name}")
    if overwrite:
        print("Mode: OVERWRITE (regenerating existing data)")
    else:
        print("Mode: SKIP (skipping existing data)")
    print(f"{'=' * 60}")

    # Step 1: Generate base data using the pipeline
    base_data_path = _generate_base_data(
        model_name=model_name,
        dataset_name=dataset_name,
        data_subset=data_subset,
        exp_config=exp_config,
        overwrite=overwrite,
    )

    # Step 2: Apply treatments
    apply_treatments(
        base_data_path=base_data_path,
        dataset_name=dataset_name,
        data_subset=data_subset,
        model_name=model_name,
        gen_config=gen_config,
        overwrite=overwrite,
    )

    print(f"\n{'=' * 60}")
    print("All data generation complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Generate data using a model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python experiments/scripts/generate_data.py \\
    --model_name=haiku-3-5 \\
    --dataset_path=data/wikisum/debug/input/input.json \\
    --dataset_config=experiments/00_data_gen/configs/config.yaml
        """,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name (e.g., 'haiku-3-5', 'gpt-4')",
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

    run_generation(
        model_name=args.model_name,
        dataset_path=args.dataset_path,
        dataset_config=args.dataset_config,
        overwrite=args.overwrite,
    )
