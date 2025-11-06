"""
Batch experiment runner for multiple models and treatment comparisons.

This script automates running experiments across different treatment types:
- other_models: Compare models against each other
- caps: Compare models against their capitalization treatments
- typos: Compare models against their typo treatments
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from run_experiment import run_judging_evals
from src.inspect.config import load_experiment_config


def run_single_evaluation(
    model_name: str,
    control_path: str,
    experiment_config: str,
    treatment_path: str | None,
    is_control: bool,
    eval_description: str,
) -> tuple[bool, str, str]:
    """
    Run a single evaluation (wrapper for parallel execution).

    Returns:
        (success, model_name, description_or_error)
    """
    try:
        run_judging_evals(
            model_name=model_name,
            dataset_path_control=control_path,
            experiment_config_path=experiment_config,
            dataset_path_treatment=treatment_path,
            is_control=is_control,
        )
        return (True, model_name, eval_description)
    except Exception as e:
        return (False, model_name, f"{eval_description}: {str(e)}")


def discover_datasets(dataset_dir_path: Path) -> dict[str, list[str]]:
    """
    Discover available datasets in the directory.

    Args:
        dataset_dir_path: Path to data/input/dataset_name/data_subset

    Returns:
        dict with keys 'base_models', 'caps_treatments', 'typos_treatments'
        Each value is a list of treatment names (directory names)
    """
    if not dataset_dir_path.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir_path}")

    base_models = []
    caps_treatments = {}  # model_name -> [treatment_names]
    typos_treatments = {}  # model_name -> [treatment_names]

    for subdir in dataset_dir_path.iterdir():
        if not subdir.is_dir():
            continue

        dir_name = subdir.name

        # Check if it's a treatment directory
        if "_caps_" in dir_name:
            base_name = dir_name.split("_caps_")[0]
            if base_name not in caps_treatments:
                caps_treatments[base_name] = []
            caps_treatments[base_name].append(dir_name)
        elif "_typos_" in dir_name:
            base_name = dir_name.split("_typos_")[0]
            if base_name not in typos_treatments:
                typos_treatments[base_name] = []
            typos_treatments[base_name].append(dir_name)
        else:
            # It's a base model directory
            base_models.append(dir_name)

    return {
        "base_models": sorted(base_models),
        "caps_treatments": {k: sorted(v) for k, v in caps_treatments.items()},
        "typos_treatments": {k: sorted(v) for k, v in typos_treatments.items()},
    }


def run_batch_experiment(
    model_names: list[str],
    treatment_type: str,
    dataset_dir_path: str,
    experiment_config: str,
    overwrite: bool = False,
    max_workers: int = 4,
):
    """
    Run batch experiments across multiple models and treatments (parallelized).

    Args:
        model_names: List of model names to evaluate
        treatment_type: One of 'other_models', 'caps', 'typos'
        dataset_dir_path: Path to dataset subset (e.g., 'data/pku_saferlhf/mismatch_1-20')
        experiment_config: Path to experiment config YAML
        overwrite: If True, re-run even if evaluation exists
        max_workers: Number of parallel workers (default: 4)
    """
    dataset_path = Path(dataset_dir_path)

    # Parse dataset path - Expected: data/input/dataset_name/data_subset
    parts = dataset_path.parts
    if parts[0] == "data" and parts[1] == "input":
        dataset_name = parts[2]
        data_subset = parts[3]
    elif parts[0] == "input":
        dataset_name = parts[1]
        data_subset = parts[2]
    else:
        raise ValueError(
            f"Invalid dataset directory format: {dataset_dir_path}. "
            f"Expected: data/input/dataset_name/data_subset"
        )

    # Load experiment config with dataset_name to determine pairwise vs individual
    exp_config = load_experiment_config(experiment_config, dataset_name=dataset_name)
    is_pairwise = exp_config.is_pairwise()

    # Discover available datasets
    datasets = discover_datasets(dataset_path)

    print(f"\n{'=' * 70}")
    print("BATCH EXPERIMENT RUNNER (Parallel)")
    print(f"{'=' * 70}")
    print(f"Dataset: {dataset_name}/{data_subset}")
    print(f"Experiment: {Path(experiment_config).parent.name}")
    print(f"Format: {'Pairwise' if is_pairwise else 'Individual'}")
    print(f"Treatment type: {treatment_type}")
    print(f"Models to evaluate: {len(model_names)}")
    print(f"  {', '.join(model_names)}")
    print(f"Parallel workers: {max_workers}")
    print(f"Mode: {'OVERWRITE' if overwrite else 'SKIP existing'}")
    print(f"{'=' * 70}\n")

    # Validate models exist
    for model in model_names:
        if model not in datasets["base_models"]:
            print(f"Warning: Model '{model}' not found in {dataset_dir_path}")

    # Build list of all evaluation tasks
    evaluation_tasks = []

    print("Building evaluation task list...\n")

    for model_name in model_names:
        if model_name not in datasets["base_models"]:
            print(
                f"  ✗ Model data not found, skipping all evaluations for {model_name}"
            )
            continue

        # Construct control path
        control_path = f"data/input/{dataset_name}/{data_subset}/{model_name}/data.json"

        if treatment_type == "other_models":
            # Model vs Model comparisons
            other_models = [m for m in model_names if m != model_name]

            for other_model in other_models:
                if other_model not in datasets["base_models"]:
                    continue

                treatment_path = (
                    f"data/input/{dataset_name}/{data_subset}/{other_model}/data.json"
                )

                if is_pairwise:
                    evaluation_tasks.append(
                        (
                            model_name,
                            control_path,
                            experiment_config,
                            treatment_path,
                            True,
                            f"{model_name} vs {other_model}",
                        )
                    )
                else:
                    # Individual: evaluate other model's data as treatment
                    evaluation_tasks.append(
                        (
                            model_name,
                            treatment_path,
                            experiment_config,
                            None,
                            False,
                            f"{model_name} evaluating {other_model}",
                        )
                    )

            # For individual mode, also evaluate control dataset
            if not is_pairwise:
                evaluation_tasks.append(
                    (
                        model_name,
                        control_path,
                        experiment_config,
                        None,
                        True,
                        f"{model_name} (control)",
                    )
                )

        elif treatment_type in ["caps", "typos"]:
            # Model vs its own treatments
            treatment_dict = (
                datasets["caps_treatments"]
                if treatment_type == "caps"
                else datasets["typos_treatments"]
            )
            treatments = treatment_dict.get(model_name, [])

            if not treatments:
                print(f"  No {treatment_type} treatments found for {model_name}")
                continue

            # For individual mode, first evaluate control
            if not is_pairwise:
                evaluation_tasks.append(
                    (
                        model_name,
                        control_path,
                        experiment_config,
                        None,
                        True,
                        f"{model_name} (control)",
                    )
                )

            # Evaluate each treatment
            for treatment_name in treatments:
                treatment_path = f"data/input/{dataset_name}/{data_subset}/{treatment_name}/data.json"

                if is_pairwise:
                    evaluation_tasks.append(
                        (
                            model_name,
                            control_path,
                            experiment_config,
                            treatment_path,
                            True,
                            f"{model_name} vs {treatment_name}",
                        )
                    )
                else:
                    # Individual: evaluate treatment as treatment
                    evaluation_tasks.append(
                        (
                            model_name,
                            treatment_path,
                            experiment_config,
                            None,
                            False,
                            f"{model_name} evaluating {treatment_name}",
                        )
                    )

    total_evals = len(evaluation_tasks)
    print(f"\nTotal evaluations to run: {total_evals}")
    print(f"Running evaluations in parallel with {max_workers} workers...\n")

    # Execute all evaluations in parallel
    successful = []
    failed = []
    completed = 0

    print_lock = Lock()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_evaluation,
                task[0],
                task[1],
                task[2],
                task[3],
                task[4],
                task[5],
            ): task
            for task in evaluation_tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            completed += 1

            try:
                success, model_name, description = future.result()

                with print_lock:
                    if success:
                        print(f"[{completed}/{total_evals}] ✓ {description}")
                        successful.append((model_name, description))
                    else:
                        print(f"[{completed}/{total_evals}] ✗ {description}")
                        failed.append((model_name, description, ""))
            except Exception as e:
                with print_lock:
                    print(
                        f"[{completed}/{total_evals}] ✗ {task[5]}: Unexpected error: {str(e)}"
                    )
                    failed.append((task[0], task[5], str(e)))

    # Summary
    print(f"\n{'=' * 70}")
    print("BATCH EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total evaluations: {total_evals}")
    print(f"Successful: {len(successful)}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for model, description, error in failed[:10]:  # Show first 10
            error_msg = error[:50] if error else "Failed"
            print(f"  ✗ {model}: {description} - {error_msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Batch run experiments across multiple models and treatments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models against each other
  python experiments/scripts/run_experiment_batch.py \\
    --model_names haiku-3-5 gpt-4.1 ll-3-1-8b \\
    --treatment_type other_models \\
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \\
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml

  # Compare models against caps treatments
  python experiments/scripts/run_experiment_batch.py \\
    --model_names haiku-3-5 gpt-4.1 \\
    --treatment_type caps \\
    --dataset_dir_path data/input/wikisum/training_set_1-20 \\
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml
        """,
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to evaluate (e.g., 'haiku-3-5 gpt-4.1')",
    )
    parser.add_argument(
        "--treatment_type",
        type=str,
        required=True,
        choices=["other_models", "caps", "typos"],
        help="Type of comparison: 'other_models', 'caps', or 'typos'",
    )
    parser.add_argument(
        "--dataset_dir_path",
        type=str,
        required=True,
        help="Path to dataset subset directory (e.g., 'data/input/pku_saferlhf/mismatch_1-20')",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Path to experiment config YAML (e.g., 'experiments/01_AT_PW-C_Rec_Pr/config.yaml')",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing evaluations (default: skip existing)",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=4,
        help="Number of parallel workers (default: 4)",
    )

    args = parser.parse_args()

    run_batch_experiment(
        model_names=args.model_names,
        treatment_type=args.treatment_type,
        dataset_dir_path=args.dataset_dir_path,
        experiment_config=args.experiment_config,
        overwrite=args.overwrite,
        max_workers=args.max_workers,
    )
