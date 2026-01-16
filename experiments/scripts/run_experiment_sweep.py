"""
Sweep experiment runner for multiple models and treatment comparisons.

This script automates running experiments across different treatment types:
- other_models: Compare models against each other
- caps: Compare models against their capitalization treatments
- typos: Compare models against their typo treatments

Note: This script runs evaluations sequentially (not in parallel) because each evaluation
requires a separate log directory, which Inspect AI doesn't support for parallel task execution.
However, Inspect AI's internal sample-level parallelism still provides good performance.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval
from run_experiment import (
    parse_dataset_path,
    check_rollout_exists,
)
from src.inspect.tasks import get_task_function
from src.inspect.config import load_experiment_config
from utils import expand_model_names


def _add_task_if_needed(
    tasks_to_run: list,
    exp_config,
    model_name: str,
    treatment_name_control: str,
    treatment_name_treatment: str | None,
    dataset_name: str,
    data_subset: str,
    experiment_name: str,
    is_control: bool,
    description: str,
    overwrite: bool,
    shared_log_dir: Path,
):
    """
    Helper to add a task to the list if it doesn't already exist (or if overwrite=True).

    Args:
        tasks_to_run: List to append (Task, description) tuples to
        exp_config: Experiment configuration
        model_name: Model name to evaluate with
        treatment_name_control: Control treatment name
        treatment_name_treatment: Treatment name (for pairwise) or None
        dataset_name: Dataset name
        data_subset: Data subset
        experiment_name: Experiment name
        is_control: Whether evaluating control or treatment
        description: Human-readable description
        overwrite: Whether to overwrite existing evaluations
        shared_log_dir: Shared log directory for all tasks
    """
    # Build task name for log filename (using hyphens to match Inspect AI's log filename format)
    if treatment_name_treatment:
        task_name = f"{model_name}-eval-on-{treatment_name_control}-vs-{treatment_name_treatment}"
    else:
        control_or_treatment = "control" if is_control else "treatment"
        task_name = (
            f"{model_name}-eval-on-{treatment_name_control}-{control_or_treatment}"
        )

    # Check if already exists (look for log files with this task name in shared log dir)
    if not overwrite:
        # Use glob to find potential matches, but verify exact match by parsing filename
        potential_logs = list(shared_log_dir.glob(f"*{task_name}*.eval"))
        existing_logs = []

        # Verify exact match by parsing filenames
        # Import parse_eval_filename from analyze_pairwise_results
        from analyze_pairwise_results import parse_eval_filename

        for log_file in potential_logs:
            parsed = parse_eval_filename(log_file.name)
            if parsed:
                parsed_evaluator, parsed_control, parsed_treatment = parsed
                # Check if this matches exactly what we're looking for
                if treatment_name_treatment:
                    # Pairwise: check evaluator, control, and treatment all match
                    if (
                        parsed_evaluator == model_name
                        and parsed_control == treatment_name_control
                        and parsed_treatment == treatment_name_treatment
                    ):
                        existing_logs.append(log_file)
                else:
                    # Non-pairwise: check evaluator and control match
                    # (treatment will be None in this case)
                    if (
                        parsed_evaluator == model_name
                        and parsed_control == treatment_name_control
                    ):
                        existing_logs.append(log_file)

        if existing_logs:
            # Check status of existing log - only skip if successful
            from inspect_ai.log import read_eval_log

            try:
                log = read_eval_log(existing_logs[0])

                # Helper function to check if log has actual results
                def has_results(log):
                    """Check if log contains samples with scores/results."""
                    if not log.samples or len(log.samples) == 0:
                        return False
                    # Check if at least one sample has scores/results
                    # Scores indicate the evaluation actually completed
                    for sample in log.samples:
                        if sample.scores and len(sample.scores) > 0:
                            return True
                    return False

                if log.status == "success":
                    if has_results(log):
                        print(f"  ⊘ {description}: already exists (success), skipping")
                        return
                    else:
                        # Success status but no results - treat as incomplete/corrupt
                        print(
                            f"  ↻ {description}: incomplete log (success but no results), re-running"
                        )
                        for old_log in existing_logs:
                            old_log.unlink()
                elif log.status == "started":
                    # Check if "started" log actually has results
                    # If it does, it might be a legitimate in-progress batch job
                    # If it doesn't, it's an orphaned/canceled batch job that should be overwritten
                    if has_results(log):
                        # Has results - might be legitimate in-progress batch job
                        print(
                            f"  ⏳ {description}: batch job in progress (has results), skipping"
                        )
                        print(
                            "     (Use scripts/list_active_batches.py to check status)"
                        )
                        return
                    else:
                        # No results - orphaned/canceled batch job, overwrite it
                        print(
                            f"  ↻ {description}: incomplete log (started but no results), re-running"
                        )
                        for old_log in existing_logs:
                            old_log.unlink()
                else:
                    # Failed/cancelled/error - delete and re-run
                    print(f"  ↻ {description}: previous run {log.status}, re-running")
                    for old_log in existing_logs:
                        old_log.unlink()
            except Exception:
                # Corrupt log - delete and re-run
                print(f"  ⚠ {description}: corrupt log, re-running")
                for old_log in existing_logs:
                    old_log.unlink()

    # Check data files exist
    control_path = (
        f"data/input/{dataset_name}/{data_subset}/{treatment_name_control}/data.json"
    )
    dataset_name_ctrl, data_subset_ctrl, treatment_ctrl = parse_dataset_path(
        control_path
    )
    if not check_rollout_exists(dataset_name_ctrl, data_subset_ctrl, treatment_ctrl):
        print(f"  ✗ {description}: missing control data, skipping")
        return

    if treatment_name_treatment:
        treatment_path = f"data/input/{dataset_name}/{data_subset}/{treatment_name_treatment}/data.json"
        dataset_name_treat, data_subset_treat, treatment_treat = parse_dataset_path(
            treatment_path
        )
        if not check_rollout_exists(
            dataset_name_treat, data_subset_treat, treatment_treat
        ):
            print(f"  ✗ {description}: missing treatment data, skipping")
            return

    # Create task with custom name
    task = get_task_function(
        exp_config=exp_config,
        model_name=model_name,
        treatment_name_control=treatment_name_control,
        treatment_name_treatment=treatment_name_treatment,
        dataset_name=dataset_name,
        data_subset=data_subset,
        is_control=is_control,
        task_name=task_name,
    )

    tasks_to_run.append((task, description))
    print(f"  ✓ {description}: added to sweep")


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


def run_sweep_experiment(
    model_names: list[str],
    treatment_type: str,
    dataset_dir_path: str,
    experiment_config: str,
    overwrite: bool = False,
    batch: bool | int | str = False,
    max_tasks: int = 8,
):
    """
    Run sweep experiments across multiple models and treatments.

    Uses Inspect AI's native parallel execution to run multiple evaluations concurrently.
    Each evaluation also benefits from sample-level parallelism (max_connections).

    All evaluation logs are saved to a shared directory:
    data/results/{dataset_name}/{data_subset}/{experiment_name}/

    Task names are automatically generated to preserve comparison information in log filenames.

    Args:
        model_names: List of model names to evaluate
        treatment_type: One of 'other_models', 'caps', 'typos'
        dataset_dir_path: Path to dataset subset (e.g., 'data/input/pku_saferlhf/mismatch_1-20')
        experiment_config: Path to experiment config YAML
        overwrite: If True, re-run even if evaluation exists
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
        max_tasks: Maximum number of tasks to run in parallel (default: 8)
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

    experiment_name = Path(experiment_config).parent.name

    # Create shared log directory for all tasks
    from src.helpers.utils import data_dir

    shared_log_dir = (
        data_dir() / "results" / dataset_name / data_subset / experiment_name
    )
    shared_log_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("SWEEP EXPERIMENT RUNNER")
    print(f"{'=' * 70}")
    print(f"Dataset: {dataset_name}/{data_subset}")
    print(f"Experiment: {experiment_name}")
    print(f"Format: {'Pairwise' if is_pairwise else 'Individual'}")
    print(f"Treatment type: {treatment_type}")
    print(f"Models to evaluate: {len(model_names)}")
    print(f"  {', '.join(model_names)}")
    print(f"Mode: {'OVERWRITE' if overwrite else 'SKIP existing'}")
    print(f"Log directory: {shared_log_dir}")
    print(f"{'=' * 70}\n")

    # Validate models exist
    for model in model_names:
        if model not in datasets["base_models"]:
            print(f"Warning: Model '{model}' not found in {dataset_dir_path}")

    # Build list of all evaluation tasks
    tasks_to_run = []  # List of (Task, description) tuples

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

                _, _, treatment_name_treat = parse_dataset_path(
                    f"data/input/{dataset_name}/{data_subset}/{other_model}/data.json"
                )

                _, _, treatment_name_ctrl = parse_dataset_path(control_path)

                if is_pairwise:
                    _add_task_if_needed(
                        tasks_to_run,
                        exp_config,
                        model_name,
                        treatment_name_ctrl,
                        treatment_name_treat,
                        dataset_name,
                        data_subset,
                        experiment_name,
                        True,
                        f"{model_name} vs {other_model}",
                        overwrite,
                        shared_log_dir,
                    )
                else:
                    # Individual: evaluate other model's data as treatment
                    _add_task_if_needed(
                        tasks_to_run,
                        exp_config,
                        model_name,
                        treatment_name_treat,
                        None,
                        dataset_name,
                        data_subset,
                        experiment_name,
                        False,
                        f"{model_name} evaluating {other_model}",
                        overwrite,
                        shared_log_dir,
                    )

            # For individual mode, also evaluate control dataset
            if not is_pairwise:
                _, _, treatment_name_ctrl = parse_dataset_path(control_path)
                _add_task_if_needed(
                    tasks_to_run,
                    exp_config,
                    model_name,
                    treatment_name_ctrl,
                    None,
                    dataset_name,
                    data_subset,
                    experiment_name,
                    True,
                    f"{model_name} (control)",
                    overwrite,
                    shared_log_dir,
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

            _, _, treatment_name_ctrl = parse_dataset_path(control_path)

            # For individual mode, first evaluate control
            if not is_pairwise:
                _add_task_if_needed(
                    tasks_to_run,
                    exp_config,
                    model_name,
                    treatment_name_ctrl,
                    None,
                    dataset_name,
                    data_subset,
                    experiment_name,
                    True,
                    f"{model_name} (control)",
                    overwrite,
                    shared_log_dir,
                )

            # Evaluate each treatment
            for treatment_name in treatments:
                if is_pairwise:
                    _add_task_if_needed(
                        tasks_to_run,
                        exp_config,
                        model_name,
                        treatment_name_ctrl,
                        treatment_name,
                        dataset_name,
                        data_subset,
                        experiment_name,
                        True,
                        f"{model_name} vs {treatment_name}",
                        overwrite,
                        shared_log_dir,
                    )
                else:
                    # Individual: evaluate treatment as treatment
                    _add_task_if_needed(
                        tasks_to_run,
                        exp_config,
                        model_name,
                        treatment_name,
                        None,
                        dataset_name,
                        data_subset,
                        experiment_name,
                        False,
                        f"{model_name} evaluating {treatment_name}",
                        overwrite,
                        shared_log_dir,
                    )

    total_evals = len(tasks_to_run)

    if total_evals == 0:
        print("\n⊘ No evaluations to run (all skipped or already exist)")
        return

    # Separate models that don't support batch mode (Google Gemini, GPT-5/o-series)
    # Google/Gemini models have bugs in Inspect AI batch mode
    # GPT-5/o-series models return unsupported_value errors in batch mode

    no_batch_tasks = []
    no_batch_descriptions = []
    batch_tasks = []
    batch_descriptions = []

    for task, desc in tasks_to_run:
        # Check if task uses a model that doesn't support batch mode as evaluator
        # The description format is "{evaluator_model} vs {comparison_model}"
        # We only care about the evaluator (first part before "vs" or "evaluating")
        evaluator_model = desc.split(" vs ")[0].split(" evaluating ")[0].strip()

        # Check if it's a model that doesn't support batch mode
        # Google/Gemini models have bugs in Inspect AI batch mode
        # GPT-5.1 specifically returns unsupported_value errors in batch mode
        # Grok models (XAI) don't support batch mode
        # Note: gpt-5 (without .1) is allowed to try batch mode
        is_gemini = "gemini" in evaluator_model.lower()
        is_gpt5_1 = (
            evaluator_model.lower() == "gpt-5.1"
            or evaluator_model.lower().startswith("gpt-5.1")
        )
        is_grok = "grok" in evaluator_model.lower()

        if is_gemini or is_gpt5_1 or is_grok:
            no_batch_tasks.append(task)
            no_batch_descriptions.append(desc)
        else:
            batch_tasks.append(task)
            batch_descriptions.append(desc)

    # Display summary and ask for confirmation
    print(f"\n{'='*70}")
    print(f"READY TO RUN: {total_evals} evaluations")
    print(f"Parallelism: max_tasks={max_tasks}")
    print(f"Batch mode: {'enabled' if batch else 'disabled'}")

    if batch and no_batch_tasks:
        print(
            f"\n\033[91m⚠ WARNING: Batch mode disabled for {len(no_batch_tasks)} evaluations"
        )
        print(
            "  (Google Gemini batch mode has bugs; GPT-5.1/Grok return unsupported_value errors)\033[0m"
        )
        print(f"  • Batch-compatible models: {len(batch_tasks)} evals WITH batch mode")
        print(f"  • Non-batch models: {len(no_batch_tasks)} evals WITHOUT batch mode")

    print(f"{'='*70}\n")

    response = input("Continue? (y/n): ").strip().lower()
    if response != "y":
        print("\n✗ Aborted by user.\n")
        return

    print("\n✓ Starting evaluation sweep...\n")

    # Run evaluations - split into two groups if batch mode + models that don't support batch
    eval_logs = []

    if batch and no_batch_tasks:
        # Run non-batch models first, then batch-compatible models with batch
        if no_batch_tasks:
            print(f"Running {len(no_batch_tasks)} evaluations WITHOUT batch mode...")
            try:
                no_batch_logs = eval(
                    no_batch_tasks,
                    log_dir=str(shared_log_dir),
                    max_tasks=max_tasks,
                    batch=False,
                )
                eval_logs.extend(no_batch_logs)
            except Exception as e:
                print(f"\n⚠ Error in non-batch evaluations: {e}")
                raise

        if batch_tasks:
            print(f"\nRunning {len(batch_tasks)} evaluations WITH batch mode...")
            try:
                batch_logs = eval(
                    batch_tasks,
                    log_dir=str(shared_log_dir),
                    max_tasks=max_tasks,
                    batch=batch,
                )
                eval_logs.extend(batch_logs)
            except Exception as e:
                print(f"\n⚠ Error in batch evaluations: {e}")
                raise

        # Merge descriptions in same order
        descriptions = no_batch_descriptions + batch_descriptions
    else:
        # Run all together
        try:
            tasks = [task for task, _ in tasks_to_run]
            descriptions = [desc for _, desc in tasks_to_run]
            eval_logs = eval(
                tasks,
                log_dir=str(shared_log_dir),
                max_tasks=max_tasks,
                batch=batch,
            )
        except Exception as e:
            print(f"\n⚠ Error in evaluations: {e}")
            raise

    # Check for any failures
    successful = []
    failed = []

    for idx, (eval_log, description) in enumerate(zip(eval_logs, descriptions)):
        if eval_log.status == "success":
            successful.append(description)
        else:
            error_msg = (
                eval_log.error if hasattr(eval_log, "error") else "Unknown error"
            )
            failed.append((description, str(error_msg)))

    # Summary
    print(f"\n{'=' * 70}")
    print("SWEEP EXPERIMENT SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total evaluations: {total_evals}")
    print(f"Successful: {len(successful)}")

    if failed:
        print(f"\nFailed: {len(failed)}")
        for description, error in failed[:10]:  # Show first 10
            error_msg = error[:50] if error else "Failed"
            print(f"  ✗ {description}: {error_msg}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    load_dotenv(override=True)

    parser = argparse.ArgumentParser(
        description="Sweep run experiments across multiple models and treatments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare models against each other
  python experiments/scripts/run_experiment_sweep.py \\
    --model_names haiku-3-5 gpt-4.1 ll-3-1-8b \\
    --treatment_type other_models \\
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \\
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml

  # Use a model set (e.g., gen_cot models)
  python experiments/scripts/run_experiment_sweep.py \\
    --model_names -set cot \\
    --treatment_type other_models \\
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \\
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml

  # Mix individual models and sets
  python experiments/scripts/run_experiment_sweep.py \\
    --model_names haiku-3-5 -set dr gpt-4.1 \\
    --treatment_type other_models \\
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \\
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml
        """,
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names to evaluate (e.g., 'haiku-3-5 gpt-4.1') or model sets (e.g., '-set cot' for gen_cot set). "
        "Can mix individual models and sets.",
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
        "--batch",
        nargs="?",
        const=True,
        default=False,
        help="Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI). "
        "Usage: --batch (default config), --batch 1000 (batch size), --batch config.yaml (config file)",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=8,
        help="Maximum number of tasks to run in parallel (default: 8)",
    )

    args = parser.parse_args()

    # Expand model set references (e.g., '-set cot' -> list of models)
    expanded_model_names = expand_model_names(args.model_names)

    # Parse batch argument
    batch_value = args.batch
    if batch_value is not False:
        # Try to convert to int if it's a number string
        if isinstance(batch_value, str) and batch_value.isdigit():
            batch_value = int(batch_value)

    run_sweep_experiment(
        model_names=expanded_model_names,
        treatment_type=args.treatment_type,
        dataset_dir_path=args.dataset_dir_path,
        experiment_config=args.experiment_config,
        overwrite=args.overwrite,
        batch=batch_value,
        max_tasks=args.max_tasks,
    )
