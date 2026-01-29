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
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add parent directory to path to allow importing utils
sys.path.insert(0, str(Path(__file__).parent.parent))

from inspect_ai import eval
from inspect_ai.log import read_eval_log
from run_experiment import (
    parse_dataset_path,
    check_rollout_exists,
)
from src.inspect.tasks import get_task_function, _is_always_reasoning_model
from src.inspect.config import load_experiment_config
from src.helpers.model_names import is_thinking_model, get_data_model_name
from utils import expand_model_names


def _collect_partial_results(log_dir: Path, eval_logs: list, descriptions: list[str]):
    """
    Try to collect any partial results from completed eval logs in the log directory.

    This is called when an exception occurs during eval() to recover any completed
    evaluations.

    Args:
        log_dir: Directory where eval logs are saved
        eval_logs: List to append recovered logs to
        descriptions: List of descriptions corresponding to tasks
    """
    try:
        # Find all .eval files in the log directory
        eval_files = list(log_dir.glob("*.eval"))
        if not eval_files:
            return

        # Sort by modification time (newest first)
        eval_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Get the most recent eval files (up to the number of tasks)
        recent_files = eval_files[: len(descriptions)]

        recovered = 0
        for eval_file in recent_files:
            try:
                log = read_eval_log(str(eval_file))
                # Check if this log was successful and has samples
                if log.status == "success" and log.samples:
                    # Check if we already have this log
                    existing_tasks = {
                        getattr(log_item.eval, "task", None) for log_item in eval_logs
                    }
                    if log.eval.task not in existing_tasks:
                        eval_logs.append(log)
                        recovered += 1
            except Exception:
                continue

        if recovered > 0:
            print(f"  ✓ Recovered {recovered} completed evaluation(s) from partial run")
    except Exception as e:
        print(f"  ⚠ Could not recover partial results: {e}")


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
    skip_batch_in_progress: bool = False,
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
        # Import parse_eval_filename from recognition_accuracy
        # Add analysis directory to path
        analysis_dir = Path(__file__).parent.parent / "analysis"
        if str(analysis_dir) not in sys.path:
            sys.path.insert(0, str(analysis_dir))
        from recognition_accuracy import parse_eval_filename

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
                        # Default behavior: re-run (usually indicates stuck eval file)
                        # Only skip if explicitly requested via --skip_batch_in_progress
                        if skip_batch_in_progress:
                            print(
                                f"  ⏳ {description}: batch job in progress (has results), skipping"
                            )
                            print(
                                "     (Use scripts/list_active_batches.py to check status)"
                            )
                            return
                        else:
                            # Default: re-run (usually indicates stuck eval file)
                            print(
                                f"  ↻ {description}: batch job in progress (has results), re-running (use --skip_batch_in_progress to skip)"
                            )
                            for old_log in existing_logs:
                                old_log.unlink()
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
    yes: bool = False,
    generator_models: list[str] | None = None,
    skip_batch_in_progress: bool = False,
):
    """
    Run sweep experiments across multiple models and treatments.

    Uses Inspect AI's native parallel execution to run multiple evaluations concurrently.
    Each evaluation also benefits from sample-level parallelism (max_connections).

    All evaluation logs are saved to a shared directory:
    data/results/{dataset_name}/{data_subset}/{experiment_name}/

    Task names are automatically generated to preserve comparison information in log filenames.

    Args:
        model_names: List of model names to evaluate (Evaluators)
        treatment_type: One of 'other_models', 'caps', 'typos'
        dataset_dir_path: Path to dataset subset (e.g., 'data/input/pku_saferlhf/mismatch_1-20')
        experiment_config: Path to experiment config YAML
        overwrite: If True, re-run even if evaluation exists
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
        max_tasks: Maximum number of tasks to run in parallel (default: 8)
        yes: If True, skip confirmation prompt and run immediately
        generator_models: Optional list of models to generate data from. If provided,
                          evaluators will be paired against these models.
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

    # Resolve path first to handle relative paths like "../config.yaml"
    experiment_name = Path(experiment_config).resolve().parent.name

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
    print(f"Evaluator models: {len(model_names)}")
    print(f"  {', '.join(model_names)}")
    if generator_models:
        print(f"Generator models: {len(generator_models)}")
        print(f"  {', '.join(generator_models)}")
    print(f"Mode: {'OVERWRITE' if overwrite else 'SKIP existing'}")
    print(f"Log directory: {shared_log_dir}")
    print(f"{'=' * 70}\n")

    # Pre-filter models that can't run in instruct mode (always-reasoning models without -thinking)
    valid_models = []
    invalid_models = []

    for model in model_names:
        if _is_always_reasoning_model(model) and not is_thinking_model(model):
            invalid_models.append(model)
            print(f"  ⚠ {model}: Cannot use in instruct mode (always uses reasoning)")
            print(f"     Use '{model}-thinking' instead.")
        else:
            valid_models.append(model)

    if invalid_models:
        print(
            f"\n⚠ Skipped {len(invalid_models)} always-reasoning model(s) without -thinking suffix\n"
        )

    model_names = valid_models

    if not model_names:
        print("\n⊘ No valid models to evaluate (all require -thinking suffix)")
        return

    # Validate models exist (check data model name for COT-I models)
    for model in model_names:
        data_model = get_data_model_name(model)
        if data_model not in datasets["base_models"]:
            if data_model != model:
                print(
                    f"Warning: Data for '{model}' (COT-I, uses '{data_model}') not found in {dataset_dir_path}"
                )
            else:
                print(f"Warning: Model '{model}' not found in {dataset_dir_path}")

    # Build list of all evaluation tasks
    tasks_to_run = []  # List of (Task, description) tuples

    print("Building evaluation task list...\n")

    for model_name in model_names:
        # Get the data model name (for COT-I models, this is the non-thinking base model)
        data_model_name = get_data_model_name(model_name)
        is_cot_i = data_model_name != model_name

        if data_model_name not in datasets["base_models"]:
            print(
                f"  ✗ Model data not found for {model_name} (looked for {data_model_name}), skipping all evaluations"
            )
            continue

        if is_cot_i:
            print(f"  ℹ {model_name} is COT-I, using data from {data_model_name}")

        # Construct control path using data model name
        control_path = (
            f"data/input/{dataset_name}/{data_subset}/{data_model_name}/data.json"
        )

        if treatment_type == "other_models":
            # Model vs Model comparisons

            # Determine target models (generators)
            # If generator_models provided, compare against them (excluding self)
            # Else, compare against all other evaluators (excluding self)
            if generator_models:
                other_models = [m for m in generator_models if m != model_name]
                should_run_control = model_name in generator_models
            else:
                other_models = [m for m in model_names if m != model_name]
                should_run_control = True

            for other_model in other_models:
                # Get data model name for other model too
                other_data_model = get_data_model_name(other_model)
                if other_data_model not in datasets["base_models"]:
                    continue

                _, _, treatment_name_treat = parse_dataset_path(
                    f"data/input/{dataset_name}/{data_subset}/{other_data_model}/data.json"
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
                        skip_batch_in_progress,
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
                        skip_batch_in_progress,
                    )

            # For individual mode, also evaluate control dataset (if appropriate)
            if not is_pairwise and should_run_control:
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
                    skip_batch_in_progress,
                )

        elif treatment_type in ["caps", "typos"]:
            # Model vs its own treatments
            # If generator_models provided, evaluate treatments of those models
            # Else, evaluate treatments of the evaluator model itself

            if generator_models:
                targets = generator_models
                should_run_control = model_name in generator_models
            else:
                targets = [model_name]
                should_run_control = True

            # Evaluate control first (if appropriate)
            if not is_pairwise and should_run_control:
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
                    skip_batch_in_progress,
                )

            for target_model in targets:
                target_data_model = get_data_model_name(target_model)
                
                treatment_dict = (
                    datasets["caps_treatments"]
                    if treatment_type == "caps"
                    else datasets["typos_treatments"]
                )
                
                treatments = treatment_dict.get(target_data_model, [])

                if not treatments:
                    print(f"  No {treatment_type} treatments found for {target_model}")
                    continue

                # Control is the base version of the target model
                # Note: In pairwise mode, we compare Target Base vs Target Treatment
                # But task definition uses 'model_name' (evaluator) to define 'control' usually?
                # Actually, check _add_task_if_needed logic for pairwise.
                # It uses treatment_name_ctrl and treatment_name_treat.
                # treatment_name_ctrl comes from control_path.
                
                # If target != model_name, we need control_path to point to target's base data
                target_control_path = (
                    f"data/input/{dataset_name}/{data_subset}/{target_data_model}/data.json"
                )
                _, _, treatment_name_ctrl = parse_dataset_path(target_control_path)

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
                            f"{model_name} evaluating {target_model} vs {treatment_name}",
                            overwrite,
                            shared_log_dir,
                            skip_batch_in_progress,
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
                            skip_batch_in_progress,
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
        # Description formats: "{eval} vs {other}", "{eval} evaluating {other}", "{eval} (control)"
        evaluator_model = desc.split(" vs ")[0].split(" evaluating ")[0].strip()
        if evaluator_model.endswith(" (control)"):
            evaluator_model = evaluator_model[:-10]
        elif evaluator_model.endswith(" (treatment)"):
            evaluator_model = evaluator_model[:-12]

        # Check if it's a model that doesn't support batch mode
        # Google/Gemini models have bugs in Inspect AI batch mode
        # GPT-5.1 and o3 models return unsupported_value errors in batch mode
        # GPT-4o-mini takes unreasonably long in batch mode
        # Grok models (XAI) have batch API issues ("missing field name")
        # ll-3.3-70b-dsR1-thinking (DeepSeek R1 Distill) has batch rejection issues
        # DeepSeek R1 full model has batch parsing issues (status_code returned as string)
        # Note: gpt-5 (without .1) is allowed to try batch mode
        # Note: Together AI batch mode is supported and works well (except DeepSeek models)
        is_gemini = "gemini" in evaluator_model.lower()
        is_gpt5_1 = (
            evaluator_model.lower() == "gpt-5.1"
            or evaluator_model.lower().startswith("gpt-5.1")
        )
        is_gpt4o_mini = evaluator_model.lower() in ["gpt-4o-mini", "gpt4o-mini"]
        is_o3 = evaluator_model.lower().startswith("o3")
        is_grok = "grok" in evaluator_model.lower()
        is_dsr1_distill = "dsr1" in evaluator_model.lower()
        is_deepseek_r1 = "deepseek-r1" in evaluator_model.lower()

        if (
            is_gemini
            or is_gpt5_1
            or is_gpt4o_mini
            or is_o3
            or is_grok
            or is_dsr1_distill
            or is_deepseek_r1
        ):
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

    if yes:
        print("✓ Auto-confirmed (--yes flag)\n")
    else:
        response = input("Continue? (y/n): ").strip().lower()
        if response != "y":
            print("\n✗ Aborted by user.\n")
            return

    print("✓ Starting evaluation sweep...\n")

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
                error_str = str(e)
                if (
                    "AlwaysReasoningModelError" in error_str
                    or "always uses reasoning" in error_str
                ):
                    print("\n⚠ One or more models cannot be used in instruct mode.")
                    print("   Use '-thinking' suffix for always-reasoning models.\n")
                else:
                    print(f"\n⚠ Error in non-batch evaluations: {error_str[:200]}\n")
                # Try to collect any partial results from log directory
                _collect_partial_results(
                    shared_log_dir, eval_logs, no_batch_descriptions
                )

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
                error_str = str(e)
                # Check for OpenAI batch API "missing field `name`" error
                if (
                    "missing field `name`" in error_str
                    or "UnprocessableEntityError" in error_str
                ):
                    print(f"\n{'=' * 70}")
                    print("⚠ BATCH API ERROR: Missing 'name' field in batch creation")
                    print(f"{'=' * 70}")
                    print(
                        "This appears to be an Inspect AI bug where the 'name' field is"
                        " sometimes missing when creating OpenAI batches."
                    )
                    print("\nSuggested actions:")
                    print("  1. Re-run the script - this error is often intermittent")
                    print(
                        "  2. Check for active batches that may have been created: "
                        "scripts/list_active_batches.py"
                    )
                    print(
                        "  3. If batches were created, wait for completion or cancel them: "
                        "scripts/cancel_all_batches.py"
                    )
                    print(f"{'=' * 70}\n")
                elif (
                    "AlwaysReasoningModelError" in error_str
                    or "always uses reasoning" in error_str
                ):
                    print("\n⚠ One or more models cannot be used in instruct mode.")
                    print("   Use '-thinking' suffix for always-reasoning models.\n")
                else:
                    print(f"\n⚠ Error in batch evaluations: {error_str[:200]}\n")
                # Try to collect any partial results from log directory
                _collect_partial_results(shared_log_dir, eval_logs, batch_descriptions)

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
            error_str = str(e)
            # Check for OpenAI batch API "missing field `name`" error
            if (
                "missing field `name`" in error_str
                or "UnprocessableEntityError" in error_str
            ):
                print(f"\n{'=' * 70}")
                print("⚠ BATCH API ERROR: Missing 'name' field in batch creation")
                print(f"{'=' * 70}")
                print(
                    "This appears to be an Inspect AI bug where the 'name' field is"
                    " sometimes missing when creating OpenAI batches."
                )
                print("\nSuggested actions:")
                print("  1. Re-run the script - this error is often intermittent")
                print(
                    "  2. Check for active batches that may have been created: "
                    "scripts/list_active_batches.py"
                )
                print(
                    "  3. If batches were created, wait for completion or cancel them: "
                    "scripts/cancel_all_batches.py"
                )
                print(f"{'=' * 70}\n")
            elif (
                "AlwaysReasoningModelError" in error_str
                or "always uses reasoning" in error_str
            ):
                print("\n⚠ One or more models cannot be used in instruct mode.")
                print("   Use '-thinking' suffix for always-reasoning models.\n")
            else:
                print(f"\n⚠ Error in evaluations: {error_str[:200]}\n")
            # Try to collect any partial results from log directory
            _collect_partial_results(shared_log_dir, eval_logs, descriptions)

    # Handle case where no evaluations completed
    if len(eval_logs) == 0:
        print(f"\n{'=' * 70}")
        print("SWEEP EXPERIMENT SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total evaluations: {total_evals}")
        print("⚠ No evaluations succeeded! Check errors above and retry.")
        print(f"{'=' * 70}\n")
        return

    # Check for any failures
    successful = []
    failed = []

    # Match eval_logs to descriptions - use min to handle partial results
    for idx, eval_log in enumerate(eval_logs):
        # Try to find matching description
        description = (
            descriptions[idx] if idx < len(descriptions) else f"Evaluation {idx}"
        )

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
    print(f"Completed: {len(eval_logs)}")
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

    # Preprocess sys.argv to handle -set before argparse sees it
    # argparse treats -set as a flag, so we need to escape it
    import sys

    if "--model_names" in sys.argv:
        model_names_idx = sys.argv.index("--model_names")
        # Find where model_names arguments end (next flag or end of args)
        for i in range(model_names_idx + 1, len(sys.argv)):
            if sys.argv[i] == "-set" and (
                i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
            ):
                # Replace -set with a placeholder that doesn't start with -
                sys.argv[i] = "SET_PLACEHOLDER"

    if "--generator_models" in sys.argv:
        gen_models_idx = sys.argv.index("--generator_models")
        # Find where generator_models arguments end (next flag or end of args)
        for i in range(gen_models_idx + 1, len(sys.argv)):
            if sys.argv[i] == "-set" and (
                i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith("--")
            ):
                # Replace -set with a placeholder that doesn't start with -
                sys.argv[i] = "SET_PLACEHOLDER"

    parser = argparse.ArgumentParser(
        description="Sweep run experiments across multiple models and treatments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,  # Disable abbreviation to allow -set as a value
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
        help="List of model names to evaluate (Evaluators) (e.g., 'haiku-3-5 gpt-4.1') or model sets (e.g., '-set cot' for gen_cot set). "
        "Can mix individual models and sets.",
    )
    parser.add_argument(
        "--generator_models",
        type=str,
        nargs="+",
        help="Optional: List of model names to generate data from (Generators). If provided, evaluators will be paired against these models. "
        "Supports individual models and sets (e.g., '-set cot').",
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
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        default=False,
        help="Skip confirmation prompt and run immediately",
    )
    parser.add_argument(
        "--skip_batch_in_progress",
        action="store_true",
        default=False,
        help="Skip evaluations when batch job is in progress (default: re-run, as this usually indicates a stuck eval file)",
    )

    args = parser.parse_args()

    # Restore -set from placeholder
    if args.model_names:
        args.model_names = [
            arg.replace("SET_PLACEHOLDER", "-set") for arg in args.model_names
        ]

    if args.generator_models:
        args.generator_models = [
            arg.replace("SET_PLACEHOLDER", "-set") for arg in args.generator_models
        ]

    # Expand model set references (e.g., '-set cot' -> list of models)
    expanded_model_names = expand_model_names(args.model_names)
    
    expanded_generator_models = None
    if args.generator_models:
        expanded_generator_models = expand_model_names(args.generator_models)

    # Parse batch argument
    # Check environment variable if --batch wasn't explicitly provided
    batch_value = args.batch
    if batch_value is False and os.getenv("SWEEP_BATCH") == "1":
        batch_value = True
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
        yes=args.yes,
        generator_models=expanded_generator_models,
        skip_batch_in_progress=args.skip_batch_in_progress,
    )
