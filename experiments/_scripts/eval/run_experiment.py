"""
Run pairwise judging evaluations for self-recognition experiments.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval
from src.inspect.tasks import get_task_function
from src.inspect.config import load_experiment_config
from src.helpers.utils import data_dir


def parse_dataset_path(dataset_path: str):
    """
    Parse a dataset path to extract components.

    Expected format: data/input/dataset_name/data_subset/treatment/data.json
    Example: data/input/wikisum/training_set_1-20/haiku-3-5/data.json

    Returns:
        tuple: (dataset_name, data_subset, treatment_name)
    """
    if dataset_path is None:
        return None, None, None
    path = Path(dataset_path)
    parts = list(path.parts)

    # Expected: data/input/dataset_name/data_subset/treatment/data.json
    if parts[0] == "data" and parts[1] == "input":
        # Remove 'data/input/' prefix
        parts = parts[2:]
    elif parts[0] == "input":
        # Remove 'input/' prefix
        parts = parts[1:]
    else:
        raise ValueError(
            f"Invalid dataset path format: {dataset_path}. "
            f"Expected: data/input/dataset_name/data_subset/treatment/data.json"
        )

    if len(parts) < 3:
        raise ValueError(
            f"Invalid dataset path format: {dataset_path}. "
            f"Expected: data/input/dataset_name/data_subset/treatment/data.json"
        )

    dataset_name = parts[0]
    data_subset = parts[1]
    treatment_name = parts[2]

    return dataset_name, data_subset, treatment_name


def check_rollout_exists(
    dataset_name: str, data_subset: str, treatment_name: str
) -> bool:
    """Check if data.json already exists."""
    data_path = (
        data_dir() / "input" / dataset_name / data_subset / treatment_name / "data.json"
    )
    return data_path.exists()


def check_eval_exists(log_dir: Path) -> bool:
    """Check if eval log directory exists and contains .eval files."""
    if not log_dir.exists():
        return False
    eval_files = list(log_dir.glob("*.eval"))
    return len(eval_files) > 0


def get_judge_log_dir(
    dataset_name: str,
    data_subset: str,
    experiment_name: str,
    model_name: str,
    treatment_name_control: str,
    treatment_name_treatment: str,
    config_name: str,
) -> Path:
    """
    Construct path for judging evaluation logs.

    Format: data/results/{dataset_name}/{data_subset}/{experiment_name}/{eval_subdir}/
    - Pairwise: {model_name}_eval_on_{control}_vs_{treatment}
    - Individual: {model_name}_eval_on_{evaluated_treatment}
    """
    if treatment_name_treatment is not None:
        eval_subdir = f"{model_name}_eval_on_{treatment_name_control}_vs_{treatment_name_treatment}"
    else:
        eval_subdir = f"{model_name}_eval_on_{treatment_name_control}"

    return (
        data_dir()
        / "results"
        / dataset_name
        / data_subset
        / experiment_name
        / eval_subdir
    )


def run_judging_evals(
    model_name: str,
    dataset_path_control: str,
    experiment_config_path: str,
    dataset_path_treatment: str | None = None,
    is_control: bool = True,
    batch: bool | int | str = False,
    logprobs: bool = False,
) -> None:
    """
    Run judging evaluations (pairwise or individual based on config).

    Args:
        model_name: Model to use as evaluator
        dataset_path_control: Path to control (original) dataset
        experiment_config_path: Path to experiment config
        dataset_path_treatment: Path to treatment (modified) dataset (for pairwise only)
        is_control: Whether we're evaluating control dataset (for individual only)
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
    """
    print("\n=== RUNNING JUDGING EVALUATIONS ===")

    # Extract experiment name from config path
    experiment_name = Path(experiment_config_path).parent.name

    # Parse dataset paths first to get dataset_name
    dataset_name_ctrl, data_subset_ctrl, treatment_name_ctrl = parse_dataset_path(
        dataset_path_control
    )
    dataset_name_treat, data_subset_treat, treatment_name_treat = parse_dataset_path(
        dataset_path_treatment
    )

    # Use dataset info from control path
    dataset_name = dataset_name_ctrl
    data_subset = data_subset_ctrl

    # Load experiment config with dataset_name from path
    exp_config = load_experiment_config(
        experiment_config_path, dataset_name=dataset_name
    )

    # Check control data exists
    if not check_rollout_exists(
        dataset_name_ctrl, data_subset_ctrl, treatment_name_ctrl
    ):
        print(f"✗ Missing data for {dataset_path_control}, skipping")
        return

    # For pairwise tasks, check treatment dataset exists
    if exp_config.is_pairwise():
        if not check_rollout_exists(
            dataset_name_treat, data_subset_treat, treatment_name_treat
        ):
            print(f"✗ Missing data for {dataset_path_treatment}, skipping")
            return
        print(
            f"\n--- Judge: {model_name} | Control: {treatment_name_ctrl} | Treatment: {treatment_name_treat} ---"
        )
        # Single call for pairwise task
        run_single_judging_task(
            exp_config=exp_config,
            model_name=model_name,
            treatment_name_control=treatment_name_ctrl,
            treatment_name_treatment=treatment_name_treat,
            data_subset=data_subset,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            is_control=True,  # Not used for pairwise
            batch=batch,
            logprobs=logprobs,
        )
    else:
        # Individual task - run on specified dataset (either control or treatment)
        # For individual, we always pass the dataset being evaluated as "control" parameter
        # The is_control flag tells the data loader how to set correct answers
        eval_treatment_name = treatment_name_ctrl
        dataset_type = "Control" if is_control else "Treatment"
        print(
            f"\n--- Judge: {model_name} | Evaluating: {eval_treatment_name} ({dataset_type}) ---"
        )
        run_single_judging_task(
            exp_config=exp_config,
            model_name=model_name,
            treatment_name_control=eval_treatment_name,
            treatment_name_treatment=None,
            data_subset=data_subset,
            dataset_name=dataset_name,
            experiment_name=experiment_name,
            is_control=is_control,
            batch=batch,
            logprobs=logprobs,
        )


def run_single_judging_task(
    exp_config,
    model_name: str,
    treatment_name_control: str,
    data_subset: str,
    dataset_name: str,
    experiment_name: str,
    is_control: bool,
    treatment_name_treatment: str | None = None,
    batch: bool | int | str = False,
    logprobs: bool = False,
) -> None:
    """
    Run a single judging task (pairwise or individual).

    Args:
        treatment_name_control: Name of control (original) treatment
        treatment_name_treatment: Name of treatment (modified) - for pairwise only
        is_control: Whether evaluating control (True) or treatment (False) - for individual only
        batch: Enable batch mode for supported providers (OpenAI, Anthropic, Google, Together AI).
               Can be True (default config), int (batch size), or str (path to config file)
        logprobs: Whether to request log probabilities from the model (default: False)
    """
    # Build task name and config name for logging
    task_name = exp_config.config_name_for_logging()
    config_name = task_name

    # Determine log directory
    log_dir = get_judge_log_dir(
        dataset_name,
        data_subset,
        experiment_name,
        model_name,
        treatment_name_control,
        treatment_name_treatment,
        config_name,
    )

    if check_eval_exists(log_dir):
        print(f"  ✓ {experiment_name}: already evaluated, skipping")
        return

    print(f"  Running {experiment_name} task...")

    # Build task name for log filename (using hyphens to match Inspect AI's log filename format)
    if treatment_name_treatment:
        task_name = f"{model_name}-eval-on-{treatment_name_control}-vs-{treatment_name_treatment}"
    else:
        control_or_treatment = "control" if is_control else "treatment"
        task_name = (
            f"{model_name}-eval-on-{treatment_name_control}-{control_or_treatment}"
        )

    # Get task - all branching logic is handled in get_task_function
    task = get_task_function(
        exp_config=exp_config,
        model_name=model_name,
        treatment_name_control=treatment_name_control,
        treatment_name_treatment=treatment_name_treatment,
        dataset_name=dataset_name,
        data_subset=data_subset,
        is_control=is_control,
        task_name=task_name,
        logprobs=logprobs,
    )

    eval(task, log_dir=str(log_dir), batch=batch)
    print(f"  ✓ {experiment_name}: completed")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete self-recognition experiment"
    )
    parser.add_argument(
        "--dataset_path_control",
        type=str,
        required=True,
        help="Control dataset path (original model output) (e.g., 'data/input/wikisum/debug/haiku-3-5/data.json')",
    )
    parser.add_argument(
        "--dataset_path_treatment",
        type=str,
        required=False,
        default=None,
        help="Treatment dataset path (modified output) - required for pairwise tasks (e.g., 'data/input/wikisum/debug/haiku-3-5_typos_S2/data.json')",
    )
    parser.add_argument(
        "--is_control",
        action="store_true",
        default=False,
        help="For individual tasks: whether evaluating control (original) dataset. If false, evaluating treatment dataset.",
    )
    parser.add_argument(
        "--experiment_config",
        type=str,
        required=True,
        help="Path to experiment config file (e.g., 'experiments/01_AT_PW-C_Rec_Pr/config.yaml')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for evaluator (e.g., 'haiku-3-5')",
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
        "--logprobs",
        action="store_true",
        default=False,
        help="Request log probabilities from the model (not currently implemented, will raise NotImplementedError)",
    )

    args = parser.parse_args()

    # Parse batch argument
    batch_value = args.batch
    if batch_value is not False:
        # Try to convert to int if it's a number string
        if isinstance(batch_value, str) and batch_value.isdigit():
            batch_value = int(batch_value)

    # Parse dataset paths for display
    dataset_name_ctrl, data_subset_ctrl, treatment_name_ctrl = parse_dataset_path(
        args.dataset_path_control
    )

    # Load experiment config for display (with dataset_name from path)
    exp_config = load_experiment_config(
        args.experiment_config, dataset_name=dataset_name_ctrl
    )

    print(f"\n{'=' * 60}")
    print("SELF-RECOGNITION EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Experiment config: {args.experiment_config}")
    print(
        f"  -> Tags: {exp_config.tags}, Format: {exp_config.format}, Task: {exp_config.task}"
    )
    print(f"  -> Dataset: {dataset_name_ctrl}, Priming: {exp_config.priming}")
    print(f"Control dataset: {args.dataset_path_control}")
    print(
        f"  -> Dataset: {dataset_name_ctrl}, Subset: {data_subset_ctrl}, Treatment: {treatment_name_ctrl}"
    )

    if args.dataset_path_treatment:
        dataset_name_treat, data_subset_treat, treatment_name_treat = (
            parse_dataset_path(args.dataset_path_treatment)
        )
        print(f"Treatment dataset: {args.dataset_path_treatment}")
        print(
            f"  -> Dataset: {dataset_name_treat}, Subset: {data_subset_treat}, Treatment: {treatment_name_treat}"
        )

    if not exp_config.is_pairwise():
        eval_type = "Control (original)" if args.is_control else "Treatment (modified)"
        print(f"Evaluating: {eval_type}")

    print(f"Evaluator model: {args.model_name}")
    print(f"{'=' * 60}")

    run_judging_evals(
        args.model_name,
        args.dataset_path_control,
        args.experiment_config,
        args.dataset_path_treatment,
        args.is_control,
        batch_value,
        args.logprobs,
    )

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
