"""
Run pairwise judging evaluations for self-recognition experiments.
"""

import argparse
from pathlib import Path
from dotenv import load_dotenv

from inspect_ai import eval
from src.protocols.pairwise.tasks import (
    conversational_self_recognition,
)

from src.helpers.utils import data_dir, rollout_json_file_path


def parse_dataset_path(dataset_path: str):
    """
    Parse a dataset path to extract components.

    Expected format: data/dataset_name/treatment_name/file.json
    Example: data/wikisum_train_1-20/haiku-3-5/wikisum_config.json
    Also handles: dataset_name/treatment_name/file.json (without data/ prefix)

    Returns:
        tuple: (dataset_name, treatment_name, file_name)
    """
    path = Path(dataset_path)
    parts = list(path.parts)

    # Remove 'data/' prefix if present
    if parts and parts[0] == "data":
        parts = parts[1:]

    if len(parts) < 3:
        raise ValueError(
            f"Invalid dataset path format: {dataset_path}. "
            f"Expected: data/dataset_name/treatment_name/file.json "
            f"or dataset_name/treatment_name/file.json"
        )

    dataset_name = parts[0]
    treatment_name = parts[1]
    file_name = parts[2]

    # Remove .json extension if present
    if file_name.endswith(".json"):
        file_name = file_name[:-5]

    return dataset_name, treatment_name, file_name


def check_rollout_exists(
    dataset_name: str, treatment_name: str, file_name: str
) -> bool:
    """Check if rollout JSON already exists."""
    rollout_path = rollout_json_file_path(dataset_name, treatment_name, file_name)
    return rollout_path.exists()


def check_eval_exists(log_dir: Path) -> bool:
    """Check if eval log directory exists and contains .eval files."""
    if not log_dir.exists():
        return False
    eval_files = list(log_dir.glob("*.eval"))
    return len(eval_files) > 0


def get_judge_log_dir(
    dataset_name: str,
    task_name: str,
    model_name: str,
    treatment_name_1: str,
    treatment_name_2: str,
    config_name: str,
) -> Path:
    """Construct path for judging evaluation logs."""
    return (
        data_dir()
        / dataset_name
        / "judge_logs"
        / task_name
        / f"{model_name}_eval_{treatment_name_1}_vs_{treatment_name_2}_in_{config_name}"
    )


def run_judging_evals(
    model_name: str,
    dataset_path_1: str,
    dataset_path_2: str,
    config_name: str,
) -> None:
    """Run pairwise judging evaluations."""
    print("\n=== RUNNING JUDGING EVALUATIONS ===")

    # Parse dataset paths
    dataset_name_1, treatment_name_1, file_name_1 = parse_dataset_path(dataset_path_1)
    dataset_name_2, treatment_name_2, file_name_2 = parse_dataset_path(dataset_path_2)

    # Use dataset_name from path 1
    dataset_name = dataset_name_1

    # Check prerequisites
    if not check_rollout_exists(dataset_name_1, treatment_name_1, file_name_1):
        print(f"✗ Missing rollout for {dataset_path_1}, skipping")
        return
    if not check_rollout_exists(dataset_name_2, treatment_name_2, file_name_2):
        print(f"✗ Missing rollout for {dataset_path_2}, skipping")
        return

    print(
        f"\n--- Judge: {model_name} | Treatment 1: {treatment_name_1} | Treatment 2: {treatment_name_2} ---"
    )

    # Run comparison task
    # run_single_judging_task(
    #     task_name="comparison",
    #     task_fn=comparison_self_recognition,
    #     model_name=model_name,
    #     treatment_name_1=treatment_name_1,
    #     treatment_name_2=treatment_name_2,
    #     dataset_file_name_1=file_name_1,
    #     dataset_file_name_2=file_name_2,
    #     dataset_name=dataset_name,
    #     config_name=config_name,
    # )

    # Run conversational task
    run_single_judging_task(
        task_name="conversational",
        task_fn=conversational_self_recognition,
        model_name=model_name,
        treatment_name_1=treatment_name_1,
        treatment_name_2=treatment_name_2,
        dataset_file_name_1=file_name_1,
        dataset_file_name_2=file_name_2,
        dataset_name=dataset_name,
        config_name=config_name,
    )


def run_single_judging_task(
    task_name: str,
    task_fn,
    model_name: str,
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_file_name_1: str,
    dataset_file_name_2: str,
    dataset_name: str,
    config_name: str,
) -> None:
    """Run a single judging task (comparison or conversational)."""
    log_dir = get_judge_log_dir(
        dataset_name,
        task_name,
        model_name,
        treatment_name_1,
        treatment_name_2,
        config_name,
    )

    if check_eval_exists(log_dir):
        print(f"  ✓ {task_name}: already evaluated, skipping")
        return

    print(f"  Running {task_name} task...")
    task = task_fn(
        model_name=model_name,
        treatment_name_1=treatment_name_1,
        treatment_name_2=treatment_name_2,
        dataset_name=dataset_name,
        dataset_file_name_1=dataset_file_name_1,
        dataset_file_name_2=dataset_file_name_2,
        config_name=config_name,
    )

    eval(task, log_dir=str(log_dir))
    print(f"  ✓ {task_name}: completed")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete self-recognition experiment"
    )
    parser.add_argument(
        "--dataset_path_1",
        type=str,
        required=True,
        help="First dataset path (e.g., 'data/wikisum_train_1-20/haiku-3-5/wikisum_config.json')",
    )
    parser.add_argument(
        "--dataset_path_2",
        type=str,
        required=True,
        help="Second dataset path (e.g., 'data/wikisum_train_1-20/sonnet-3-5/wikisum_config.json')",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        required=True,
        help="Protocol config name (e.g., 'summarisation')",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name for evaluator (e.g., 'haiku-3-5')",
    )

    args = parser.parse_args()

    # Parse dataset paths for display
    dataset_name_1, treatment_name_1, file_name_1 = parse_dataset_path(
        args.dataset_path_1
    )
    dataset_name_2, treatment_name_2, file_name_2 = parse_dataset_path(
        args.dataset_path_2
    )

    print(f"\n{'=' * 60}")
    print("SELF-RECOGNITION EXPERIMENT")
    print(f"{'=' * 60}")
    print(f"Dataset path 1: {args.dataset_path_1}")
    print(
        f"  -> Dataset: {dataset_name_1}, Treatment: {treatment_name_1}, File: {file_name_1}"
    )
    print(f"Dataset path 2: {args.dataset_path_2}")
    print(
        f"  -> Dataset: {dataset_name_2}, Treatment: {treatment_name_2}, File: {file_name_2}"
    )
    print(f"Protocol config: {args.config_name}")
    print(f"Evaluator model: {args.model_name}")
    print(f"{'=' * 60}")

    run_judging_evals(
        args.model_name,
        args.dataset_path_1,
        args.dataset_path_2,
        args.config_name,
    )

    print(f"\n{'=' * 60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
