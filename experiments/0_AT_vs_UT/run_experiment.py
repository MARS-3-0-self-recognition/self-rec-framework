"""
Run a complete self-recognition experiment:
1. Generate base rollouts for all models
2. Run pairwise judging evaluations for all model pairs
"""

import argparse
from pathlib import Path
from itertools import permutations
from dotenv import load_dotenv

from inspect_ai import eval
from src.protocols.pairwise.tasks import (
    comparison_self_recognition,
    conversational_self_recognition,
)
from src.data_gen.gen import run_base_generation

from src.helpers.utils import data_dir, rollout_json_file_path


def check_rollout_exists(
    dataset_name: str, model_name: str, generation_string: str
) -> bool:
    """Check if rollout JSON already exists."""
    rollout_path = rollout_json_file_path(dataset_name, model_name, generation_string)
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
    alt_model_name: str,
    generation_string: str,
) -> Path:
    """Construct path for judging evaluation logs."""
    return (
        data_dir()
        / dataset_name
        / "judge_logs"
        / task_name
        / f"{model_name}_vs_{alt_model_name}_{generation_string}"
    )


def run_base_generations(
    model_names: list[str],
    generation_string: str,
    dataset_name: str,
    pairwise_config_string: str,
) -> None:
    """Generate base rollouts for all models."""
    print("\n=== RUNNING BASE GENERATIONS ===")

    for model_name in model_names:
        print(f"\n--- Model: {model_name} ---")

        if check_rollout_exists(dataset_name, model_name, generation_string):
            print("✓ Rollout already exists, skipping")
            continue

        run_base_generation(
            model_name=model_name,
            model_generation_string=generation_string,
            dataset_name=dataset_name,
            pairwise_config_string=pairwise_config_string,
        )


def run_judging_evals(
    model_names: list[str],
    generation_string: str,
    dataset_name: str,
    pairwise_config_string: str,
) -> None:
    """Run all pairwise judging evaluations."""
    print("\n=== RUNNING JUDGING EVALUATIONS ===")

    # Generate all ordered pairs (N × (N-1))
    pairs = [(m1, m2) for m1, m2 in permutations(model_names, 2)]

    print(f"\nTotal pairs to evaluate: {len(pairs)}")
    print(f"Total judging evals: {len(pairs) * 2} (comparison + conversational)")

    for model_name, alt_model_name in pairs:
        print(f"\n--- Judge: {model_name} | Alt: {alt_model_name} ---")

        # Check prerequisites
        if not check_rollout_exists(dataset_name, model_name, generation_string):
            print(f"✗ Missing rollout for {model_name}, skipping")
            continue
        if not check_rollout_exists(dataset_name, alt_model_name, generation_string):
            print(f"✗ Missing rollout for {alt_model_name}, skipping")
            continue

        # Run comparison task
        run_single_judging_task(
            task_name="comparison",
            task_fn=comparison_self_recognition,
            model_name=model_name,
            alt_model_name=alt_model_name,
            generation_string=generation_string,
            dataset_name=dataset_name,
            pairwise_config_string=pairwise_config_string,
        )

        # Run conversational task
        run_single_judging_task(
            task_name="conversational",
            task_fn=conversational_self_recognition,
            model_name=model_name,
            alt_model_name=alt_model_name,
            generation_string=generation_string,
            dataset_name=dataset_name,
            pairwise_config_string=pairwise_config_string,
        )


def run_single_judging_task(
    task_name: str,
    task_fn,
    model_name: str,
    alt_model_name: str,
    generation_string: str,
    dataset_name: str,
    pairwise_config_string: str,
) -> None:
    """Run a single judging task (comparison or conversational)."""
    log_dir = get_judge_log_dir(
        dataset_name, task_name, model_name, alt_model_name, generation_string
    )

    if check_eval_exists(log_dir):
        print(f"  ✓ {task_name}: already evaluated, skipping")
        return

    print(f"  Running {task_name} task...")
    task = task_fn(
        model_name=model_name,
        alternative_model_name=alt_model_name,
        dataset_name=dataset_name,
        model_generation_string=generation_string,
        alternative_model_generation_string=generation_string,
        pairwise_config_string=pairwise_config_string,
    )

    eval(task, log_dir=str(log_dir))
    print(f"  ✓ {task_name}: completed")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete self-recognition experiment"
    )
    parser.add_argument(
        "--generation_string",
        type=str,
        required=True,
        help="Generation identifier (e.g., 'simple_config')",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Dataset name (e.g., 'cnn_debug')",
    )
    parser.add_argument(
        "--pairwise_config_string",
        type=str,
        required=True,
        help="Pairwise config name (e.g., 'summarisation')",
    )
    parser.add_argument(
        "--model_names",
        type=str,
        nargs="+",
        required=True,
        help="List of model names (e.g., '3-5-sonnet' '3-5-haiku')",
    )
    parser.add_argument(
        "--skip_generation",
        action="store_true",
        help="Skip base generation step (assumes rollouts exist)",
    )
    parser.add_argument(
        "--skip_judging",
        action="store_true",
        help="Skip judging evaluation step",
    )

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("SELF-RECOGNITION EXPERIMENT")
    print(f"{'='*60}")
    print(f"Dataset: {args.dataset_name}")
    print(f"Generation string: {args.generation_string}")
    print(f"Pairwise config: {args.pairwise_config_string}")
    print(f"Models: {', '.join(args.model_names)}")
    print(f"{'='*60}")

    if not args.skip_generation:
        run_base_generations(
            args.model_names,
            args.generation_string,
            args.dataset_name,
            args.pairwise_config_string,
        )
    else:
        print("\n=== SKIPPING BASE GENERATIONS ===")

    if not args.skip_judging:
        run_judging_evals(
            args.model_names,
            args.generation_string,
            args.dataset_name,
            args.pairwise_config_string,
        )
    else:
        print("\n=== SKIPPING JUDGING EVALUATIONS ===")

    print(f"\n{'='*60}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    load_dotenv(override=True)
    main()
