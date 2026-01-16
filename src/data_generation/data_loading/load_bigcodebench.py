#!/usr/bin/env python3
"""
Load BigCodeBench dataset from HuggingFace and save to data/bigcodebench/input.json

BigCodeBench is a benchmark for code generation tasks, containing programming problems
with test cases and canonical solutions. Useful for testing self-recognition on code
generation tasks.

Dataset: https://huggingface.co/datasets/bigcode/bigcodebench
Paper: BigCodeBench: Benchmarking Code Generation with Diverse Functionality and Language

Usage:
    # Load first 100 samples from default split
    uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100

    # Load specific range of samples (10-50 inclusive)
    uv run src/data_generation/data_loading/load_bigcodebench.py --range=10-50

    # Load 200 samples with custom dataset name
    uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=200 --dataset_name=bigcodebench_200

    # Load from specific version
    uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100 --version=v0.1.0_hf

    # Use complete_prompt instead of instruct_prompt
    uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100 --prompt_type=complete

Note: Must specify either --num_samples or --range (one required, mutually exclusive)
"""

import argparse
import uuid

from datasets import load_dataset

from src.helpers.constants import MY_DATASET_NAMESPACE
from src.helpers.utils import save_json, data_dir


def load_bigcodebench_data(
    num_samples: int = None,
    sample_range: tuple[int, int] = None,
    version: str = "v0.1.0_hf",
    prompt_type: str = "instruct",
    dataset_name: str = "bigcodebench",
) -> None:
    """
    Load BigCodeBench dataset from HuggingFace and save to input.json format.

    Args:
        num_samples: Number of samples to include from the start
        sample_range: Tuple of (start, end) indices to select specific range
        version: Dataset version to use (default: 'v0.1.0_hf')
        prompt_type: Type of prompt to use ('instruct' or 'complete', default: 'instruct')
        dataset_name: Name of subdirectory to save data in (default: 'bigcodebench')
    """
    if num_samples is None and sample_range is None:
        raise ValueError("Must specify either --num_samples or --range")

    if num_samples is not None and sample_range is not None:
        raise ValueError("Cannot specify both --num_samples and --range")

    if prompt_type not in ["instruct", "complete"]:
        raise ValueError("prompt_type must be 'instruct' or 'complete'")

    print(
        f"Loading BigCodeBench dataset (version: {version}, prompt_type: {prompt_type})..."
    )

    # Load dataset from HuggingFace
    # The dataset uses a subset 'default' and returns a DatasetDict with version splits
    print("  Downloading dataset from HuggingFace...")
    dataset_dict = load_dataset("bigcode/bigcodebench", "default")

    # Check if it's a DatasetDict and get the requested version split
    if hasattr(dataset_dict, "keys"):
        available_versions = list(dataset_dict.keys())
        if version not in available_versions:
            raise ValueError(
                f"Version '{version}' not found. Available versions: {available_versions}"
            )
        dataset = dataset_dict[version]
    else:
        # If it's not a DatasetDict, use it directly
        dataset = dataset_dict

    original_size = len(dataset)
    print(f"  Loaded {original_size} samples from dataset")

    # Select samples based on num_samples or range
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    elif sample_range is not None:
        start, end = sample_range
        if start < 0 or end >= len(dataset) or start > end:
            raise ValueError(
                f"Invalid range {start}-{end}. Dataset has {len(dataset)} samples (indices 0-{len(dataset) - 1})"
            )
        dataset = dataset.select(range(start, end + 1))  # +1 to make end inclusive

    print(f"Processing {len(dataset)} samples...")

    # Create input.json mapping: UUID -> prompt
    input_dict = {}
    skipped_count = 0

    # Determine which prompt field to use
    prompt_field = "instruct_prompt" if prompt_type == "instruct" else "complete_prompt"

    for idx, sample in enumerate(dataset):
        # Get the prompt based on prompt_type
        if prompt_field not in sample:
            # Fallback: try the other prompt type if available
            fallback_field = (
                "complete_prompt" if prompt_type == "instruct" else "instruct_prompt"
            )
            if fallback_field in sample:
                prompt = sample[fallback_field]
                print(
                    f"  Warning: {prompt_field} not found for sample {idx}, using {fallback_field}"
                )
            else:
                skipped_count += 1
                continue
        else:
            prompt = sample[prompt_field]

        if not prompt or not prompt.strip():
            skipped_count += 1
            continue

        # Generate UUID from prompt for reproducibility
        sample_uuid = str(uuid.uuid3(MY_DATASET_NAMESPACE, prompt))

        # Store in input dict (skip if duplicate UUID)
        if sample_uuid not in input_dict:
            input_dict[sample_uuid] = prompt
        else:
            skipped_count += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples (empty prompt or duplicate UUIDs)")

    # Save to data/{dataset_name}/input.json
    output_path = data_dir() / dataset_name / "input.json"
    save_json(input_dict, output_path)

    print(f"\n✓ Successfully saved {len(input_dict)} samples to {output_path}")
    print("\nDataset info:")
    print("  - Source: bigcode/bigcodebench (HuggingFace)")
    print(f"  - Version: {version}")
    print(f"  - Prompt type: {prompt_type} ({prompt_field})")
    print(f"  - Samples: {len(input_dict)}")
    print("  - Content: Code generation prompts from BigCodeBench")
    print("  - Task: Code generation")


def parse_range(range_str: str) -> tuple[int, int]:
    """Parse a range string like '5-15' into a tuple of (start, end)."""
    try:
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError
        start, end = int(parts[0]), int(parts[1])
        return (start, end)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(
            f"Range must be in format 'START-END' (e.g., '5-15'), got: {range_str}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Load BigCodeBench dataset from HuggingFace",
        epilog="Examples:\n"
        "  %(prog)s --num_samples=100\n"
        "  %(prog)s --range=10-50\n"
        "  %(prog)s --num_samples=200 --dataset_name=bigcodebench_200\n"
        "  %(prog)s --num_samples=100 --version=v0.1.0_hf\n"
        "  %(prog)s --num_samples=100 --prompt_type=complete",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Make num_samples and range mutually exclusive, with one required
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--num_samples",
        type=int,
        help="Number of samples to load from the start (e.g., --num_samples=100)",
    )
    group.add_argument(
        "--range",
        type=parse_range,
        metavar="START-END",
        help="Range of samples to load (e.g., --range=5-15 for samples 5 through 15 inclusive)",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default="bigcodebench",
        help="Name of subdirectory to save data in (default: bigcodebench)",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v0.1.0_hf",
        help="Dataset version/split to use (default: v0.1.0_hf). "
        "Other versions may include: v0.1.1, v0.1.2, v0.1.3, v0.1.4",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="instruct",
        choices=["instruct", "complete"],
        help="Type of prompt to use: 'instruct' uses instruct_prompt (default), "
        "'complete' uses complete_prompt",
    )

    args = parser.parse_args()

    load_bigcodebench_data(
        num_samples=args.num_samples,
        sample_range=args.range,
        version=args.version,
        prompt_type=args.prompt_type,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
