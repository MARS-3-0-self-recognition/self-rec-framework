#!/usr/bin/env python3
"""
Load PKU-SafeRLHF dataset from HuggingFace and save to data/pku_saferlhf/input.json

PKU-SafeRLHF is a dataset with prompts and safe/unsafe responses, useful for testing
self-recognition on question answering and dialogue tasks.

Dataset: https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF
Contains prompts with multiple responses labeled for safety.

Usage:
    # Load first 100 samples from train set
    uv run data_loader/load_pku_saferlhf.py --num_samples=100 --split=train

    # Load specific range of samples (10-50 inclusive)
    uv run data_loader/load_pku_saferlhf.py --range=10-50 --split=train

    # Load 200 samples with custom dataset name
    uv run data_loader/load_pku_saferlhf.py --num_samples=200 --dataset_name=pku_200

    # Load samples from test split
    uv run data_loader/load_pku_saferlhf.py --range=0-99 --split=test

    # Load from specific subset (alpaca-7b, alpaca2-7b, alpaca3-8b)
    uv run data_loader/load_pku_saferlhf.py --num_samples=100 --subset=alpaca-7b

    # Load only prompts where responses have different safety ratings
    uv run data_loader/load_pku_saferlhf.py --num_samples=100 --filter_safety_mismatch

    # Combine filters: first 50 safety-mismatched prompts from alpaca subset
    uv run data_loader/load_pku_saferlhf.py --num_samples=50 --subset=alpaca-7b --filter_safety_mismatch

    uv run data_loader/load_pku_saferlhf.py --num_samples=20 --subset=default --filter_safety_mismatch --dataset_name=pku_safety-mismatch_1-20

Note: Must specify either --num_samples or --range (one required, mutually exclusive)
"""

import argparse
import uuid

from datasets import load_dataset

from src.helpers.constants import MY_DATASET_NAMESPACE
from src.helpers.utils import save_json, data_dir


def load_pku_saferlhf_data(
    split: str = "train",
    subset: str = "default",
    num_samples: int = None,
    sample_range: tuple[int, int] = None,
    filter_safety_mismatch: bool = False,
    dataset_name: str = "pku_saferlhf",
) -> None:
    """
    Load PKU-SafeRLHF dataset from HuggingFace and save to input.json format.

    Args:
        split: Dataset split to use ('train' or 'test')
        subset: Dataset subset ('default', 'alpaca-7b', 'alpaca2-7b', 'alpaca3-8b')
        num_samples: Number of samples to include from the start
        sample_range: Tuple of (start, end) indices to select specific range
        filter_safety_mismatch: If True, only include prompts where is_response_0_safe != is_response_1_safe
        dataset_name: Name of subdirectory to save data in (default: 'pku_saferlhf')
    """
    if num_samples is None and sample_range is None:
        raise ValueError("Must specify either --num_samples or --range")

    if num_samples is not None and sample_range is not None:
        raise ValueError("Cannot specify both --num_samples and --range")

    filter_msg = " (filtering for safety mismatches)" if filter_safety_mismatch else ""
    print(
        f"Loading PKU-SafeRLHF dataset (subset: {subset}, split: {split}){filter_msg}..."
    )

    # Load dataset from HuggingFace
    dataset = load_dataset("PKU-Alignment/PKU-SafeRLHF", subset, split=split)

    original_size = len(dataset)

    # Filter for safety mismatches if requested
    if filter_safety_mismatch:
        print("Filtering for prompts where response safety ratings differ...")
        filtered_indices = [
            i
            for i, sample in enumerate(dataset)
            if sample["is_response_0_safe"] != sample["is_response_1_safe"]
        ]
        dataset = dataset.select(filtered_indices)
        print(
            f"  Filtered from {original_size} to {len(dataset)} samples with safety mismatches"
        )

    # Select samples based on num_samples or range
    if num_samples is not None:
        dataset = dataset.select(range(min(num_samples, len(dataset))))
    elif sample_range is not None:
        start, end = sample_range
        if start < 0 or end >= len(dataset) or start > end:
            raise ValueError(
                f"Invalid range {start}-{end}. Dataset has {len(dataset)} samples (indices 0-{len(dataset)-1})"
            )
        dataset = dataset.select(range(start, end + 1))  # +1 to make end inclusive

    print(f"Processing {len(dataset)} samples...")

    # Create input.json mapping: UUID -> prompt
    input_dict = {}

    for idx, sample in enumerate(dataset):
        # Use the prompt as the content to be answered
        prompt = sample["prompt"]

        # Generate UUID from prompt for reproducibility
        sample_uuid = str(uuid.uuid3(MY_DATASET_NAMESPACE, prompt))

        # Store in input dict
        input_dict[sample_uuid] = prompt

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")

    # Save to data/{dataset_name}/input.json
    output_path = data_dir() / dataset_name / "input.json"
    save_json(input_dict, output_path)

    print(f"\n✓ Successfully saved {len(input_dict)} samples to {output_path}")
    print("\nDataset info:")
    print("  - Source: PKU-Alignment/PKU-SafeRLHF (HuggingFace)")
    print(f"  - Subset: {subset}")
    print(f"  - Split: {split}")
    print(f"  - Samples: {len(input_dict)}")
    print("  - Content: Prompts/questions from SafeRLHF")
    print("  - Task: Question answering / dialogue")


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
        description="Load PKU-SafeRLHF dataset from HuggingFace",
        epilog="Examples:\n"
        "  %(prog)s --num_samples=100\n"
        "  %(prog)s --range=10-50 --split=test\n"
        "  %(prog)s --num_samples=200 --subset=alpaca-7b --dataset_name=pku_alpaca_200\n"
        "  %(prog)s --num_samples=100 --filter_safety_mismatch\n"
        "  %(prog)s --num_samples=50 --subset=alpaca-7b --filter_safety_mismatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="default",
        choices=["default", "alpaca-7b", "alpaca2-7b", "alpaca3-8b"],
        help="Dataset subset to use (default: default)",
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
        default="pku_saferlhf",
        help="Name of subdirectory to save data in (default: pku_saferlhf)",
    )
    parser.add_argument(
        "--filter_safety_mismatch",
        action="store_true",
        help="Only include prompts where is_response_0_safe != is_response_1_safe",
    )

    args = parser.parse_args()

    load_pku_saferlhf_data(
        split=args.split,
        subset=args.subset,
        num_samples=args.num_samples,
        sample_range=args.range,
        filter_safety_mismatch=args.filter_safety_mismatch,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
