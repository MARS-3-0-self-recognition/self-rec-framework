#!/usr/bin/env python3
"""
Load WikiSum dataset from HuggingFace and save to data/wikisum/input.json

WikiSum is a dataset of WikiHow articles with summaries, useful for testing
self-recognition on summarization tasks.

Dataset: https://huggingface.co/datasets/d0rj/wikisum
Paper: WikiSum: Coherent Summarization Dataset for Efficient Human-Evaluation

Usage:
    # Load first 100 samples from validation set
    uv run src/data_generation/data_loading/load_wikisum.py --num_samples=100 --split=validation

    # Load specific range of samples (5-15 inclusive)
    uv run src/data_generation/data_loading/load_wikisum.py --range=5-15 --split=validation

    # Load 200 samples with custom dataset name
    uv run src/data_generation/data_loading/load_wikisum.py --num_samples=200 --dataset_name=wikisum_200

    # Load samples 100-199 from test split
    uv run src/data_generation/data_loading/load_wikisum.py --range=1-30 --split=test --dataset_name=test_set_1-30

Note: Must specify either --num_samples or --range (one required, mutually exclusive)
"""

import argparse
import uuid

from datasets import load_dataset

from src.helpers.constants import MY_DATASET_NAMESPACE
from src.helpers.utils import save_json, data_dir


def load_wikisum_data(
    split: str = "train",
    num_samples: int = None,
    sample_range: tuple[int, int] = None,
    dataset_name: str = "wikisum",
) -> None:
    """
    Load WikiSum dataset from HuggingFace and save to input.json format.

    Args:
        split: Dataset split to use ('train', 'validation', or 'test')
        num_samples: Number of samples to include from the start
        sample_range: Tuple of (start, end) indices to select specific range
        dataset_name: Name of subdirectory to save data in (default: 'wikisum')
    """
    if num_samples is None and sample_range is None:
        raise ValueError("Must specify either --num_samples or --range")

    if num_samples is not None and sample_range is not None:
        raise ValueError("Cannot specify both --num_samples and --range")

    print(f"Loading WikiSum dataset (split: {split})...")

    # Load dataset from HuggingFace
    dataset = load_dataset("d0rj/wikisum", split=split)

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

    # Create input.json mapping: UUID -> article content
    input_dict = {}

    for idx, sample in enumerate(dataset):
        # Use the article text as the content to be summarized
        article = sample["article"]

        # Generate UUID from article content for reproducibility
        sample_uuid = str(uuid.uuid3(MY_DATASET_NAMESPACE, article))

        # Store in input dict
        input_dict[sample_uuid] = article

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1}/{len(dataset)} samples...")

    # Save to data/{dataset_name}/input.json
    output_path = data_dir() / dataset_name / "input.json"
    save_json(input_dict, output_path)

    print(f"\n✓ Successfully saved {len(input_dict)} samples to {output_path}")
    print("\nDataset info:")
    print("  - Source: d0rj/wikisum (HuggingFace)")
    print(f"  - Split: {split}")
    print(f"  - Samples: {len(input_dict)}")
    print("  - Content: WikiHow articles")
    print("  - Task: Article summarization")


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
        description="Load WikiSum dataset from HuggingFace",
        epilog="Examples:\n"
        "  %(prog)s --num_samples=100\n"
        "  %(prog)s --range=5-15 --split=validation\n"
        "  %(prog)s --num_samples=200 --dataset_name=wikisum_200",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "validation", "test"],
        help="Dataset split to use (default: train)",
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
        default="wikisum",
        help="Name of subdirectory to save data in (default: wikisum)",
    )

    args = parser.parse_args()

    load_wikisum_data(
        split=args.split,
        num_samples=args.num_samples,
        sample_range=args.range,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
