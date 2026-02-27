#!/usr/bin/env python3
"""
Load ShareGPT52K dataset from HuggingFace and save to data/sharegpt/input.json

ShareGPT52K is a collection of approximately 90,000 conversations scraped via the ShareGPT API.
These conversations include both user prompts and responses from OpenAI's ChatGPT.

Dataset: https://huggingface.co/datasets/RyokoAI/ShareGPT52K
Contains realistic human-AI conversations useful for testing self-recognition on dialogue tasks.

Usage:
    # Load first 100 samples
    uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=100

    # Load specific range of samples (10-50 inclusive)
    uv run src/data_generation/data_loading/load_sharegpt.py --range=10-50

    # Load 200 samples with custom dataset name
    uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=200 --dataset_name=sharegpt_200

    # Filter for conversations with minimum length
    uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=100 --min_conversation_length=2

Note: Must specify either --num_samples or --range (one required, mutually exclusive)
"""

import argparse
import json
import uuid
import re

from huggingface_hub import hf_hub_download

from src.helpers.constants import MY_DATASET_NAMESPACE
from src.helpers.utils import save_json, data_dir


def clean_html(text: str) -> str:
    """
    Remove HTML tags from text (ShareGPT responses contain HTML).

    Args:
        text: Text potentially containing HTML

    Returns:
        Text with HTML tags removed
    """
    # Remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # Decode common HTML entities
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = text.replace("&#39;", "'")
    # Clean up extra whitespace
    text = " ".join(text.split())
    return text.strip()


def extract_first_user_prompt(conversations: list) -> str | None:
    """
    Extract the first user message from a conversation.

    Args:
        conversations: List of conversation turns with 'from' and 'value' fields

    Returns:
        First user message text (cleaned of HTML), or None if no user message found
    """
    for turn in conversations:
        if turn.get("from") == "human":
            value = turn.get("value", "")
            if value:
                return clean_html(value)
    return None


def load_sharegpt_data(
    num_samples: int = None,
    sample_range: tuple[int, int] = None,
    min_conversation_length: int = 1,
    dataset_name: str = "sharegpt",
) -> None:
    """
    Load ShareGPT52K dataset from HuggingFace and save to input.json format.

    Args:
        num_samples: Number of samples to include from the start
        sample_range: Tuple of (start, end) indices to select specific range
        min_conversation_length: Minimum number of conversation turns to include (default: 1)
        dataset_name: Name of subdirectory to save data in (default: 'sharegpt')
    """
    if num_samples is None and sample_range is None:
        raise ValueError("Must specify either --num_samples or --range")

    if num_samples is not None and sample_range is not None:
        raise ValueError("Cannot specify both --num_samples and --range")

    print("Loading ShareGPT52K dataset...")

    # Download and load raw JSON files directly (avoiding Arrow conversion issues)
    # The dataset has multiple JSON files: sg_90k_part1.json, sg_90k_part2.json, etc.
    print("  Downloading dataset files from HuggingFace...")

    # Try to download all part files
    all_samples = []
    part_num = 1

    while True:
        try:
            filename = f"sg_90k_part{part_num}.json"
            file_path = hf_hub_download(
                repo_id="RyokoAI/ShareGPT52K",
                filename=filename,
                repo_type="dataset",
            )

            print(f"  Loading {filename}...")
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
                # Try to parse as JSON array first
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        all_samples.extend(data)
                    else:
                        # Single object
                        all_samples.append(data)
                except json.JSONDecodeError:
                    # Try JSONL format (one JSON object per line)
                    for line in content.split("\n"):
                        line = line.strip()
                        if line:
                            try:
                                sample = json.loads(line)
                                all_samples.append(sample)
                            except json.JSONDecodeError:
                                continue  # Skip malformed lines

            part_num += 1
        except Exception:
            # No more part files
            break

    if not all_samples:
        raise ValueError(
            "No samples found in dataset. The dataset structure may have changed."
        )

    original_size = len(all_samples)
    print(f"  Loaded {original_size} samples from {part_num - 1} file(s)")

    # Filter for minimum conversation length if requested
    if min_conversation_length > 1:
        print(
            f"Filtering for conversations with at least {min_conversation_length} turns..."
        )
        filtered_samples = [
            sample
            for sample in all_samples
            if isinstance(sample.get("conversations"), list)
            and len(sample.get("conversations", [])) >= min_conversation_length
        ]
        all_samples = filtered_samples
        print(
            f"  Filtered from {original_size} to {len(all_samples)} samples with >= {min_conversation_length} turns"
        )

    # Select samples based on num_samples or range
    if num_samples is not None:
        all_samples = all_samples[: min(num_samples, len(all_samples))]
    elif sample_range is not None:
        start, end = sample_range
        if start < 0 or end >= len(all_samples) or start > end:
            raise ValueError(
                f"Invalid range {start}-{end}. Dataset has {len(all_samples)} samples (indices 0-{len(all_samples)-1})"
            )
        all_samples = all_samples[start : end + 1]  # +1 to make end inclusive

    print(f"Processing {len(all_samples)} samples...")

    # Create input.json mapping: UUID -> prompt
    input_dict = {}
    skipped_count = 0

    for idx, sample in enumerate(all_samples):
        conversations = sample.get("conversations", [])

        # Skip if conversations is not a list
        if not isinstance(conversations, list):
            skipped_count += 1
            continue

        # Extract first user message as the prompt
        prompt = extract_first_user_prompt(conversations)

        if prompt is None or not prompt.strip():
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
            print(f"  Processed {idx + 1}/{len(all_samples)} samples...")

    if skipped_count > 0:
        print(f"  Skipped {skipped_count} samples (no user message or duplicate UUIDs)")

    # Save to data/{dataset_name}/input.json
    output_path = data_dir() / dataset_name / "input.json"
    save_json(input_dict, output_path)

    print(f"\n✓ Successfully saved {len(input_dict)} samples to {output_path}")
    print("\nDataset info:")
    print("  - Source: RyokoAI/ShareGPT52K (HuggingFace)")
    print(f"  - Samples: {len(input_dict)}")
    print("  - Content: First user prompts from ShareGPT conversations")
    print("  - Task: Dialogue / conversation")


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
        description="Load ShareGPT52K dataset from HuggingFace",
        epilog="Examples:\n"
        "  %(prog)s --num_samples=100\n"
        "  %(prog)s --range=10-50\n"
        "  %(prog)s --num_samples=200 --dataset_name=sharegpt_200\n"
        "  %(prog)s --num_samples=100 --min_conversation_length=2",
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
        default="sharegpt",
        help="Name of subdirectory to save data in (default: sharegpt)",
    )
    parser.add_argument(
        "--min_conversation_length",
        type=int,
        default=1,
        help="Minimum number of conversation turns to include (default: 1)",
    )

    args = parser.parse_args()

    load_sharegpt_data(
        num_samples=args.num_samples,
        sample_range=args.range,
        min_conversation_length=args.min_conversation_length,
        dataset_name=args.dataset_name,
    )


if __name__ == "__main__":
    main()
