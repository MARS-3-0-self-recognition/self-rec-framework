"""Data loading utilities for pairwise self-recognition tasks."""

from typing import List, Dict, Any

from src.helpers.utils import data_dir, load_json, load_rollout_json


def load_dataset(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset for pairwise comparison.

    Creates TWO samples per UUID - one with each ordering (model-first, alt-first).
    This ensures we test both presentation orders.

    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model
        dataset_name: Name of the dataset directory
        model_generation_string: Generation identifier for the evaluated model (e.g., "temp0", "default")
        alternative_model_generation_string: Generation identifier for alternative model
        config: PairwiseConfig with file paths and field names

    Returns:
        List of sample dictionaries (2 per UUID) containing:
        - content: The article/question text
        - output1: First output
        - output2: Second output
        - metadata: Dict with correct_answer, ordering, and other info
    """
    # Load content (articles or questions)
    contents = load_json(data_dir() / dataset_name / "input.json")

    model_outputs = load_rollout_json(dataset_name, model_name, model_generation_string)
    alt_outputs = load_rollout_json(
        dataset_name, alternative_model_name, alternative_model_generation_string
    )

    # Create TWO samples per UUID - one for each ordering
    samples = []
    skipped_uuids = []

    for uuid in contents.keys():
        if uuid not in model_outputs or uuid not in alt_outputs:
            # Skip if either output is missing
            # Puria TODO: Review this error handling
            print(f"Skipping UUID {uuid} because it is missing in one or both outputs")
            skipped_uuids.append(uuid)
            continue

        content = contents[uuid]
        model_output = model_outputs[uuid]
        alt_output = alt_outputs[uuid]

        metadata = {
            "uuid": uuid,
            "model_name": model_name,
            "alternative_model_name": alternative_model_name,
            "model_generation_string": model_generation_string,
            "alternative_model_generation_string": alternative_model_generation_string,
        }

        # Sample 1: Model output first (position 1)
        samples.append(
            {
                "content": content,
                "output1": model_output,
                "output2": alt_output,
                "metadata": {
                    **metadata,
                    "correct_answer": "1",
                },
            }
        )

        # Sample 2: Alternative model output first (position 1)
        samples.append(
            {
                "content": content,
                "output1": alt_output,
                "output2": model_output,
                "metadata": {
                    **metadata,
                    "correct_answer": "2",
                },
            }
        )

    if skipped_uuids:
        print(
            f"Warning: Skipped {len(skipped_uuids)} UUIDs with missing outputs: {skipped_uuids[:5]}{'...' if len(skipped_uuids) > 5 else ''}"
        )

    return samples
