"""Data loading utilities for pairwise self-recognition tasks."""

from typing import List, Dict, Any

from src.helpers.utils import data_dir, load_json, load_rollout_json


def load_dataset_pairwise(
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_name: str,
    file_name_1: str,
    file_name_2: str,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset for pairwise comparison.

    Creates TWO samples per UUID - one with each ordering (treatment1-first, treatment2-first).
    This ensures we test both presentation orders.

    Args:
        treatment_name_1: Name of the first treatment being compared
        treatment_name_2: Name of the second treatment being compared
        dataset_name: Name of the dataset directory
        file_name_1: Generation/treatment identifier for the first treatment (e.g., "temp0", "typo_s1", "default")
        file_name_2: Generation/treatment identifier for the second treatment

    Returns:
        List of sample dictionaries (2 per UUID) containing:
        - content: The article/question text
        - output1: First output
        - output2: Second output
        - metadata: Dict with correct_answer, ordering, and other info
    """
    # Load content (articles or questions)
    contents = load_json(data_dir() / dataset_name / "input.json")

    outputs_1 = load_rollout_json(dataset_name, treatment_name_1, file_name_1)
    outputs_2 = load_rollout_json(dataset_name, treatment_name_2, file_name_2)

    # Create TWO samples per UUID - one for each ordering
    samples = []
    skipped_uuids = []

    for uuid in contents.keys():
        if uuid not in outputs_1 or uuid not in outputs_2:
            # Skip if either output is missing
            # Puria TODO: Review this error handling
            print(f"Skipping UUID {uuid} because it is missing in one or both outputs")
            skipped_uuids.append(uuid)
            continue

        content = contents[uuid]
        output_1 = outputs_1[uuid]
        output_2 = outputs_2[uuid]

        metadata = {
            "uuid": uuid,
            "treatment_name_1": treatment_name_1,
            "treatment_name_2": treatment_name_2,
            "file_name_1": file_name_1,
            "file_name_2": file_name_2,
        }

        # Sample 1: Treatment 1 output first (position 1)
        samples.append(
            {
                "content": content,
                "output1": output_1,
                "output2": output_2,
                "metadata": {
                    **metadata,
                    "correct_answer": "1",
                },
            }
        )

        # Sample 2: Treatment 2 output first (position 1)
        samples.append(
            {
                "content": content,
                "output1": output_2,
                "output2": output_1,
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


def load_dataset_individual(
    treatment_name: str,
    dataset_name: str,
    file_name: str,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset for individual evaluation.

    Creates a single sample per UUID.

    Args:
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory
        file_name: Generation/treatment identifier (e.g., "temp0", "typo_s1", "default")

    Returns:
        List of sample dictionaries containing:
        - content: The article/question text
        - output: The model's response
        - metadata: Dict with correct_answer and other info
    """
    # Load content (articles or questions)
    contents = load_json(data_dir() / dataset_name / "input.json")

    outputs = load_rollout_json(dataset_name, treatment_name, file_name)

    # Create TWO samples per UUID for the varying indicator variable (correct_answer = 1 or 2)
    samples = []
    skipped_uuids = []

    for uuid in contents.keys():
        if uuid not in outputs:
            # Skip if output is missing
            # Puria TODO: Review this error handling
            print(f"Skipping UUID {uuid} because it is missing in the output")
            skipped_uuids.append(uuid)
            continue

        content = contents[uuid]
        output = outputs[uuid]

        metadata = {
            "uuid": uuid,
            "treatment_name": treatment_name,
            "file_name": file_name,
        }

        # Sample 1: Varying indicator variable (correct_answer = 1)
        samples.append(
            {
                "content": content,
                "output": output,
                "metadata": {
                    **metadata,
                    "correct_answer": "1",
                },
            }
        )

        # Sample 2: Varying indicator variable (correct_answer = 2)
        samples.append(
            {
                "content": content,
                "output": output,
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
