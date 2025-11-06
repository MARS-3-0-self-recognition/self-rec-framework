"""Data loading utilities for self-recognition tasks."""  # mistral 24b llama 70b qwen instruct

from typing import List, Dict, Any

from src.helpers.utils import data_dir, load_json


def load_dataset_pairwise(
    treatment_name_1: str,
    treatment_name_2: str,
    dataset_name: str,
    data_subset: str,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset for pairwise comparison.

    Creates TWO samples per UUID - one with each ordering (treatment1-first, treatment2-first).
    This ensures we test both presentation orders.

    Args:
        treatment_name_1: Name of the first treatment being compared
        treatment_name_2: Name of the second treatment being compared
        dataset_name: Name of the dataset directory
        data_subset: Data subset directory (e.g., 'training_set_1-20')

    Returns:
        List of sample dictionaries (2 per UUID) containing:
        - content: The article/question text
        - output1: First output
        - output2: Second output
        - metadata: Dict with correct_answer, ordering, and other info
    """
    # Load content (articles or questions)
    contents = load_json(
        data_dir() / "input" / dataset_name / data_subset / "input.json"
    )

    outputs_1 = load_json(
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_1
        / "data.json"
    )
    outputs_2 = load_json(
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_2
        / "data.json"
    )

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
            "dataset_name": dataset_name,
            "data_subset": data_subset,
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
    data_subset: str,
    is_control: bool = True,
) -> List[Dict[str, Any]]:
    """
    Load and prepare dataset for individual evaluation.

    Creates TWO samples per UUID with varying indicator tokens.

    Args:
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        is_control: Whether this is control (original) or treatment (modified) data

    Returns:
        List of sample dictionaries containing:
        - content: The article/question text
        - output: The model's response
        - metadata: Dict with correct_answer, choice tokens, etc.

    For control datasets: correct_choice_token indicates "yes, this is my original output"
    For treatment datasets: correct_choice_token indicates "no, this contains modified text"
    """
    # Load content (articles or questions)
    contents = load_json(
        data_dir() / "input" / dataset_name / data_subset / "input.json"
    )

    outputs = load_json(
        data_dir() / "input" / dataset_name / data_subset / treatment_name / "data.json"
    )

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
            "dataset_name": dataset_name,
            "data_subset": data_subset,
            "is_control": is_control,
        }

        # Prompt asks: "Answer {correct_choice_token} if this is your original output,
        #                or {incorrect_choice_token} if it contains modified text"
        #
        # For CONTROL data (original output):
        #   - Correct answer = correct_choice_token (it IS the model's original)
        # For TREATMENT data (modified output):
        #   - Correct answer = incorrect_choice_token (it DOES contain modified text)

        # Sample 1: correct_choice_token = "1", incorrect_choice_token = "2"
        if is_control:
            correct_answer_1 = (
                "1"  # Control: "1" means "yes, original" which is correct
            )
        else:
            correct_answer_1 = (
                "2"  # Treatment: "2" means "no, modified" which is correct
            )

        samples.append(
            {
                "content": content,
                "output": output,
                "metadata": {
                    **metadata,
                    "correct_answer": correct_answer_1,
                    "correct_choice_token": "1",
                    "incorrect_choice_token": "2",
                },
            }
        )

        # Sample 2: correct_choice_token = "2", incorrect_choice_token = "1"
        if is_control:
            correct_answer_2 = (
                "2"  # Control: "2" means "yes, original" which is correct
            )
        else:
            correct_answer_2 = (
                "1"  # Treatment: "1" means "no, modified" which is correct
            )

        samples.append(
            {
                "content": content,
                "output": output,
                "metadata": {
                    **metadata,
                    "correct_answer": correct_answer_2,
                    "correct_choice_token": "2",
                    "incorrect_choice_token": "1",
                },
            }
        )

    if skipped_uuids:
        print(
            f"Warning: Skipped {len(skipped_uuids)} UUIDs with missing outputs: {skipped_uuids[:5]}{'...' if len(skipped_uuids) > 5 else ''}"
        )

    return samples
