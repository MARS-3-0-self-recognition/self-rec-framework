"""Data loading utilities for self-recognition tasks."""  # mistral 24b llama 70b qwen instruct

from typing import List, Dict, Any

from src.helpers.utils import data_dir, load_json
from src.helpers.model_names import is_thinking_model


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
        - cot1: CoT for first output (if thinking model, None otherwise)
        - cot2: CoT for second output (if thinking model, None otherwise)
        - signature1: Signature for first output (if available, None otherwise)
        - signature2: Signature for second output (if available, None otherwise)
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

    # Load CoT data if available (for thinking models)
    cot_1 = None
    cot_2 = None
    signature_1 = None
    signature_2 = None
    cot_path_1 = (
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_1
        / "data_cot.json"
    )
    cot_path_2 = (
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_2
        / "data_cot.json"
    )
    signature_path_1 = (
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_1
        / "data_signatures.json"
    )
    signature_path_2 = (
        data_dir()
        / "input"
        / dataset_name
        / data_subset
        / treatment_name_2
        / "data_signatures.json"
    )

    if is_thinking_model(treatment_name_1) and cot_path_1.exists():
        cot_1 = load_json(cot_path_1)

    if is_thinking_model(treatment_name_2) and cot_path_2.exists():
        cot_2 = load_json(cot_path_2)

    if signature_path_1.exists():
        signature_1 = load_json(signature_path_1)

    if signature_path_2.exists():
        signature_2 = load_json(signature_path_2)

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

        # Get CoT and signatures if available
        cot_1_value = cot_1.get(uuid) if cot_1 and uuid in cot_1 else None
        cot_2_value = cot_2.get(uuid) if cot_2 and uuid in cot_2 else None
        signature_1_value = (
            signature_1.get(uuid) if signature_1 and uuid in signature_1 else None
        )
        signature_2_value = (
            signature_2.get(uuid) if signature_2 and uuid in signature_2 else None
        )

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
                "cot1": cot_1_value,
                "cot2": cot_2_value,
                "signature1": signature_1_value,
                "signature2": signature_2_value,
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
                "cot1": cot_2_value,  # Swapped because output1 is now output_2
                "cot2": cot_1_value,  # Swapped because output2 is now output_1
                "signature1": signature_2_value,  # Swapped
                "signature2": signature_1_value,  # Swapped
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
