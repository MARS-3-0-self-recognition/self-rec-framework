"""Dataset loading for pairwise recognition.

Uses UUID alignment to match content with outputs from different models.
Creates 2 samples per UUID (swapped ordering) to test ordering bias.
"""

from pathlib import Path
from typing import List
import json

from inspect_ai.dataset import Sample


def load_pairwise_dataset(
    dataset_name: str,
    model_name: str,
    alternative_model_name: str,
    model_gen_string: str,
    alternative_gen_string: str,
    content_file: str = "articles.json",
    output_file: str = "summaries.json",
) -> List[Sample]:
    """Load pairwise dataset using UUID alignment.

    Loads content and outputs from different models, aligning them by UUID.
    Creates 2 samples per UUID with swapped ordering to test both positions.

    Expected directory structure:
        data/
        └── {dataset_name}/
            ├── {content_file}              # {uuid: content_text}
            ├── {model_name}/
            │   └── {model_gen_string}_{output_file}    # {uuid: output_text}
            └── {alternative_model_name}/
                └── {alternative_gen_string}_{output_file}

    Args:
        dataset_name: Dataset directory name
        model_name: Model being evaluated (short name, e.g., 'claude')
        alternative_model_name: Alternative model (short name, e.g., 'gpt4')
        model_gen_string: Generation config string (e.g., 'temp0', 'temp1')
        alternative_gen_string: Alternative generation config string
        content_file: Content filename (e.g., 'articles.json', 'questions.json')
        output_file: Output filename (e.g., 'summaries.json', 'answers.json')

    Returns:
        List of Inspect Samples with metadata.
        Each UUID produces 2 samples (one for each ordering).
    """
    from src.utils.config import resolve_model_name

    # Resolve model names to full paths for directory structure
    model_full = resolve_model_name(model_name)
    alt_full = resolve_model_name(alternative_model_name)

    data_dir = Path("data") / dataset_name

    # Load content (articles or questions)
    content_path = data_dir / content_file
    if not content_path.exists():
        raise FileNotFoundError(f"Content file not found: {content_path}")

    with open(content_path) as f:
        contents = json.load(f)

    # Load model outputs
    model_output_path = data_dir / model_full / f"{model_gen_string}_{output_file}"
    alt_output_path = data_dir / alt_full / f"{alternative_gen_string}_{output_file}"

    if not model_output_path.exists():
        raise FileNotFoundError(f"Model output file not found: {model_output_path}")
    if not alt_output_path.exists():
        raise FileNotFoundError(f"Alternative output file not found: {alt_output_path}")

    with open(model_output_path) as f:
        model_outputs = json.load(f)

    with open(alt_output_path) as f:
        alt_outputs = json.load(f)

    # Create samples with UUID alignment
    samples = []
    skipped = []

    for uuid in contents.keys():
        # Skip if either output is missing
        if uuid not in model_outputs or uuid not in alt_outputs:
            skipped.append(uuid)
            continue

        # Base metadata shared by both orderings
        base_metadata = {
            "uuid": uuid,
            "content": contents[uuid],
            "model_name": model_name,
            "alternative_model_name": alternative_model_name,
            "model_gen_string": model_gen_string,
            "alternative_gen_string": alternative_gen_string,
            "dataset_name": dataset_name,
        }

        # Sample 1: Model output in position 1
        samples.append(
            Sample(
                input="",  # Protocol will construct the prompt
                target="1",
                metadata={
                    **base_metadata,
                    "output1": model_outputs[uuid],
                    "output2": alt_outputs[uuid],
                    "correct_answer": "1",
                    "ordering": "model_first",
                },
            )
        )

        # Sample 2: Alternative output in position 1 (swapped)
        samples.append(
            Sample(
                input="",  # Protocol will construct the prompt
                target="2",
                metadata={
                    **base_metadata,
                    "output1": alt_outputs[uuid],
                    "output2": model_outputs[uuid],
                    "correct_answer": "2",
                    "ordering": "alternative_first",
                },
            )
        )

    if skipped:
        print(f"Warning: Skipped {len(skipped)} UUIDs with missing outputs")
        if len(skipped) <= 10:
            print(f"  Skipped UUIDs: {skipped}")
        else:
            print(f"  First 10 skipped: {skipped[:10]}")

    print(
        f"Loaded {len(samples)} samples ({len(samples)//2} UUID pairs) from {dataset_name}"
    )

    return samples
