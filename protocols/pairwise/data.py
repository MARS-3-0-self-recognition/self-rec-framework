"""Data loading utilities for pairwise self-recognition tasks."""

import json
from pathlib import Path
from typing import List, Dict, Any

from protocols.pairwise.config import PairwiseConfig


def load_dataset(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    model_generation_string: str,
    alternative_model_generation_string: str,
    config: PairwiseConfig
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
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / "data" / dataset_name
    
    # Load content (articles or questions)
    content_path = data_dir / config.content_file
    with open(content_path, 'r') as f:
        contents = json.load(f)
    
    # Load outputs for both models
    # New structure: data/{dataset_name}/{model_name}/{generation_string}_{output_file}
    model_output_path = data_dir / model_name / f"{model_generation_string}_{config.output_file}"
    alt_output_path = data_dir / alternative_model_name / f"{alternative_model_generation_string}_{config.output_file}"
    
    with open(model_output_path, 'r') as f:
        model_outputs = json.load(f)
    
    with open(alt_output_path, 'r') as f:
        alt_outputs = json.load(f)
    
    # Create TWO samples per UUID - one for each ordering
    samples = []
    skipped_uuids = []
    
    for uuid in contents.keys():
        if uuid not in model_outputs or uuid not in alt_outputs:
            # Skip if either output is missing
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
        samples.append({
            "content": content,
            "output1": model_output,
            "output2": alt_output,
            "metadata": {
                **metadata,
                "correct_answer": "1",
            }
        })
        
        # Sample 2: Alternative model output first (position 1)
        samples.append({
            "content": content,
            "output1": alt_output,
            "output2": model_output,
            "metadata": {
                **metadata,
                "correct_answer": "2",
            }
        })
    
    if skipped_uuids:
        print(f"Warning: Skipped {len(skipped_uuids)} UUIDs with missing outputs: {skipped_uuids[:5]}{'...' if len(skipped_uuids) > 5 else ''}")
    
    return samples