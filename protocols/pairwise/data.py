
import json
import random
from pathlib import Path
from typing import List
from inspect_ai.dataset import Sample

from .config import PairwiseConfig


def load_dataset(
    model_name: str,
    alternative_model_name: str,
    dataset_name: str,
    config: PairwiseConfig
) -> List[Sample]:
    """
    Load and prepare dataset for pairwise comparison.
    
    Args:
        model_name: Name of the model being evaluated
        alternative_model_name: Name of the alternative model
        dataset_name: Name of the dataset directory
        config: PairwiseConfig with file paths and field names
        
    Returns:
        List of Sample objects with randomized order
    """
    data_dir = Path("data") / dataset_name
    
    # Load content (articles or questions)
    content_path = data_dir / config.content_file
    with open(content_path, 'r') as f:
        contents = json.load(f)
    
    # Load outputs for both models
    model_output_path = data_dir / f"{model_name}_{config.output_file}"
    alt_output_path = data_dir / f"{alternative_model_name}_{config.output_file}"
    
    with open(model_output_path, 'r') as f:
        model_outputs = json.load(f)
    
    with open(alt_output_path, 'r') as f:
        alt_outputs = json.load(f)
    
    # Create samples for all matching UUIDs
    samples = []
    for uuid in contents.keys():
        if uuid not in model_outputs or uuid not in alt_outputs:
            # Skip if either output is missing
            # TODO: Consider logging warning for missing data
            continue
        
        content = contents[uuid]
        model_output = model_outputs[uuid]
        alt_output = alt_outputs[uuid]
        
        # Randomize order
        if random.random() < 0.5:
            output1, output2 = model_output, alt_output
            correct_answer = "1"
        else:
            output1, output2 = alt_output, model_output
            correct_answer = "2"
        
        # Store metadata for scoring
        metadata = {
            "uuid": uuid,
            "correct_answer": correct_answer,
            "model_name": model_name,
            "alternative_model_name": alternative_model_name,
            "model_is_first": correct_answer == "1"
        }
        
        samples.append({
            "content": content,
            "output1": output1,
            "output2": output2,
            "metadata": metadata
        })
    
    return samples

