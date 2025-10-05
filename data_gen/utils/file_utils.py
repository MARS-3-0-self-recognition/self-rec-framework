"""File I/O utilities for data generation."""

import json
from pathlib import Path
from typing import Dict, List, Any


def load_content_file(data_dir: Path, output_type: str) -> Dict[str, str]:
    """
    Load content file (articles or questions).
    
    Args:
        data_dir: Path to dataset directory (e.g., data/cnn)
        output_type: "summaries" or "answers"
        
    Returns:
        Dict mapping uuid to content text
    """
    if output_type == "summaries":
        content_file = "articles.json"
    elif output_type == "answers":
        content_file = "questions.json"
    else:
        raise ValueError(f"output_type must be 'summaries' or 'answers', got {output_type}")
    
    content_path = data_dir / content_file
    if not content_path.exists():
        raise FileNotFoundError(f"Content file not found: {content_path}")
    
    with open(content_path, 'r') as f:
        return json.load(f)


def write_jsonl(data: List[dict], output_path: Path):
    """
    Write list of dicts to JSONL file.
    
    Args:
        data: List of dictionaries
        output_path: Where to write JSONL
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def read_jsonl(input_path: Path) -> List[dict]:
    """
    Read JSONL file into list of dicts.
    
    Args:
        input_path: Path to JSONL file
        
    Returns:
        List of dictionaries
    """
    results = []
    with open(input_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def get_output_filename(output_type: str) -> str:
    """
    Get output filename based on type.
    
    Args:
        output_type: "summaries" or "answers"
        
    Returns:
        Filename string
    """
    if output_type == "summaries":
        return "summaries.json"
    elif output_type == "answers":
        return "answers.json"
    else:
        raise ValueError(f"output_type must be 'summaries' or 'answers', got {output_type}")


def ensure_output_directory(data_dir: Path, model_name: str) -> Path:
    """
    Create model subdirectory if needed and return path.
    
    Args:
        data_dir: Path to dataset directory (e.g., data/cnn)
        model_name: Model name with -- instead of / (e.g., openai--gpt-4)
        
    Returns:
        Path to model directory
    """
    model_dir = data_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def save_json(data: Any, output_path: Path):
    """Save data as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(input_path: Path) -> Any:
    """Load JSON from file."""
    with open(input_path, 'r') as f:
        return json.load(f)