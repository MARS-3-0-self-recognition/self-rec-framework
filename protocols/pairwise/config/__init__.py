"""Configuration for pairwise self-recognition tasks."""

from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class PairwiseConfig:
    """Configuration for pairwise self-recognition evaluation."""
    
    content_field: str  # "Article" or "Question"
    output_field: str  # "Summary" or "Answer"
    content_file: str  # "articles.json" or "questions.json"
    output_file: str  # "summaries.json" or "answers.json"
    prospective_task_prompt: str  # For the single-message prospective form
    conversational_generation_prompt: str  # For "Please summarise..." in conversational form
    conversational_verification_prompt: str  # Final question in conversational form


def load_config(config_name: str) -> PairwiseConfig:
    """
    Load a PairwiseConfig from a YAML file.
    
    Args:
        config_name: Name of the config (e.g., "summarisation" or "qa")
        
    Returns:
        PairwiseConfig instance
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{config_name}.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return PairwiseConfig(**config_dict)


# Convenience function to get common configs
def get_summarisation_config() -> PairwiseConfig:
    """Load the article summarisation config."""
    return load_config("summarisation")


def get_qa_config() -> PairwiseConfig:
    """Load the question answering config."""
    return load_config("qa")
    