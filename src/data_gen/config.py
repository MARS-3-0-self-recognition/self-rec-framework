"""Configuration for data generation tasks."""

from dataclasses import dataclass
import yaml
from typing import Optional

from src.helpers.utils import project_root


@dataclass
class DataGenerationConfig:
    """Configuration for data generation tasks."""

    user_prompt: str  # For the user prompt
    system_prompt: Optional[str] = None  # Optional system prompt

    # Common generation parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    seed: Optional[int] = None
    stop_seqs: Optional[list[str]] = None


def load_generation_config(config_name: str) -> DataGenerationConfig:
    """
    Load a DataGenerationConfig from a YAML file.

    Args:
        config_name: Name of the config (e.g., "wikisum_config")

    Returns:
        DataGenerationConfig instance
    """
    config_dir = project_root() / "configs" / "generation"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    # Map config fields to our dataclass
    return DataGenerationConfig(
        system_prompt=config_dict.get("system_prompt"),
        user_prompt=config_dict.get("user_prompt", config_dict.get("prompt", "")),
        temperature=config_dict.get("temperature"),
        max_tokens=config_dict.get("max_tokens"),
        top_p=config_dict.get("top_p"),
        top_k=config_dict.get("top_k"),
        seed=config_dict.get("seed"),
        stop_seqs=config_dict.get("stop_seqs"),
    )
