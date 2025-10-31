"""Configuration for pairwise self-recognition tasks."""

from dataclasses import dataclass
import yaml
from typing import Optional
from src.helpers.utils import project_root


@dataclass
class SelfRecognitionConfig:
    """Configuration for pairwise self-recognition evaluation."""

    generation_prompt: str  # For "Please summarise..." in conversational form
    # task_prompt: str  # For the task prompt
    SR_task_prompt: Optional[str] = None  # For the single-message comparison form
    system_prompt: Optional[str] = None  # For the system prompt
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None


def load_self_recognition_config(config_name: str) -> SelfRecognitionConfig:
    """
    Load a SelfRecognitionConfig from a YAML file.

    Args:
        config_name: Name of the config (e.g., "summarisation" or "qa")

    Returns:
        SelfRecognitionConfig instance
    """
    config_dir = project_root() / "configs" / "protocols" / "pairwise"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return SelfRecognitionConfig(
        generation_prompt=config_dict.get("generation_prompt"),
        # task_prompt=config_dict.get("task_prompt"),
        SR_task_prompt=config_dict.get("SR_task_prompt"),
        system_prompt=config_dict.get("system_prompt"),
        temperature=config_dict.get("temperature"),
        max_tokens=config_dict.get("max_tokens"),
    )
