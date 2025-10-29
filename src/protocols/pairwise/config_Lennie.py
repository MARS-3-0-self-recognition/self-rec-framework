"""Configuration for pairwise self-recognition tasks."""

from dataclasses import dataclass
import yaml

from src.helpers.utils import project_root


@dataclass
class PairwiseConfig:
    """Configuration for pairwise self-recognition evaluation."""

    comparison_task_prompt: str  # For the single-message comparison form
    generation_prompt: str  # For "Please summarise..." in conversational form
    conversational_verification_prompt: str  # Final question in conversational form
    system_prompt: str  # For the system prompt


def load_pairwise_config(config_name: str) -> PairwiseConfig:
    """
    Load a PairwiseConfig from a YAML file.

    Args:
        config_name: Name of the config (e.g., "summarisation" or "qa")

    Returns:
        PairwiseConfig instance
    """
    config_dir = project_root() / "configs" / "protocols" / "pairwise"
    config_path = config_dir / f"{config_name}.yaml"

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    return PairwiseConfig(**config_dict)
