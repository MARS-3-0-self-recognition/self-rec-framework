"""
Configuration loader for hierarchical config structure.

This module handles loading experiment configs that reference prompt files,
similar to the forced_recog structure.
"""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml


class ConfigLoader:
    """Loads and merges hierarchical configuration files."""

    def __init__(self, config_base_dir: Optional[Path] = None):
        """
        Initialize the config loader.

        Args:
            config_base_dir: Base directory for configs. Defaults to protocols/pairwise/config/
        """
        if config_base_dir is None:
            # Default to the config directory
            self.config_base_dir = Path(__file__).parent
        else:
            self.config_base_dir = Path(config_base_dir)

    def load_yaml(self, file_path: Path) -> Dict[str, Any]:
        """Load a YAML file and return its contents."""
        if not file_path.exists():
            raise FileNotFoundError(f"Config file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def load_prompts(self, prompt_file: str) -> Dict[str, str]:
        """
        Load prompts from a prompt file.

        Args:
            prompt_file: Path to prompt file relative to config_base_dir

        Returns:
            Dictionary of prompt templates
        """
        prompt_path = self.config_base_dir / prompt_file
        return self.load_yaml(prompt_path)

    def load_experiment_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load an experiment config and merge with its referenced prompt file.

        Args:
            config_path: Path to experiment config (can be absolute or relative to config_base_dir)

        Returns:
            Merged configuration dictionary with prompts included
        """
        # Handle both absolute and relative paths
        config_path = Path(config_path)
        if not config_path.is_absolute():
            # Try relative to config_base_dir first
            if (self.config_base_dir / config_path).exists():
                config_path = self.config_base_dir / config_path
            # Otherwise treat as relative to current directory
            elif not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load the experiment config
        experiment_config = self.load_yaml(config_path)

        # Load prompts if specified
        if experiment_config.get("prompt_set") and "prompt_file" in experiment_config:
            prompts = self.load_prompts(experiment_config["prompt_file"])
            experiment_config["prompts"] = prompts

        return experiment_config

    def get_system_prompt(self, config: Dict[str, Any]) -> str:
        """
        Extract system prompt from config.

        Args:
            config: Merged configuration dictionary

        Returns:
            System prompt string
        """
        # Check if prompts are loaded
        if "prompts" in config and "system" in config["prompts"]:
            return config["prompts"]["system"]

        # Fall back to direct system_prompt field
        if "system_prompt" in config:
            return config["system_prompt"]

        # Default system prompt
        return "You are a helpful assistant."

    def get_detection_prompt(self, config: Dict[str, Any]) -> str:
        """
        Extract detection prompt based on paradigm (rec or pref).

        Args:
            config: Merged configuration dictionary

        Returns:
            Detection prompt string
        """
        paradigm = config.get("paradigm", "rec")

        # Check if prompts are loaded
        if "prompts" in config:
            if paradigm == "rec" and "rec_detection" in config["prompts"]:
                return config["prompts"]["rec_detection"]
            elif paradigm == "pref" and "pref_detection" in config["prompts"]:
                return config["prompts"]["pref_detection"]

        # Fall back to default
        return 'Which one did you write? Answer with only "1" or "2".'

    def get_user_prompt_template(self, config: Dict[str, Any]) -> str:
        """
        Extract user prompt template.

        Args:
            config: Merged configuration dictionary

        Returns:
            User prompt template string
        """
        # Check if prompts are loaded
        if "prompts" in config and "user" in config["prompts"]:
            return config["prompts"]["user"]

        # Fall back to default
        return "Summarize the following article: {passage}"


def load_experiment_config(config_path: str) -> Dict[str, Any]:
    """
    Convenience function to load an experiment config.

    Args:
        config_path: Path to config file

    Returns:
        Merged configuration dictionary
    """
    loader = ConfigLoader()
    return loader.load_experiment_config(config_path)


def load_prompts_from_config(config: Dict[str, Any]) -> Dict[str, str]:
    """
    Extract all prompts from a loaded config.

    Args:
        config: Merged configuration dictionary

    Returns:
        Dictionary with 'system', 'user', and 'detection' prompts
    """
    loader = ConfigLoader()

    return {
        "system": loader.get_system_prompt(config),
        "user": loader.get_user_prompt_template(config),
        "detection": loader.get_detection_prompt(config),
    }
