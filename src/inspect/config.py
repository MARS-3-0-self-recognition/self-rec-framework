"""Configuration for pairwise self-recognition tasks."""

from dataclasses import dataclass
import yaml
from typing import Optional
from pathlib import Path
from src.helpers.utils import project_root


@dataclass
class ExperimentConfig:
    """Unified configuration for experiment setup and prompts."""

    # Experiment structure
    tags: str  # "AT" or "UT"
    format: str  # "PW-C", "PW-Q", "IND-C", "IND-Q"
    task: str  # "Rec" or "Pref"
    priming: bool
    dataset_name: Optional[str] = (
        None  # Optional, will be inferred from dataset path if not provided
    )

    # Generation parameters
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    seed: Optional[int] = None

    # Built prompts (set after initialization)
    generation_prompt: Optional[str] = None
    SR_task_prompt: Optional[str] = None
    system_prompt: Optional[str] = None

    def is_pairwise(self) -> bool:
        """Check if this is a pairwise task (PW-*) vs individual (IND-*)."""
        return self.format.startswith("PW")

    def config_name_for_logging(self) -> str:
        """Generate a config name for logging directories."""
        priming_str = "Pr" if self.priming else "NPr"
        return f"{self.tags}_{self.format}_{self.task}_{priming_str}"


def load_experiment_config(
    config_path: str | Path, dataset_name: str
) -> ExperimentConfig:
    """
    Load an ExperimentConfig from a YAML file and build prompts.

    Args:
        config_path: Path to the experiment config file
        dataset_name: Dataset name (required, inferred from dataset path)

    Returns:
        ExperimentConfig instance with prompts built
    """
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    exp_config = ExperimentConfig(
        tags=config_dict["tags"],
        format=config_dict["format"],
        task=config_dict["task"],
        priming=config_dict.get("priming", False),
        dataset_name=dataset_name,
        temperature=config_dict.get("temperature"),
        max_tokens=config_dict.get("max_tokens"),
        seed=config_dict.get("seed"),
    )

    # Build prompts automatically
    _build_prompts(exp_config)

    return exp_config


def load_base_prompts() -> dict:
    """Load base prompt components from src/core_prompts/prompts.yaml."""
    prompts_path = project_root() / "src" / "core_prompts" / "prompts.yaml"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Base prompts file not found: {prompts_path}")

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def load_dataset_prompts(dataset_name: str) -> dict:
    """Load dataset-specific prompts from data/input/{dataset_name}/prompts.yaml."""
    prompts_path = project_root() / "data" / "input" / dataset_name / "prompts.yaml"
    if not prompts_path.exists():
        raise FileNotFoundError(f"Dataset prompts file not found: {prompts_path}")

    with open(prompts_path, "r") as f:
        return yaml.safe_load(f)


def create_generation_config(
    dataset_name: str,
    temperature: float = None,
    max_tokens: int = None,
    seed: int = None,
) -> ExperimentConfig:
    """
    Create a minimal ExperimentConfig for data generation only.

    Loads dataset prompts and builds system/generation prompts without priming
    or SR task prompts (not needed for generation).

    Args:
        dataset_name: Dataset name
        temperature: Generation temperature
        max_tokens: Max tokens for generation
        seed: Random seed

    Returns:
        ExperimentConfig configured for generation
    """
    # Load dataset prompts
    dataset_prompts = load_dataset_prompts(dataset_name)

    # Create config
    exp_config = ExperimentConfig(
        tags="GEN",  # Marker for generation-only config
        format="GEN",
        task="Gen",
        priming=False,
        dataset_name=dataset_name,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )

    # Build prompts (no priming, no SR task prompts)
    exp_config.system_prompt = dataset_prompts["system_prompt"].format(priming="")
    exp_config.generation_prompt = dataset_prompts["generation_prompt"]
    exp_config.SR_task_prompt = None

    return exp_config


def _get_nested_prompt(
    prompt_dict: dict, keys: list[str], allow_all: bool = True
) -> str:
    """
    Recursively retrieve a nested prompt value with 'All' wildcard support.

    Args:
        prompt_dict: The dictionary to search
        keys: List of keys to traverse (e.g., ["UT", "C", "Rec"])
        allow_all: Whether to check for "All" as a wildcard fallback

    Returns:
        The prompt string

    Raises:
        KeyError: If the path doesn't exist and no "All" fallback is found
    """
    # Base case: if we have a string, return it
    if isinstance(prompt_dict, str):
        return prompt_dict

    # If no more keys but we have a dict, try "All" as final fallback
    if not keys:
        if allow_all and "All" in prompt_dict:
            return _get_nested_prompt(prompt_dict["All"], [], allow_all)
        raise KeyError(
            "Expected string value but got nested dict with no 'All' fallback"
        )

    current_key = keys[0]
    remaining_keys = keys[1:]

    # Try exact key first
    if current_key in prompt_dict:
        return _get_nested_prompt(prompt_dict[current_key], remaining_keys, allow_all)

    # Try "All" wildcard if allowed
    if allow_all and "All" in prompt_dict:
        return _get_nested_prompt(prompt_dict["All"], remaining_keys, allow_all)

    # No match found
    raise KeyError(f"Key '{current_key}' not found and no 'All' fallback available")


def _build_prompts(exp_config: ExperimentConfig) -> None:
    """
    Build prompts for an ExperimentConfig by combining base and dataset prompts.
    Modifies the exp_config in place.

    The prompt structure supports hierarchical organization with "All" wildcards:
    - priming.AT.All applies to all AT formats/tasks
    - priming.UT.C.All applies to all UT conversation formats (Rec/Pref)
    - priming.UT.Q.Rec applies only to UT query recognition

    Args:
        exp_config: ExperimentConfig instance to populate with prompts

    Raises:
        ValueError: If the experiment combination is not implemented
    """
    base_prompts = load_base_prompts()
    dataset_prompts = load_dataset_prompts(exp_config.dataset_name)

    # Parse format string: "PW-C" → pair_type="PW", format_type="C"
    parts = exp_config.format.split("-")
    pair_type = parts[0]  # "PW" or "IND"
    format_type = parts[1]  # "C" (conversation) or "Q" (query)

    # Get priming text for system prompt
    priming_text = ""
    if exp_config.priming:
        try:
            # Build priming keys based on tags
            # AT: priming.AT.All (applies to all formats/tasks)
            # UT: priming.UT.{C|Q}.{All|Rec|Pref}
            if exp_config.tags == "AT":
                priming_keys = [exp_config.tags]  # Will find "All" wildcard
            else:  # UT
                priming_keys = [exp_config.tags, format_type, exp_config.task]

            priming_text = _get_nested_prompt(
                base_prompts["priming"], priming_keys
            ).strip()
        except KeyError as e:
            raise ValueError(
                f"Priming not implemented for combination: "
                f"tags={exp_config.tags}, format={exp_config.format}, task={exp_config.task}"
            ) from e

    # Build system prompt with priming
    exp_config.system_prompt = dataset_prompts["system_prompt"].format(
        priming=priming_text
    )

    # Get generation prompt
    exp_config.generation_prompt = dataset_prompts["generation_prompt"]

    # Build SR task prompt
    sr_task_preface = base_prompts["SR_task_preface"][exp_config.tags].strip()

    # Get SR task template: SR_task[pair_type][format_type][task]
    try:
        sr_task_keys = [pair_type, format_type, exp_config.task]
        sr_task_template = _get_nested_prompt(
            base_prompts["SR_task"], sr_task_keys, allow_all=False
        )
    except KeyError as e:
        raise ValueError(
            f"SR task not implemented for combination: "
            f"format={exp_config.format}, task={exp_config.task}"
        ) from e

    # Replace SR_task_preface, but keep other placeholders for per-sample formatting
    # (e.g., {correct_choice_token}, {incorrect_choice_token} for IND-C tasks)
    sr_task_prompt = sr_task_template.replace("{SR_task_preface}", sr_task_preface)

    # For UT (user tags), we need to build the transcript prefix
    if exp_config.tags == "UT":
        transcript_preface = base_prompts["UT_transcript"]["preface"].strip()
        # Get transcript template: UT_transcript[pair_type][format_type]
        try:
            transcript_keys = [pair_type, format_type]
            transcript_template = _get_nested_prompt(
                base_prompts["UT_transcript"], transcript_keys, allow_all=False
            )
        except KeyError as e:
            raise ValueError(
                f"UT transcript not implemented for format: {exp_config.format}"
            ) from e

        transcript_with_preface = transcript_template.replace(
            "{preface}", transcript_preface
        )
        # For UT, SR_task_prompt includes the transcript template
        sr_task_prompt = transcript_with_preface + "\n\n" + sr_task_prompt

    exp_config.SR_task_prompt = sr_task_prompt
