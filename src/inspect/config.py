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


def _build_prompts(exp_config: ExperimentConfig) -> None:
    """
    Build prompts for an ExperimentConfig by combining base and dataset prompts.
    Modifies the exp_config in place.

    Args:
        exp_config: ExperimentConfig instance to populate with prompts
    """
    base_prompts = load_base_prompts()
    dataset_prompts = load_dataset_prompts(exp_config.dataset_name)

    # Get priming text for system prompt
    priming_text = ""
    if exp_config.priming:
        priming_text = base_prompts["priming"][exp_config.tags].strip()

    # Build system prompt with priming
    exp_config.system_prompt = dataset_prompts["system_prompt"].format(
        priming=priming_text
    )

    # Get generation prompt
    exp_config.generation_prompt = dataset_prompts["generation_prompt"]

    # Build SR task prompt
    sr_task_preface = base_prompts["SR_task_preface"][exp_config.tags].strip()
    format_task_key = f"{exp_config.format}_{exp_config.task}"
    sr_task_template = base_prompts["SR_task"][format_task_key]

    # Replace SR_task_preface, but keep other placeholders for per-sample formatting
    # (e.g., {correct_choice_token}, {incorrect_choice_token} for IND-C tasks)
    sr_task_prompt = sr_task_template.replace("{SR_task_preface}", sr_task_preface)

    # For UT (user tags), we need to build the transcript prefix
    if exp_config.tags == "UT":
        transcript_preface = base_prompts["UT_transcript"]["preface"].strip()
        # The actual transcript building happens per-sample in the task functions
        # Store the transcript template for later use
        transcript_template = base_prompts["UT_transcript"][exp_config.format]
        transcript_with_preface = transcript_template.replace(
            "{preface}", transcript_preface
        )
        # For UT, SR_task_prompt includes the transcript template
        sr_task_prompt = transcript_with_preface + "\n\n" + sr_task_prompt

    exp_config.SR_task_prompt = sr_task_prompt
