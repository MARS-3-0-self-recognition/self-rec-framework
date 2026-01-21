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
    task: str  # "Rec" or "Pref" or "Pref-N", "Pref-S", "Pref-Q" (N=Neutral, S=Submission, Q=Quality)
    priming: bool
    # Evaluator reasoning mode: "DR" (direct response) or "CoT" (chain of thought)
    # When "CoT", the actual prompt variant is determined by model type:
    #   - Thinking models (-thinking suffix) → "CoT-R" (trained reasoning)
    #   - Instruct models → "CoT-I" (prompted reasoning)
    evaluator_reasoning: Optional[str] = None
    dataset_name: Optional[str] = (
        None  # Optional, will be inferred from dataset path if not provided
    )

    # Generation parameters
    temperature: Optional[float] = None
    max_final_answer_tokens: Optional[int] = None
    max_thinking_tokens: Optional[int] = (
        None  # Max tokens for thinking models (default: 8192)
    )
    seed: Optional[int] = None

    # Reasoning parameters for thinking models
    reasoning_effort: Optional[str] = (
        "high"  # OpenAI: "low", "medium", "high" (default: "high")
    )
    reasoning_summary: Optional[str] = (
        None  # OpenAI: None, "auto", "last", "concise" (default: None)
    )
    reasoning_history: Optional[str] = "all"  # "all" or "none" (default: "all")

    # Prompt variant selection
    # Controls how generator outputs are inserted into UT transcripts
    # Valid values: "FA" | "RT-FA" | "RT"
    #   FA = Final Answer only
    #   RT-FA = Reasoning Trace + Final Answer
    #   RT = Reasoning Trace only
    generator_output: Optional[str] = None

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
        # Normalize task name: "Pref-N" → "Pref-N", "Pref" → "Pref" (backward compat)
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
        max_final_answer_tokens=config_dict.get("max_final_answer_tokens")
        or config_dict.get("max_tokens"),  # Backward compat
        max_thinking_tokens=config_dict.get("max_thinking_tokens"),
        seed=config_dict.get("seed"),
        evaluator_reasoning=config_dict.get("evaluator_reasoning"),
        generator_output=config_dict.get("generator_output"),
        reasoning_effort=config_dict.get("reasoning_effort", "high"),
        reasoning_summary=config_dict.get("reasoning_summary", None),
        reasoning_history=config_dict.get("reasoning_history", "all"),
    )

    # Build prompts automatically
    _build_prompts(exp_config)

    return exp_config


def ensure_evaluator_reasoning(
    exp_config: ExperimentConfig, evaluator_model_name: str
) -> None:
    """
    Set a default evaluator reasoning mode based on evaluator model name if none is provided,
    then rebuild prompts so SR_task_prompt reflects the selection.

    Default: "CoT" for models ending with "-thinking", otherwise "DR" (direct response).
    Caller should invoke this before using SR_task_prompt in evaluation tasks.
    """
    if exp_config.evaluator_reasoning is None:
        exp_config.evaluator_reasoning = (
            "CoT" if evaluator_model_name.endswith("-thinking") else "DR"
        )

    # Rebuild prompts with evaluator model name to resolve CoT variant
    _build_prompts(exp_config, evaluator_model_name)


def _resolve_reasoning_prompt_key(
    evaluator_reasoning: str, evaluator_model_name: str
) -> str:
    """
    Resolve the evaluator_reasoning config value to the actual prompt key.

    Args:
        evaluator_reasoning: "DR" or "CoT"
        evaluator_model_name: The evaluator model name (used to determine CoT variant)

    Returns:
        The prompt key to use: "DR", "CoT-I", or "CoT-R"
    """
    if evaluator_reasoning == "DR":
        return "DR"
    elif evaluator_reasoning == "CoT":
        # CoT variant depends on model type:
        # - Thinking models use CoT-R (trained reasoning)
        # - Instruct models use CoT-I (prompted reasoning)
        if evaluator_model_name and evaluator_model_name.endswith("-thinking"):
            return "CoT-R"
        else:
            return "CoT-I"
    else:
        raise ValueError(
            f"Invalid evaluator_reasoning: {evaluator_reasoning}. "
            "Must be 'DR' or 'CoT'."
        )


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
    max_final_answer_tokens: int = None,
    seed: int = None,
) -> ExperimentConfig:
    """
    Create a minimal ExperimentConfig for data generation only.

    Loads dataset prompts and builds system/generation prompts without priming
    or SR task prompts (not needed for generation).

    Args:
        dataset_name: Dataset name
        temperature: Generation temperature
        max_final_answer_tokens: Max tokens for final answer generation
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
        max_final_answer_tokens=max_final_answer_tokens,
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


def _build_prompts(
    exp_config: ExperimentConfig, evaluator_model_name: str | None = None
) -> None:
    """
    Build prompts for an ExperimentConfig by combining base and dataset prompts.
    Modifies the exp_config in place.

    The prompt structure supports hierarchical organization with "All" wildcards:
    - priming.AT.All applies to all AT formats/tasks
    - priming.UT.C.All applies to all UT conversation formats (Rec/Pref)
    - priming.UT.Q.Rec applies only to UT query recognition

    Args:
        exp_config: ExperimentConfig instance to populate with prompts
        evaluator_model_name: Optional evaluator model name (used to resolve CoT variant)

    Raises:
        ValueError: If the experiment combination is not implemented
    """
    base_prompts = load_base_prompts()
    dataset_prompts = load_dataset_prompts(exp_config.dataset_name)

    # Parse format string: "PW-C" → pair_type="PW", format_type="C"
    parts = exp_config.format.split("-")
    pair_type = parts[0]  # "PW" or "IND"
    format_type = parts[1]  # "C" (conversation) or "Q" (query)

    # Parse task to extract base task type (for priming lookup)
    # "Pref-N" → "Pref", "Pref-S" → "Pref", "Pref-Q" → "Pref", "Rec" → "Rec"
    base_task = (
        exp_config.task.split("-")[0] if "-" in exp_config.task else exp_config.task
    )

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
                priming_keys = [exp_config.tags, format_type, base_task]

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
    # SR_task_preface structure:
    #   AT.All - for all AT tasks
    #   UT.{C|Q} - for UT tasks based on format_type
    try:
        if exp_config.tags == "AT":
            sr_task_preface_keys = [exp_config.tags]  # Will find "All" wildcard
        else:  # UT
            sr_task_preface_keys = [exp_config.tags, format_type]
        sr_task_preface = _get_nested_prompt(
            base_prompts["SR_task_preface"], sr_task_preface_keys
        ).strip()
    except KeyError as e:
        raise ValueError(
            f"SR_task_preface not implemented for tags={exp_config.tags}, format={format_type}"
        ) from e

    # Get SR task template: SR_task[pair_type][format_type][base_task]
    # Use base_task ("Pref" or "Rec") for template lookup
    try:
        # Rec tasks can have generator_output variants (FA / RT-FA / RT)
        sr_task_keys = [pair_type, format_type, base_task]
        if base_task == "Rec":
            generator_output = exp_config.generator_output
            if generator_output is None:
                # Default to RT-FA (reasoning trace + final answer) for CoT evaluators
                # Default to FA (final answer only) for DR evaluators
                if exp_config.evaluator_reasoning == "CoT":
                    generator_output = "RT-FA"
                else:
                    generator_output = "FA"
            sr_task_keys.append(generator_output)

        sr_task_template = _get_nested_prompt(
            base_prompts["SR_task"], sr_task_keys, allow_all=False
        )
    except KeyError as e:
        raise ValueError(
            f"SR task not implemented for combination: "
            f"format={exp_config.format}, task={exp_config.task} "
            f"generator_output={exp_config.generator_output}"
        ) from e

    # Replace SR_task_preface, but keep other placeholders for per-sample formatting
    # (e.g., {correct_choice_token}, {incorrect_choice_token} for IND-C tasks)
    sr_task_prompt = sr_task_template.replace("{SR_task_preface}", sr_task_preface)

    # Build reasoning+format string using SR_task_prompt_builder
    # Resolve evaluator_reasoning ("DR" or "CoT") to the actual prompt key:
    # - "DR" → "DR"
    # - "CoT" + thinking model → "CoT-R"
    # - "CoT" + instruct model → "CoT-I"
    if exp_config.evaluator_reasoning is None:
        # Default to "DR" when no evaluator reasoning specified
        reasoning_key = "DR"
    else:
        reasoning_key = _resolve_reasoning_prompt_key(
            exp_config.evaluator_reasoning, evaluator_model_name
        )
    try:
        reasoning_template = base_prompts["SR_task_prompt_builder"]["Reasoning"][
            reasoning_key
        ].strip()
    except KeyError as e:
        raise ValueError(f"Unknown reasoning key: {reasoning_key}") from e

    # Select format string; try nested path [pair_type, format_type], with "All" fallback
    try:
        format_template = _get_nested_prompt(
            base_prompts["SR_task_prompt_builder"]["Format"],
            [pair_type, format_type],
            allow_all=True,
        ).strip()
    except KeyError as e:
        raise ValueError(
            f"Format not implemented for pair_type={pair_type}, format_type={format_type}"
        ) from e

    reasoning_format_text = reasoning_template.replace("{Format}", format_template)

    # Insert reasoning+format into SR task prompt
    sr_task_prompt = sr_task_prompt.replace(
        "{SR_task_prompt_builder_Reasoning_Format}", reasoning_format_text
    )

    # For preference tasks, replace preference type placeholder using new builder
    if base_task == "Pref":
        pref_type_map = {
            "N": "Neutral",
            "S": "Submission",
            "Q": "Quality",
        }

        if "-" in exp_config.task:
            pref_suffix = exp_config.task.split("-")[1]
            pref_type = pref_type_map.get(pref_suffix, "Neutral")
        else:
            pref_type = "Neutral"

        try:
            pref_type_text = base_prompts["SR_task_prompt_builder"]["Pref"]["Type"][
                pref_type
            ].strip()
        except KeyError as e:
            raise ValueError(f"Preference type not implemented: {pref_type}") from e

        sr_task_prompt = sr_task_prompt.replace(
            "{SR_task_prompt_builder_Pref_Type}", pref_type_text
        )

    # For UT (user tags), we need to build the transcript prefix
    if exp_config.tags == "UT":
        transcript_preface = base_prompts["UT_transcript"]["preface"].strip()
        # Decide which generator output variant to use.
        # Default: FA for non-CoT runs; RT-FA when evaluator reasoning is CoT and no override is provided.
        generator_output = exp_config.generator_output
        if generator_output is None:
            # Default to RT-FA (reasoning trace + final answer) for CoT evaluators
            # Default to FA (final answer only) for DR evaluators
            if exp_config.evaluator_reasoning == "CoT":
                generator_output = "RT-FA"
            else:
                generator_output = "FA"

        # Get transcript template:
        #   - Pairwise (PW-*): UT_transcript[pair_type][format_type][generator_output]
        #   - Individual (IND-C): UT_transcript[IND][C] (string, no generator_output variants)
        #   - Individual (IND-Q): UT_transcript[IND][Q][generator_output] (has FA/RT-FA/RT variants)
        try:
            if pair_type == "PW":
                transcript_keys = [pair_type, format_type, generator_output]
            elif pair_type == "IND":
                if format_type == "Q":
                    # IND-Q has generator_output variants (FA, RT-FA, RT)
                    transcript_keys = [pair_type, format_type, generator_output]
                else:
                    # IND-C is a simple string
                    transcript_keys = [pair_type, format_type]
            else:
                raise ValueError(
                    f"Unsupported pair_type for UT_transcript: {pair_type}"
                )

            transcript_template = _get_nested_prompt(
                base_prompts["UT_transcript"], transcript_keys, allow_all=False
            )
        except KeyError as e:
            raise ValueError(
                f"UT transcript not implemented for format={exp_config.format} "
                f"and generator_output={generator_output}"
            ) from e

        transcript_with_preface = transcript_template.replace(
            "{preface}", transcript_preface
        )
        # For UT, SR_task_prompt includes the transcript template
        sr_task_prompt = transcript_with_preface + "\n\n" + sr_task_prompt

    exp_config.SR_task_prompt = sr_task_prompt
