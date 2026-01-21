"""Self-recognition tasks."""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig,
    get_model,
)
from inspect_ai._util.content import ContentReasoning, ContentText

from src.helpers.model_names import (
    inspect_model_name,
    get_base_model_name,
    needs_reasoning_params,
    INSPECT_MODEL_NAMES,
    is_thinking_model,
)
from src.inspect.config import ExperimentConfig, load_experiment_config
from src.inspect.scorer import logprob_scorer, answer_length_scorer
from src.inspect.data import load_dataset_pairwise, load_dataset_individual

from src.helpers.utils import (
    data_dir,
    load_json,
)


def _get_model_with_custom_base_url(model_name: str, inspect_model_str: str):
    """
    Get a Model object, handling custom base URLs for providers like XAI.

    XAI (Grok) models use the OpenAI provider but need a different base URL.
    This function checks for XAI_API_KEY and returns a properly configured Model.

    Args:
        model_name: Short model name (e.g., "grok-3-mini-thinking")
        inspect_model_str: Inspect model string (e.g., "openai/grok-3-mini")

    Returns:
        Model object or string (for standard models)
    """
    # Check if this is an XAI/Grok model
    if model_name.startswith("grok-"):
        xai_api_key = os.environ.get("XAI_API_KEY")
        if xai_api_key:
            return get_model(
                inspect_model_str,
                base_url="https://api.x.ai/v1",
                api_key=xai_api_key,
            )
        else:
            raise ValueError(
                f"XAI_API_KEY environment variable not set for model {model_name}"
            )

    # Standard models - return string (Task will resolve it)
    return inspect_model_str


def _is_dual_mode_model(model_name: str) -> bool:
    """
    Check if a model is a dual-mode model that can switch between instruct and thinking modes.

    Dual-mode models (Claude 3.7+) have thinking/reasoning that can be enabled/disabled
    at the API level. When using these models WITHOUT the "-thinking" suffix, we need to
    explicitly disable reasoning.

    Note: Gemini 2.5 models are NOT dual-mode - they always use reasoning and it cannot
    be disabled via the API.

    Args:
        model_name: Short model name (e.g., "gemini-2.5-pro", "sonnet-4.5")

    Returns:
        True if the model is dual-mode (can switch), False otherwise
    """
    # Extract base model name (remove -thinking suffix if present)
    base_name = get_base_model_name(model_name)

    # Dual-mode models that can switch between instruct and thinking modes
    # Note: Gemini 2.5 models are NOT included - they always use reasoning
    dual_mode_prefixes = [
        "sonnet-3.7",  # Claude 3.7 Sonnet
        "sonnet-4.5",  # Claude 4.5 Sonnet
        "opus-4.1",  # Claude 4.1 Opus
        "grok-3-mini",  # Grok 3 Mini
    ]

    return any(base_name.startswith(prefix) for prefix in dual_mode_prefixes)


def _is_always_reasoning_model(model_name: str) -> bool:
    """
    Check if a model always uses reasoning and cannot be used in instruct mode.

    These models (like Gemini 2.5) have reasoning enabled by default and it cannot
    be disabled via the API. They should only be called with the "-thinking" suffix.

    Args:
        model_name: Short model name (e.g., "gemini-2.5-pro", "sonnet-4.5")

    Returns:
        True if the model always uses reasoning, False otherwise
    """
    # Extract base model name (remove -thinking suffix if present)
    base_name = get_base_model_name(model_name)

    # Models that always use reasoning (cannot be disabled)
    always_reasoning_prefixes = [
        "gemini-2.5",  # Gemini 2.5 Flash/Pro - always uses reasoning
    ]

    return any(base_name.startswith(prefix) for prefix in always_reasoning_prefixes)


class AlwaysReasoningModelError(Exception):
    """Raised when trying to use an always-reasoning model without -thinking suffix."""

    pass


def _disable_reasoning_for_instruct_mode(
    model_name: str,
    config_params: dict,
    inspect_model_str: str | None = None,
) -> None:
    """
    Explicitly disable reasoning/thinking for dual-mode models used in instruct mode.

    Dual-mode models (Claude 3.7+) have thinking that can be enabled/disabled.
    When using these models WITHOUT the "-thinking" suffix, we ensure reasoning is off.

    Raises:
        AlwaysReasoningModelError: If the model always uses reasoning and cannot be
            used in instruct mode (e.g., Gemini 2.5).

    Args:
        model_name: Short model name (e.g., "gemini-2.5-pro", "sonnet-4.5")
        config_params: Dictionary to update with GenerateConfig parameters
        inspect_model_str: Inspect model string (for provider detection)
    """
    # Only disable reasoning for models used WITHOUT -thinking suffix
    if is_thinking_model(model_name):
        return  # Model has -thinking suffix, reasoning should be enabled

    # Check if this is an always-reasoning model being used without -thinking
    if _is_always_reasoning_model(model_name):
        raise AlwaysReasoningModelError(
            f"Model '{model_name}' always uses reasoning and cannot be used in instruct mode. "
            f"Use '{model_name}-thinking' instead to explicitly enable reasoning mode."
        )

    if not _is_dual_mode_model(model_name):
        return  # Not a dual-mode model, no need to disable reasoning

    # Explicitly disable reasoning for dual-mode models in instruct mode
    if inspect_model_str:
        if "anthropic" in inspect_model_str:
            # Anthropic: Don't enable extended_thinking (it's off by default)
            # Just ensure we don't set any thinking-related parameters
            # Note: Anthropic models may still show reasoning if they've been fine-tuned
            # to include it, but the API-level thinking feature is off by default
            pass
        elif "openai" in inspect_model_str and "grok" in model_name.lower():
            # XAI/Grok: Use reasoning_effort="none" if available, otherwise just don't enable it
            # Note: Grok models may have different behavior; this is a best-effort approach
            pass


def _configure_thinking_model_params(
    model_name: str,
    config: ExperimentConfig,
    config_params: dict,
    model: str | None = None,
) -> None:
    """
    Configure max_tokens, reasoning_tokens, and reasoning parameters for thinking models.

    For thinking models:
    - max_final_answer_tokens: Controls final answer output (separate from reasoning)
    - max_thinking_tokens: Used for reasoning_tokens parameter (where supported)
    - For models without separate reasoning_tokens (Together AI), max_tokens = max_thinking_tokens + max_final_answer_tokens
    - reasoning_effort: OpenAI models only ("low", "medium", "high", default: "high")
    - reasoning_summary: OpenAI models only (None, "auto", "last", "concise", default: None)
    - reasoning_history: All thinking models ("all" or "none", default: "all")

    For dual-mode models (Gemini 2.5, Claude 3.7+) without -thinking suffix:
    - Explicitly disable reasoning by setting reasoning_tokens=0 (Google)

    Args:
        model_name: Short model name (e.g., "sonnet-4.5-thinking")
        config: ExperimentConfig with max_final_answer_tokens, max_thinking_tokens, and reasoning parameters
        config_params: Dictionary to update with GenerateConfig parameters
        model: Optional inspect model string (for provider detection)
    """
    # Determine provider if model string provided
    inspect_model_str = str(model) if model else None

    # For dual-mode models WITHOUT -thinking suffix, explicitly disable reasoning
    if not is_thinking_model(model_name):
        _disable_reasoning_for_instruct_mode(model_name, config_params, inspect_model_str)
        return

    # Set reasoning_history for all thinking models (default: "all")
    if config.reasoning_history:
        config_params["reasoning_history"] = config.reasoning_history

    # For models that support separate reasoning_tokens parameter
    if needs_reasoning_params(model_name):
        # Set reasoning_tokens for Anthropic and Google models
        if inspect_model_str:
            if "anthropic" in inspect_model_str or "google" in inspect_model_str:
                # Anthropic/Google: use max_thinking_tokens for reasoning_tokens
                reasoning_tokens = (
                    config.max_thinking_tokens
                    if config.max_thinking_tokens is not None
                    else 4096
                )
                config_params["reasoning_tokens"] = reasoning_tokens

                # For Anthropic models: max_tokens must be greater than thinking.budget_tokens
                # This is an API requirement. We ensure max_tokens > reasoning_tokens.
                if "max_tokens" not in config_params:
                    if config.max_final_answer_tokens is not None:
                        # User specified a limit - use it, but ensure it's > reasoning_tokens
                        max_answer = config.max_final_answer_tokens
                        config_params["max_tokens"] = max(
                            max_answer, reasoning_tokens + 1
                        )
                    else:
                        # User set null (no limit) - set to a high value to effectively be unlimited
                        # Anthropic API supports up to 128k tokens, but we use 8192 as a practical "unlimited" value
                        # This satisfies the API requirement (max_tokens > reasoning_tokens) while allowing
                        # the model to generate long final answers
                        config_params["max_tokens"] = max(8192, reasoning_tokens + 1)
            elif "openai" in inspect_model_str:
                # OpenAI: reasoning_effort (no separate reasoning_tokens parameter)
                # Default to "high" if not specified in config
                config_params["reasoning_effort"] = config.reasoning_effort or "high"
                # Set reasoning_summary (default: None)
                if config.reasoning_summary is not None:
                    config_params["reasoning_summary"] = config.reasoning_summary

        # For models with separate reasoning_tokens (non-Anthropic): max_tokens is for final answer only
        # Use config.max_final_answer_tokens if set, otherwise reasonable default
        # Note: Anthropic models are handled above with special logic
        if "max_tokens" not in config_params:
            if config.max_final_answer_tokens is not None:
                config_params["max_tokens"] = config.max_final_answer_tokens
            else:
                config_params["max_tokens"] = 2048  # Default for final answer
    else:
        # Together AI models (no separate reasoning_tokens): max_tokens controls total output
        # Total = max_thinking_tokens (for reasoning) + max_final_answer_tokens (for final answer)
        max_thinking = (
            config.max_thinking_tokens
            if config.max_thinking_tokens is not None
            else 8192
        )
        # For Together AI, max_final_answer_tokens in config represents the answer budget, not the total
        # If max_tokens was already set in config_params, use it as the answer budget
        # Otherwise, use config.max_final_answer_tokens or default
        if "max_tokens" in config_params:
            # max_tokens was already set (e.g., from config.max_final_answer_tokens in generation function)
            # Use it as the answer budget and add thinking budget
            max_answer = config_params["max_tokens"]
        else:
            # max_tokens not set yet, use config.max_final_answer_tokens or default
            max_answer = (
                config.max_final_answer_tokens
                if config.max_final_answer_tokens is not None
                else 2048
            )
        # Always set total = thinking + answer for Together AI models
        config_params["max_tokens"] = max_thinking + max_answer


def get_task_function(
    exp_config: ExperimentConfig,
    model_name: str,
    treatment_name_control: str,
    dataset_name: str,
    data_subset: str,
    is_control: bool,
    treatment_name_treatment: str | None = None,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Get and execute the appropriate task function based on experiment configuration.

    Args:
        exp_config: ExperimentConfig with tags, format, and task fields
        model_name: Name of the model being used as evaluator
        treatment_name_control: Name of control (original) treatment
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        is_control: Whether evaluating control (True) or treatment (False) dataset
        treatment_name_treatment: Name of treatment (modified) - for pairwise only
        task_name: Optional custom name for the task (used in log filenames)
        logprobs: Whether to request logprobs from the model (default: False)

    Returns:
        Task object ready to be evaluated

    Raises:
        ValueError: If the combination of tags/format/task is not supported
        NotImplementedError: If logprobs is requested (not yet supported)
    """
    if logprobs:
        raise NotImplementedError(
            "Logprobs support is not yet implemented. "
            "Current scorer uses text-based parsing. "
            "To enable logprobs, update the scorer and fix provider-specific parsing issues."
        )
    tags = exp_config.tags
    format_type = exp_config.format
    task_type = exp_config.task

    # Normalize task type: "Pref-N" → "Pref", "Pref-S" → "Pref", "Pref-Q" → "Pref"
    # Extract base task type for function lookup (task variants use same functions)
    base_task = task_type.split("-")[0] if "-" in task_type else task_type

    # Map (tags, format, task) to task function
    task_map = {
        # Assistant Tags (AT) - conversation in assistant messages
        ("AT", "PW-C", "Rec"): pairwise_conversation_assistant_tags,
        ("AT", "PW-C", "Pref"): pairwise_conversation_assistant_tags,
        ("AT", "IND-C", "Rec"): individual_conversation_assistant_tags,
        ("AT", "IND-C", "Pref"): individual_conversation_assistant_tags,
        # User Tags (UT) - conversation in user messages as transcript
        ("UT", "PW-C", "Rec"): pairwise_conversation_user_tags,
        ("UT", "PW-C", "Pref"): pairwise_conversation_user_tags,
        ("UT", "IND-C", "Rec"): individual_conversation_user_tags,
        # Query format - single message (no conversation history)
        ("AT", "PW-Q", "Rec"): pairwise_query,
        ("AT", "PW-Q", "Pref"): pairwise_query,
        ("UT", "PW-Q", "Rec"): pairwise_query,  # Query format same for AT/UT
        ("UT", "PW-Q", "Pref"): pairwise_query,
        ("AT", "IND-Q", "Rec"): individual_query,
        ("UT", "IND-Q", "Rec"): individual_query,
    }

    key = (tags, format_type, base_task)
    if key not in task_map:
        raise ValueError(
            f"Unsupported combination: tags={tags}, format={format_type}, task={task_type}"
        )

    task_fn = task_map[key]

    # Call the task function with appropriate arguments
    if exp_config.is_pairwise():
        return task_fn(
            model_name=model_name,
            treatment_name_control=treatment_name_control,
            treatment_name_treatment=treatment_name_treatment,
            dataset_name=dataset_name,
            data_subset=data_subset,
            exp_config=exp_config,
            task_name=task_name,
            logprobs=logprobs,
        )
    else:
        return task_fn(
            model_name=model_name,
            treatment_name=treatment_name_control,
            dataset_name=dataset_name,
            data_subset=data_subset,
            exp_config=exp_config,
            is_control=is_control,
            task_name=task_name,
            logprobs=logprobs,
        )


@task
def pairwise_query(
    model_name: str,
    treatment_name_control: str,
    treatment_name_treatment: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Base comparison self-recognition task.

    Single message asking the model to identify which of two outputs it created.
    For UT format, presents the query and responses as a transcript.
    Control dataset contains correct answers.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_control: Name of control (original) treatment
        treatment_name_treatment: Name of treatment (modified) treatment
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        exp_config: ExperimentConfig with prompts and settings
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config
    from src.inspect.config import ensure_evaluator_reasoning

    ensure_evaluator_reasoning(config, model_name)

    inspect_model_str: str = inspect_model_name(model_name)
    # Get model object (handles custom base URLs for XAI/Grok, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Load dataset - control is always first (correct answers)
    dataset_samples = load_dataset_pairwise(
        treatment_name_control,
        treatment_name_treatment,
        dataset_name,
        data_subset,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        # For UT format, SR_task_prompt includes UT_transcript with {generation_prompt} placeholder
        # For AT format, this would need different handling (not currently implemented)
        if config.tags == "UT":
            # Format generation prompt first, then format SR_task_prompt
            generation_prompt = config.generation_prompt.format(
                content=sample_data["content"]
            )
            reasoning1 = sample_data.get("cot1") or ""
            reasoning2 = sample_data.get("cot2") or ""
            prompt = config.SR_task_prompt.format(
                generation_prompt=generation_prompt,
                output1=sample_data["output1"],
                output2=sample_data["output2"],
                reasoning1=reasoning1,
                reasoning2=reasoning2,
            )
        else:
            # AT format - not fully implemented for PW-Q
            # Would need placeholders added to prompts.yaml
            raise NotImplementedError(
                f"PW-Q format with {config.tags} tags is not yet implemented. "
                f"The prompts.yaml needs to be updated with appropriate placeholders."
            )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    # Configure thinking model parameters (max_tokens, reasoning_tokens, reasoning_history, etc.)
    # This will set reasoning_history from config if model is a thinking model
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str
    )
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def pairwise_conversation_assistant_tags(
    model_name: str,
    treatment_name_control: str,
    treatment_name_treatment: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Base conversational self-recognition task.

    Creates a fake conversation history where both outputs appear as assistant
    messages, then asks which one the model actually wrote.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_control: Name of control (original) treatment
        treatment_name_treatment: Name of treatment (modified) treatment
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        exp_config: ExperimentConfig with prompts and settings
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config
    from src.inspect.config import ensure_evaluator_reasoning

    ensure_evaluator_reasoning(config, model_name)

    inspect_model_str: str = inspect_model_name(model_name)

    # Get model object (handles custom base URLs for XAI, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Check if evaluator is Anthropic (they have strict requirements for reasoning content)
    is_anthropic_evaluator = "anthropic" in inspect_model_str.lower()

    # Check if treatment models are Anthropic (to determine if we can use their reasoning format)
    # Get base model names (remove -thinking suffix and treatment suffixes)
    from src.helpers.model_names import get_base_model_name

    base_control = get_base_model_name(treatment_name_control)
    base_treatment = get_base_model_name(treatment_name_treatment)

    # Check if treatment models map to Anthropic endpoints
    control_inspect = (
        inspect_model_name(base_control) if base_control in INSPECT_MODEL_NAMES else ""
    )
    treatment_inspect = (
        inspect_model_name(base_treatment)
        if base_treatment in INSPECT_MODEL_NAMES
        else ""
    )

    is_control_anthropic = "anthropic" in control_inspect.lower()
    is_treatment_anthropic = "anthropic" in treatment_inspect.lower()

    # Load dataset - control is always first (correct answers)
    dataset_samples = load_dataset_pairwise(
        treatment_name_control,
        treatment_name_treatment,
        dataset_name,
        data_subset,
    )

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Build conversation history
        messages = []

        # First interaction with output1
        messages.append(ChatMessageUser(content=generation_prompt))

        # Add output1 with CoT if available
        cot1 = sample_data.get("cot1")
        signature1 = sample_data.get("signature1")
        if cot1:
            # Determine which treatment model generated output1
            # In sample 1: output1 is from treatment_name_1 (control)
            # In sample 2: output1 is from treatment_name_2 (treatment, swapped)
            # Check metadata to see which sample this is
            is_first_sample = sample_data["metadata"].get("correct_answer") == "1"
            cot1_source_is_anthropic = (
                is_control_anthropic if is_first_sample else is_treatment_anthropic
            )

            # Use ContentReasoning with signature if available (for Anthropic models)
            # Note: Signatures are tied to specific reasoning content. We allow cross-Anthropic
            # model usage (e.g., sonnet 3.7 signature with opus 4.1 evaluator) - if the API
            # rejects this, we'll get an error and can adjust.
            # Otherwise use redacted=True for other providers
            if signature1 and is_anthropic_evaluator and cot1_source_is_anthropic:
                # Anthropic evaluator with Anthropic CoT - use signature
                # (May work across different Anthropic models, e.g., sonnet -> opus)
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot1, signature=signature1),
                            ContentText(text=sample_data["output1"]),
                        ]
                    )
                )
            elif not is_anthropic_evaluator:
                # Non-Anthropic evaluator - include plaintext reasoning so it appears in logs
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot1),
                            ContentText(text=sample_data["output1"]),
                        ]
                    )
                )
            elif cot1_source_is_anthropic:
                # Anthropic evaluator with Anthropic CoT but no signature - fall back to redacted
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot1, redacted=True),
                            ContentText(text=sample_data["output1"]),
                        ]
                    )
                )
            else:
                # Skip CoT for Anthropic evaluator with non-Anthropic CoT (no signature)
                messages.append(ChatMessageAssistant(content=sample_data["output1"]))
        else:
            messages.append(ChatMessageAssistant(content=sample_data["output1"]))

        # Second interaction with output2 (same article/question)
        messages.append(ChatMessageUser(content=generation_prompt))

        # Add output2 with CoT if available
        cot2 = sample_data.get("cot2")
        signature2 = sample_data.get("signature2")
        if cot2:
            # Determine which treatment model generated output2
            is_first_sample = sample_data["metadata"].get("correct_answer") == "1"
            cot2_source_is_anthropic = (
                is_treatment_anthropic if is_first_sample else is_control_anthropic
            )

            # Use ContentReasoning with signature if available (for Anthropic models)
            # Note: Signatures are tied to specific reasoning content. We allow cross-Anthropic
            # model usage (e.g., sonnet 3.7 signature with opus 4.1 evaluator) - if the API
            # rejects this, we'll get an error and can adjust.
            # Otherwise use redacted=True for other providers
            if signature2 and is_anthropic_evaluator and cot2_source_is_anthropic:
                # Anthropic evaluator with Anthropic CoT - use signature
                # (May work across different Anthropic models, e.g., sonnet -> opus)
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot2, signature=signature2),
                            ContentText(text=sample_data["output2"]),
                        ]
                    )
                )
            elif not is_anthropic_evaluator:
                # Non-Anthropic evaluator - include plaintext reasoning so it appears in logs
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot2),
                            ContentText(text=sample_data["output2"]),
                        ]
                    )
                )
            elif cot2_source_is_anthropic:
                # Anthropic evaluator with Anthropic CoT but no signature - fall back to redacted
                messages.append(
                    ChatMessageAssistant(
                        content=[
                            ContentReasoning(reasoning=cot2, redacted=True),
                            ContentText(text=sample_data["output2"]),
                        ]
                    )
                )
            else:
                # Skip CoT for Anthropic evaluator with non-Anthropic CoT (no signature)
                messages.append(ChatMessageAssistant(content=sample_data["output2"]))
        else:
            messages.append(ChatMessageAssistant(content=sample_data["output2"]))

        # Final verification question
        messages.append(ChatMessageUser(content=config.SR_task_prompt))

        inspect_samples.append(
            Sample(
                input=messages,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    # Configure thinking model parameters (max_tokens, reasoning_tokens, reasoning_history, etc.)
    # This will set reasoning_history from config if model is a thinking model
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str
    )
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def pairwise_conversation_user_tags(
    model_name: str,
    treatment_name_control: str,
    treatment_name_treatment: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Base comparison self-recognition task.

    Creates a fake transcript of a model conversation structured like pairwise_conversation_assistant_tags, but where all content appears in the user messages. Informs the model that the transcript is of a previous conversation it had, and asks it to identify which of the two responses it originally produced.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name_control: Name of control (original) treatment
        treatment_name_treatment: Name of treatment (modified) treatment
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        exp_config: ExperimentConfig with prompts and settings
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config
    from src.inspect.config import ensure_evaluator_reasoning

    ensure_evaluator_reasoning(config, model_name)

    inspect_model_str: str = inspect_model_name(model_name)
    # Get model object (handles custom base URLs for XAI/Grok, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Load dataset - control is always first (correct answers)
    dataset_samples = load_dataset_pairwise(
        treatment_name_control,
        treatment_name_treatment,
        dataset_name,
        data_subset,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output1=sample_data["output1"],
            output2=sample_data["output2"],
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    # Configure thinking model parameters (max_tokens, reasoning_tokens)
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str
    )
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def individual_conversation_assistant_tags(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    is_control: bool = True,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Base conversational self-recognition task.

    Creates a fake conversation history where the model's response appears as an assistant message, then asks it to evaluate if it was originally produced by the model or not.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory (e.g., 'training_set_1-20')
        exp_config: ExperimentConfig with prompts and settings
        is_control: Whether evaluating control dataset
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config
    from src.inspect.config import ensure_evaluator_reasoning

    ensure_evaluator_reasoning(config, model_name)

    inspect_model_str: str = inspect_model_name(model_name)
    # Get model object (handles custom base URLs for XAI/Grok, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Load dataset
    dataset_samples = load_dataset_individual(
        treatment_name,
        dataset_name,
        data_subset,
        is_control=is_control,
    )

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the generation prompt
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Format SR task prompt with choice tokens
        sr_task_prompt = config.SR_task_prompt.format(
            correct_choice_token=sample_data["metadata"]["correct_choice_token"],
            incorrect_choice_token=sample_data["metadata"]["incorrect_choice_token"],
        )

        # Build conversation history
        messages = [
            # First interaction with output
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output"]),
            # Final verification question
            ChatMessageUser(content=sr_task_prompt),
        ]

        inspect_samples.append(
            Sample(
                input=messages,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    # Configure thinking model parameters (max_tokens, reasoning_tokens, reasoning_history, etc.)
    # This will set reasoning_history from config if model is a thinking model
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str
    )
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def individual_conversation_user_tags(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    is_control: bool = True,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Base comparison self-recognition task.

    Creates a fake transcript of a model conversation structured like individual_conversation_assistant_tags, but where all content appears in the user messages. Informs the model that the transcript is of a previous conversation it had, and asks it to evaluate if its response was originally produced by the model or not.
    Returns a Task object that can be returned directly or modified for variants.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory
        exp_config: ExperimentConfig with prompts and settings
        is_control: Whether evaluating control dataset
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config

    inspect_model_str: str = inspect_model_name(model_name)
    # Get model object (handles custom base URLs for XAI/Grok, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Load dataset
    dataset_samples = load_dataset_individual(
        treatment_name,
        dataset_name,
        data_subset,
        is_control=is_control,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format the prompt using the config template
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output=sample_data["output"],
            correct_choice_token=sample_data["metadata"]["correct_choice_token"],
            incorrect_choice_token=sample_data["metadata"]["incorrect_choice_token"],
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    # Configure thinking model parameters (max_tokens, reasoning_tokens)
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str
    )
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def individual_query(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    is_control: bool = True,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """
    Individual self-recognition task with single query message.

    Presents a single output and asks if the model wrote it.

    Args:
        model_name: Name of the model being used as evaluator
        treatment_name: Name of the treatment being evaluated
        dataset_name: Name of the dataset directory under data/
        data_subset: Data subset directory
        exp_config: ExperimentConfig with prompts and settings
        is_control: Whether evaluating control dataset
        task_name: Optional custom name for the task (used in log filenames)

    Returns:
        Task object configured with logprobs enabled
    """
    config = exp_config

    inspect_model_str: str = inspect_model_name(model_name)
    # Get model object (handles custom base URLs for XAI/Grok, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Load dataset
    dataset_samples = load_dataset_individual(
        treatment_name,
        dataset_name,
        data_subset,
        is_control=is_control,
    )

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        # Format generation prompt from content
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Format the SR task prompt using the config template
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output=sample_data["output"],
            reasoning=sample_data.get("reasoning", ""),
            correct_choice_token=sample_data["metadata"].get(
                "correct_choice_token", "1"
            ),
            incorrect_choice_token=sample_data["metadata"].get(
                "incorrect_choice_token", "2"
            ),
        )

        inspect_samples.append(
            Sample(
                input=prompt,
                target=sample_data["metadata"]["correct_answer"],
                metadata=sample_data["metadata"],
            )
        )

    # Build GenerateConfig with optional logprobs
    config_params = {"system_message": config.system_prompt}
    if logprobs:
        config_params["logprobs"] = True
        config_params["top_logprobs"] = 2

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=logprob_scorer(),
        model=inspect_model,
        config=GenerateConfig(**config_params),
        name=task_name,
    )


@task
def generation(
    model_name: str,
    dataset_name: str,
    data_subset: str,
    config_path: str = None,
    exp_config: ExperimentConfig = None,
) -> Task:
    """
    Base generation task.

    Args:
        model_name: Model to use for generation
        dataset_name: Dataset name
        data_subset: Data subset directory (e.g., 'debug', 'training_set_1-20')
        config_path: Path to experiment config (if not providing exp_config)
        exp_config: ExperimentConfig instance (if not providing config_path)
    """
    contents = load_json(
        data_dir() / "input" / dataset_name / data_subset / "input.json"
    )

    # Load config if not provided
    if exp_config is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or exp_config")
        config = load_experiment_config(config_path, dataset_name=dataset_name)
    else:
        config = exp_config

    # Handle thinking models: if model_name ends with "-thinking" and needs API params,
    # use the base model name for the API call but add reasoning parameters.
    # For models like "qwen-3.0-80b-thinking", the full name maps to a different endpoint.
    if model_name.endswith("-thinking") and needs_reasoning_params(model_name):
        # Use base model name for API call (same endpoint, different params)
        base_model_name = get_base_model_name(model_name)
        inspect_model_str = inspect_model_name(base_model_name)
    else:
        # Use model name as-is (either non-thinking or different endpoint)
        inspect_model_str = inspect_model_name(model_name)

    # Get model object (handles custom base URLs for XAI, etc.)
    model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # create base samples
    inspect_samples = []
    for uuid in contents.keys():
        sample_input = contents[uuid]
        generation_prompt = config.generation_prompt.format(
            content=sample_input,
        )
        metadata = {
            "uuid": uuid,
            "model_name": model_name,  # Keep original name in metadata
            "config_path": config_path,
        }
        inspect_samples.append(
            Sample(
                input=generation_prompt,
                id=uuid,
                metadata=metadata,
            )
        )

    # Create GenerateConfig with only non-None parameters
    generate_config_params = {}
    if config.system_prompt is not None:
        generate_config_params["system_message"] = config.system_prompt
    if config.temperature is not None:
        generate_config_params["temperature"] = config.temperature
    if config.max_final_answer_tokens is not None:
        generate_config_params["max_tokens"] = config.max_final_answer_tokens
    if config.seed is not None:
        generate_config_params["seed"] = config.seed

    # Configure thinking model parameters (max_tokens, reasoning_tokens)
    # Note: max_tokens may already be set from config.max_final_answer_tokens above (line 927)
    # The helper will respect existing max_tokens and only set it if not present
    _configure_thinking_model_params(
        model_name, config, generate_config_params, inspect_model_str
    )

    return Task(
        dataset=inspect_samples,
        solver=generate(),
        scorer=answer_length_scorer(),
        model=model,
        config=GenerateConfig(**generate_config_params)
        if generate_config_params
        else None,
    )
