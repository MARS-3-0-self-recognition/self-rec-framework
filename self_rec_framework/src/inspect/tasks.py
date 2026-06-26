"""Self-recognition tasks."""

import os

from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, multiple_choice
from inspect_ai.scorer import choice
from inspect_ai.model import (
    ChatMessageUser,
    ChatMessageAssistant,
    GenerateConfig,
    get_model,
)
from inspect_ai._util.content import ContentReasoning, ContentText

from self_rec_framework.src.helpers.model_names import (
    inspect_model_name,
    get_base_model_name,
    needs_reasoning_params,
    needs_together_reasoning_activation,
    INSPECT_MODEL_NAMES,
    is_thinking_model,
    get_model_output_token_cap,
    get_model_context_window,
    get_model_max_thinking_tokens,
)
from self_rec_framework.src.inspect.config import ExperimentConfig, load_experiment_config
from self_rec_framework.src.inspect.scorer import logprob_scorer, answer_length_scorer
from self_rec_framework.src.inspect.data import load_dataset_pairwise, load_dataset_individual, load_dataset_mmlu, load_icl_examples, load_icl_pool, select_icl_from_pool

from self_rec_framework.src.helpers.utils import (
    data_dir,
    load_json,
)


def _load_icl_for_task(
    exp_config: ExperimentConfig,
    dataset_samples: list,
    dataset_name: str,
    data_subset: str,
) -> list[dict]:
    """Load ICL examples (single shared set) if configured, excluding evaluation UUIDs."""
    if not exp_config.icl_count or not exp_config.icl_model:
        return []
    eval_uuids = {s["metadata"]["uuid"] for s in dataset_samples}
    seed = exp_config.icl_seed if exp_config.icl_seed is not None else (exp_config.seed or 42)
    icl_dataset = exp_config.icl_dataset or dataset_name
    icl_subset = exp_config.icl_data_subset or data_subset
    same_pool = (icl_dataset == dataset_name) and (icl_subset == data_subset)
    exclude = eval_uuids if same_pool else set()
    examples = load_icl_examples(
        icl_model=exp_config.icl_model,
        dataset_name=icl_dataset,
        data_subset=icl_subset,
        count=exp_config.icl_count,
        exclude_uuids=exclude,
        seed=seed,
    )
    print(f"  ICL: loaded {len(examples)} examples from {exp_config.icl_model}/{icl_dataset}/{icl_subset} "
          f"(excluded {len(exclude)} UUIDs)")
    return examples


def _load_icl_pool_for_task(
    exp_config: ExperimentConfig,
    dataset_samples: list,
    dataset_name: str,
    data_subset: str,
):
    """Load the full ICL pool (for per-sample shuffling). Returns (pool_dict, base_seed) or (None, None)."""
    if not exp_config.icl_count or not exp_config.icl_model:
        return None, None
    eval_uuids = {s["metadata"]["uuid"] for s in dataset_samples}
    base_seed = exp_config.icl_seed if exp_config.icl_seed is not None else (exp_config.seed or 42)
    icl_dataset = exp_config.icl_dataset or dataset_name
    icl_subset = exp_config.icl_data_subset or data_subset
    same_pool = (icl_dataset == dataset_name) and (icl_subset == data_subset)
    exclude = eval_uuids if same_pool else set()
    pool = load_icl_pool(
        icl_model=exp_config.icl_model,
        dataset_name=icl_dataset,
        data_subset=icl_subset,
        exclude_uuids=exclude,
    )
    print(f"  ICL (per-sample): pool={len(pool)} examples from {exp_config.icl_model}/{icl_dataset}/{icl_subset} "
          f"(excluded {len(exclude)} UUIDs)")
    return pool, base_seed


def _icl_seed_for_uuid(base_seed: int, uuid: str) -> int:
    """Derive a deterministic per-UUID seed from the base seed. Stable across runs.

    The two answer-order-bias samples sharing a UUID get the SAME seed, so they
    see the SAME ICL context — preserving a fair same-context comparison.
    """
    import hashlib
    h = hashlib.sha256(f"{base_seed}:{uuid}".encode()).digest()
    return int.from_bytes(h[:8], "big", signed=False)


def _build_icl_messages(
    icl_examples: list[dict],
    generation_prompt_template: str,
) -> list:
    """Build ChatMessage turn pairs for AT format ICL injection."""
    messages = []
    for ex in icl_examples:
        prompt = generation_prompt_template.format(content=ex["prompt"])
        messages.append(ChatMessageUser(content=prompt))
        messages.append(ChatMessageAssistant(content=ex["response"]))
    return messages


def _build_icl_resolver(
    exp_config: ExperimentConfig,
    dataset_samples: list,
    dataset_name: str,
    data_subset: str,
):
    """Return a callable `resolve(uuid) -> list[ChatMessage]` for ICL injection.

    - If ICL isn't configured: returns a no-op (returns []).
    - If icl_shuffle_per_sample=True: loads the pool once, selects per-UUID with
      a deterministic seed on each call.
    - Otherwise: loads a single shared set once and returns it for every UUID
      (old behavior).
    """
    if not exp_config.icl_count or not exp_config.icl_model:
        return lambda uuid: []

    if exp_config.icl_shuffle_per_sample:
        pool, base_seed = _load_icl_pool_for_task(exp_config, dataset_samples,
                                                  dataset_name, data_subset)

        def resolve(uuid: str):
            examples = select_icl_from_pool(
                pool=pool,
                count=exp_config.icl_count,
                seed=_icl_seed_for_uuid(base_seed, uuid),
            )
            return _build_icl_messages(examples, exp_config.generation_prompt)

        return resolve

    shared_examples = _load_icl_for_task(exp_config, dataset_samples,
                                         dataset_name, data_subset)
    shared_messages = (_build_icl_messages(shared_examples, exp_config.generation_prompt)
                       if shared_examples else [])
    return lambda uuid: shared_messages


    # _build_icl_text removed — all formats now use _build_icl_messages
    # to inject ICL as real ChatMessage turn pairs (no framing text)


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


class ReasoningTokensRequiredError(Exception):
    """Raised when reasoning is enabled but the required CoT token budget is unset."""

    pass


class ContextWindowExceededError(Exception):
    """Raised when a model's context window can't fit the estimated input plus a
    minimal output budget — i.e. the evaluator genuinely can't run this task. The
    sweep catches this to skip the model gracefully rather than abort."""

    pass


def _disable_reasoning_for_instruct_mode(
    model_name: str,
    config_params: dict,
    inspect_model_str: str = "",
) -> None:
    """
    Explicitly disable reasoning/thinking for dual-mode models used in instruct mode.

    Reasoning on/off is driven by the "-thinking" suffix (the config-level toggle):
    a model name without the suffix means instruct mode, in which case dual-mode
    models (Claude 3.7+, Grok 3 Mini) must have their reasoning kept off.

    Raises:
        AlwaysReasoningModelError: If the model always uses reasoning and cannot be
            run in instruct mode (e.g., Gemini 2.5). Use the "-thinking" variant.

    Args:
        model_name: Short model name (e.g., "gemini-2.5-pro", "sonnet-4.5")
        config_params: GenerateConfig parameter dict, mutated in place.
        inspect_model_str: Inspect model string for provider detection
            (e.g., "anthropic/claude-..."). Defaults to "" so the membership
            checks below are always safe without a guard.
    """
    # Fail loudly and first: always-reasoning models cannot run in instruct mode at
    # all. Guarded on the suffix so the "-thinking" variant is never rejected here.
    if _is_always_reasoning_model(model_name) and not is_thinking_model(model_name):
        raise AlwaysReasoningModelError(
            f"Model '{model_name}' always uses reasoning and cannot be used in instruct mode. "
            f"Use '{model_name}-thinking' instead to explicitly enable reasoning mode."
        )

    # Nothing to disable: thinking models keep reasoning on, and non-dual-mode
    # models have no API-level reasoning to turn off.
    if is_thinking_model(model_name) or not _is_dual_mode_model(model_name):
        return

    # Dual-mode model in instruct mode: disable reasoning per provider. Both
    # currently-supported providers are intentional no-ops:
    #   - Anthropic: extended thinking is OFF by default; we simply never set
    #     reasoning_tokens, so there is no parameter to add here.
    #   - XAI/Grok: the API exposes no "off" switch (reasoning_effort has no
    #     "none" value), so disabling is best-effort / not possible.
    # This block is the extension point: a future provider that needs an explicit
    # disable parameter would set it on config_params here.
    if "anthropic" in inspect_model_str:
        pass
    elif "openai" in inspect_model_str and "grok" in model_name.lower():
        pass


def _resolve_max_tokens(value, model_name: str, kind: str = "output"):
    """
    Resolve a token-budget config value to a concrete int (or None).

    - None -> None (unset)
    - int  -> the int unchanged
    - "max" -> the model's ceiling: the output ceiling (MODEL_OUTPUT_TOKEN_CAP) when
      kind="output", or the thinking-budget cap (get_model_max_thinking_tokens)
      when kind="thinking". Raises ValueError if the model is unknown.

    Any other string is rejected (config validation also catches this at load).
    """
    if value is None or isinstance(value, int):
        return value
    if isinstance(value, str) and value.lower() == "max":
        if kind == "thinking":
            return get_model_max_thinking_tokens(model_name)
        return get_model_output_token_cap(model_name)
    raise ValueError(
        f"Invalid token budget {value!r} for model '{model_name}': "
        f"expected an integer, null, or 'max'."
    )


def _is_max_sentinel(value) -> bool:
    """True if a token-budget config value is the 'max' sentinel."""
    return isinstance(value, str) and value.lower() == "max"


def _sample_input_chars(sample_input) -> int:
    """Best-effort character count of a sample's input.

    Handles both a plain string prompt and a list of chat messages (whose
    content may itself be a string or a list of content blocks). Used to
    estimate input tokens when resolving a "max" output budget.
    """
    if isinstance(sample_input, str):
        return len(sample_input)
    total = 0
    for msg in sample_input:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            total += len(content)
        elif isinstance(content, list):
            for block in content:
                total += len(getattr(block, "text", "") or "")
        else:
            total += len(str(content))
    return total


# Reservation applied when an output budget is "max", so input + max_tokens
# stays within the provider's context window. Input tokens are estimated from
# characters (~chars/3, which conservatively over-counts English/code) plus the
# system prompt; the margin covers chat-template special tokens and estimate
# variance. If the reservation leaves less than the floor for output, we raise.
_INPUT_RESERVE_MARGIN = 512
_OUTPUT_FLOOR = 512


def _configure_thinking_model_params(
    model_name: str,
    config: ExperimentConfig,
    config_params: dict,
    model: str | None = None,
    samples: list | None = None,
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
    # Determine provider if model string provided ("" when absent so the
    # membership checks below and in _disable_reasoning_for_instruct_mode are safe)
    inspect_model_str = str(model) if model else ""

    # Resolve a "max" output budget to a concrete int. The provider enforces
    # (input + output) <= context window, so the most we can emit is
    #   min(output_cap, context_window - estimated_input - margin).
    # output_cap (the API's max generation tokens) and context_window are per-model
    # specs; estimated_input is measured from the actual prompts (~chars/3); margin
    # is fixed slack. Only used when a budget is "max"; computed lazily so non-"max"
    # runs do no extra work. Raises ContextWindowExceededError when the input can't
    # fit, so the sweep can skip this evaluator instead of failing every request.
    def max_output_ceiling() -> int:
        output_cap = get_model_output_token_cap(model_name)
        if not samples:
            return output_cap
        max_in = max((_sample_input_chars(s.input) for s in samples), default=0)
        sys_chars = len(config.system_prompt or "")
        # ceil((input chars + system chars) / 3) conservatively over-counts tokens
        est_input = -(-(max_in + sys_chars) // 3)
        context_window = get_model_context_window(model_name)
        usable = min(output_cap, context_window - est_input - _INPUT_RESERVE_MARGIN)
        if usable < _OUTPUT_FLOOR:
            raise ContextWindowExceededError(
                f"Estimated input (~{est_input} tokens) leaves < {_OUTPUT_FLOOR} "
                f"tokens for output within '{model_name}' context window "
                f"{context_window}. This evaluator can't fit the prompt — reduce "
                f"input length or use a longer-context model."
            )
        return usable

    # A call site (mmlu/generation) may have pre-seeded config_params["max_tokens"]
    # straight from config.max_final_answer_tokens, including the raw "max" sentinel.
    # Resolve it here so every downstream path sees a concrete int.
    if "max_tokens" in config_params:
        seed = config_params["max_tokens"]
        config_params["max_tokens"] = (
            max_output_ceiling() if _is_max_sentinel(seed)
            else _resolve_max_tokens(seed, model_name)
        )

    # Resolve the answer-token budget ("max" -> reserved model output ceiling).
    # Applies to every model regardless of reasoning mode, so instruct models
    # honor it too.
    if _is_max_sentinel(config.max_final_answer_tokens):
        resolved_answer = max_output_ceiling()
    else:
        resolved_answer = _resolve_max_tokens(config.max_final_answer_tokens, model_name)

    # For dual-mode models WITHOUT -thinking suffix, explicitly disable reasoning
    if not is_thinking_model(model_name):
        _disable_reasoning_for_instruct_mode(
            model_name, config_params, inspect_model_str
        )
        # Instruct models cap output via max_final_answer_tokens (the whole budget
        # is the answer — no reasoning to share it with).
        if resolved_answer is not None and "max_tokens" not in config_params:
            config_params["max_tokens"] = resolved_answer
        return

    # Set reasoning_history for all thinking models (default: "all")
    if config.reasoning_history:
        config_params["reasoning_history"] = config.reasoning_history

    # Resolve the "max" sentinel for the thinking budget ("max" -> the thinking cap,
    # which is the shared output ceiling except where a model enforces a smaller one,
    # e.g. Gemini 2.5). When "max" was requested for either budget, also fetch the
    # output ceiling so combined budgets can be capped to it below (a thinking
    # model's reasoning and answer share one output budget).
    resolved_thinking = _resolve_max_tokens(
        config.max_thinking_tokens, model_name, kind="thinking"
    )
    uses_max = "max" in (config.max_thinking_tokens, config.max_final_answer_tokens)
    # Reserve input headroom in the shared output ceiling so thinking+answer
    # budgets can't overflow the context window.
    model_ceiling = max_output_ceiling() if uses_max else None

    # For models that support separate reasoning_tokens parameter
    if needs_reasoning_params(model_name):
        # Set reasoning_tokens for Anthropic and Google models
        if inspect_model_str:
            if "anthropic" in inspect_model_str or "google" in inspect_model_str:
                # Anthropic/Google: reasoning_tokens IS the CoT token budget and
                # must be specified explicitly — don't silently guess a default.
                if resolved_thinking is None:
                    raise ReasoningTokensRequiredError(
                        f"Reasoning is enabled for '{model_name}' but 'max_thinking_tokens' "
                        f"is not set. This provider requires an explicit CoT token budget — "
                        f"set 'max_thinking_tokens' (an int or 'max') in the experiment config."
                    )
                reasoning_tokens = resolved_thinking
                # The thinking budget must leave room for the answer under the
                # model's total output ceiling (and stay < max_tokens, an Anthropic
                # API requirement). Only cap when a "max" sentinel was used.
                if model_ceiling is not None:
                    reasoning_tokens = min(reasoning_tokens, model_ceiling - 1)
                config_params["reasoning_tokens"] = reasoning_tokens

                # For Anthropic models: max_tokens must be greater than thinking.budget_tokens
                # This is an API requirement. We ensure max_tokens > reasoning_tokens.
                if "max_tokens" not in config_params:
                    # User-specified answer limit, else a high practical default.
                    max_answer = resolved_answer if resolved_answer is not None else 8192
                    max_tokens = max(max_answer, reasoning_tokens + 1)
                    if model_ceiling is not None:
                        max_tokens = min(max_tokens, model_ceiling)
                    config_params["max_tokens"] = max_tokens
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
            max_answer = resolved_answer if resolved_answer is not None else 2048
            if model_ceiling is not None:
                max_answer = min(max_answer, model_ceiling)
            config_params["max_tokens"] = max_answer
    else:
        # Together AI models (no separate reasoning_tokens): max_tokens controls total output
        # Total = max_thinking_tokens (for reasoning) + max_final_answer_tokens (for final answer)
        max_thinking = resolved_thinking if resolved_thinking is not None else 8192
        # For Together AI, max_final_answer_tokens in config represents the answer budget, not the total
        # If max_tokens was already set in config_params, use it as the answer budget
        # Otherwise, use config.max_final_answer_tokens or default
        if "max_tokens" in config_params:
            # max_tokens was already set (e.g., from config.max_final_answer_tokens in generation function)
            # Use it as the answer budget and add thinking budget
            max_answer = config_params["max_tokens"]
        else:
            # max_tokens not set yet, use config.max_final_answer_tokens or default
            max_answer = resolved_answer if resolved_answer is not None else 2048
        # Total = thinking + answer, capped at the model ceiling when "max" was used
        # (reasoning and answer share one output budget, so they can't both be full).
        total = max_thinking + max_answer
        if model_ceiling is not None:
            total = min(total, model_ceiling)
        config_params["max_tokens"] = total

        # For Together AI hybrid models (e.g., DeepSeek V3.1, Kimi K2.5),
        # the same endpoint serves both instruct and thinking modes.
        # Must pass `reasoning: {"enabled": true}` via extra_body to activate thinking.
        if needs_together_reasoning_activation(model_name):
            extra_body = config_params.get("extra_body", {})
            extra_body["reasoning"] = {"enabled": True}
            config_params["extra_body"] = extra_body


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
        (
            "UT",
            "IND-Q",
            "Pref",
        ): individual_query,  # Pref tasks use same function as Rec
        # MMLU-ICA capabilities-control eval (no tags/format dimension)
        ("NA", "MMLU-MC", "MMLU"): mmlu_multiple_choice,
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
    from self_rec_framework.src.inspect.config import ensure_evaluator_reasoning

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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
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
            # If ICL, wrap prompt as messages: ICL turns + final user message
            if icl_messages:
                prompt = icl_messages + [ChatMessageUser(content=prompt)]
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
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
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
    from self_rec_framework.src.inspect.config import ensure_evaluator_reasoning

    ensure_evaluator_reasoning(config, model_name)

    inspect_model_str: str = inspect_model_name(model_name)

    # Get model object (handles custom base URLs for XAI, etc.)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    # Check if evaluator is Anthropic (they have strict requirements for reasoning content)
    is_anthropic_evaluator = "anthropic" in inspect_model_str.lower()

    # Check if treatment models are Anthropic (to determine if we can use their reasoning format)
    # Get base model names (remove -thinking suffix and treatment suffixes)
    from self_rec_framework.src.helpers.model_names import get_base_model_name

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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
        # Format the generation prompt
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )

        # Build conversation history
        messages = []

        # Prepend ICL turn pairs if configured
        messages.extend(icl_messages)

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
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
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
    from self_rec_framework.src.inspect.config import ensure_evaluator_reasoning

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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
        # Format the prompt using the config template
        generation_prompt = config.generation_prompt.format(
            content=sample_data["content"]
        )
        prompt = config.SR_task_prompt.format(
            generation_prompt=generation_prompt,
            output1=sample_data["output1"],
            output2=sample_data["output2"],
        )
        if icl_messages:
            prompt = icl_messages + [ChatMessageUser(content=prompt)]

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
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
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
    from self_rec_framework.src.inspect.config import ensure_evaluator_reasoning

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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples with conversation history
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
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
        messages = []
        messages.extend(icl_messages)  # ICL turns first
        messages.extend([
            # First interaction with output
            ChatMessageUser(content=generation_prompt),
            ChatMessageAssistant(content=sample_data["output"]),
            # Final verification question
            ChatMessageUser(content=sr_task_prompt),
        ])

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
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
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
        if icl_messages:
            prompt = icl_messages + [ChatMessageUser(content=prompt)]

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
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
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

    # Load ICL examples if configured (per-sample resolver; returns [] if ICL not set)
    icl_resolve = _build_icl_resolver(config, dataset_samples, dataset_name, data_subset)

    # Create Inspect samples
    inspect_samples = []
    for sample_data in dataset_samples:
        uuid = sample_data["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
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
        if icl_messages:
            prompt = icl_messages + [ChatMessageUser(content=prompt)]

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
def mmlu_multiple_choice(
    model_name: str,
    treatment_name: str,
    dataset_name: str,
    data_subset: str,
    exp_config: ExperimentConfig,
    is_control: bool = True,
    task_name: str | None = None,
    logprobs: bool = False,
) -> Task:
    """MMLU multiple-choice eval with optional ICA (in-context attack) prefix.

    - Evaluation: MMLU questions loaded from data/input/{dataset_name}/{data_subset}/input.json.
    - ICA (if configured): QA pairs from exp_config.icl_model served from
      exp_config.icl_dataset/icl_data_subset, prepended as alternating user/assistant
      chat turns before the final MMLU question. Uses inspect_ai's built-in
      multiple_choice solver (formats A/B/C/D block, parses "ANSWER: X") and
      `choice` scorer.
    - `treatment_name` and `is_control` are unused (MMLU has no treatment/control pair).
    """
    config = exp_config

    inspect_model_str: str = inspect_model_name(model_name)
    inspect_model = _get_model_with_custom_base_url(model_name, inspect_model_str)

    samples = load_dataset_mmlu(dataset_name, data_subset)

    # ICA pool comes from a (potentially different) dataset — typically the SGTR
    # ShareGPT pool. Fall back to the eval dataset if not overridden.
    icl_dataset = config.icl_dataset or dataset_name

    # _build_icl_messages expects a generation_prompt template. For ICA we just
    # pass the author's ShareGPT user prompt through verbatim — the template is
    # a trivial passthrough. (Skipping _build_prompts also means this was never set.)
    if config.generation_prompt is None:
        config.generation_prompt = "{content}"

    icl_resolve = _build_icl_resolver(config, samples, icl_dataset,
                                      config.icl_data_subset or data_subset)

    inspect_samples = []
    for s in samples:
        uuid = s["metadata"]["uuid"]
        icl_messages = icl_resolve(uuid)
        if icl_messages:
            input_msgs = icl_messages + [ChatMessageUser(content=s["question"])]
        else:
            input_msgs = s["question"]
        inspect_samples.append(
            Sample(
                input=input_msgs,
                choices=s["choices"],
                target=s["answer"],
                metadata=s["metadata"],
            )
        )

    config_params: dict = {}
    if config.temperature is not None:
        config_params["temperature"] = config.temperature
    if config.max_final_answer_tokens is not None:
        config_params["max_tokens"] = config.max_final_answer_tokens
    if config.seed is not None:
        config_params["seed"] = config.seed
    _configure_thinking_model_params(
        model_name, config, config_params, inspect_model_str, samples=inspect_samples
    )

    return Task(
        dataset=inspect_samples,
        solver=multiple_choice(),
        scorer=choice(),
        model=inspect_model,
        config=GenerateConfig(**config_params) if config_params else None,
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
        model_name, config, generate_config_params, inspect_model_str, samples=inspect_samples
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
