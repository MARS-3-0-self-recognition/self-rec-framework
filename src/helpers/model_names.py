from collections import defaultdict


INSPECT_MODEL_NAMES: dict = {
    # OpenAI
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1-mini": "openai/gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "openai/gpt-4.1-2025-04-14",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5": "openai/gpt-5",
    "gpt-oss-20b-thinking": "together/openai/gpt-oss-20b",
    "gpt-oss-120b-thinking": "together/openai/gpt-oss-120b",
    # "gpt-5-thinking": "openai/gpt-5",
    # "gpt-5-mini-thinking": "openai/gpt-5-mini",
    # "o1": "openai/o1",
    # "o1-thinking": "openai/o1",
    # "o1-mini": "openai/o1-mini",
    # "o1-mini-thinking": "openai/o1-mini",
    # "o3": "openai/o3",
    # "o3-thinking": "openai/o3",
    # "o3-mini": "openai/o3-mini",
    # "o3-mini-thinking": "openai/o3-mini",
    # "o4-mini": "openai/o4-mini",
    # "o4-mini-thinking": "openai/o4-mini",
    # Anthropic
    "sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "sonnet-4.5-thinking": "anthropic/claude-sonnet-4-5-20250929",
    "sonnet-3.7": "anthropic/claude-3-7-sonnet-20250219",
    "sonnet-3.7-thinking": "anthropic/claude-3-7-sonnet-20250219",
    "haiku-3.5": "anthropic/claude-3-5-haiku-20241022",
    "haiku-4.5": "anthropic/claude-4-5-haiku-20251001",
    "opus-4.1": "anthropic/claude-opus-4-1-20250805",
    "opus-4.1-thinking": "anthropic/claude-opus-4-1-20250805",
    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-flash-thinking": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-pro-thinking": "google/gemini-2.5-pro",
    # XAI (uses OpenAI provider with custom base URL via INSPECT_MODELS_OPENAI_GROK_3_MINI_BETA_*)
    "grok-3-mini": "openai/grok-3-mini",
    "grok-3-mini-thinking": "openai/grok-3-mini",
    ## Together-specific models
    # Llama models
    "ll-3.1-8b": "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "ll-3.1-70b": "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "ll-3.3-70b-dsR1-thinking": "together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "ll-3.1-405b": "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # Qwen models
    "qwen-2.5-7b": "together/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen-2.5-72b": "together/Qwen/Qwen2.5-72B-Instruct-Turbo",
    "qwen-3.0-80b": "together/Qwen/Qwen3-Next-80B-A3B-Instruct",
    "qwen-3.0-80b-thinking": "together/Qwen/Qwen3-Next-80B-A3B-Thinking",
    "qwen-3.0-235b": "together/Qwen/Qwen3-235B-A22B-Instruct-2507",
    "qwen-3.0-235b-thinking": "together/Qwen/Qwen3-235B-A22B-Thinking-2507",
    # DeepSeek models
    "deepseek-3.0": "together/deepseek-ai/DeepSeek-V3",
    "deepseek-3.1": "together/deepseek-ai/DeepSeek-V3.1",
    "deepseek-r1-thinking": "together/deepseek-ai/DeepSeek-R1",  # reasoning model
    # Moonshot
    "kimi-k2": "together/moonshotai/Kimi-K2-Instruct-0905",
    "kimi-k2-thinking": "together/moonshotai/Kimi-K2-Thinking",
    ## Fireworks
    # Llama
    "ll-3.1-8b_fw": "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
    "ll-3.1-70b_fw": "fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
    "ll-3.1-405b_fw": "fireworks/accounts/fireworks/models/llama-v3p1-405b-instruct",
    # Qwen
    "qwen-3.0-30b_fw": "fireworks/accounts/fireworks/models/qwen3-30b-a3b",
    "qwen-3.0-235b_fw": "fireworks/accounts/fireworks/models/qwen3-vl-235b-a22b-instruct",
    # DeepSeek
    # "deepseek-v3": "fireworks/accounts/fireworks/models/deepseek-v3-0324",
    "deepseek-3.1_fw": "fireworks/accounts/fireworks/models/deepseek-v3p1",
    "deepseek-r1_fw": "fireworks/accounts/fireworks/models/deepseek-r1-0528",  # reasoning model
}

# Build a reverse mapping that preserves ALL short names per inspect name.
# Multiple short names (e.g., "sonnet-4.5" and "sonnet-4.5-thinking") can
# legitimately map to the same inspect model, so we store a list rather than
# silently overwriting during dict inversion.
_INSPECT_TO_SHORT: dict[str, list[str]] = defaultdict(list)
for short, inspect in INSPECT_MODEL_NAMES.items():
    _INSPECT_TO_SHORT[inspect].append(short)

# Public constant (inspect model name -> list of possible short names)
SHORT_MODEL_NAMES: dict[str, list[str]] = dict(_INSPECT_TO_SHORT)


def inspect_model_name(short_model_name: str) -> str:
    """
    Read from hard-coded INSPECT_MODEL_NAMES dict.
    """
    return INSPECT_MODEL_NAMES[short_model_name]


def short_model_name(model: str) -> str:
    """
    Return a canonical short model name for a given inspect model name.

    When multiple short names map to the same inspect model (e.g.,
    "sonnet-4.5" and "sonnet-4.5-thinking"), we:
      - Prefer a "-thinking" variant if present (for reasoning-focused workflows)
      - Otherwise, fall back to the first name in sorted order for determinism
    """
    shorts = SHORT_MODEL_NAMES.get(model)
    if not shorts:
        raise KeyError(f"No short model name found for inspect model '{model}'")

    # Prefer thinking variants if available
    thinking_variants = [s for s in shorts if s.endswith("-thinking")]
    if thinking_variants:
        return sorted(thinking_variants)[0]

    # Fallback: deterministic choice among non-thinking variants
    return sorted(shorts)[0]


def is_thinking_model(treatment_name: str) -> bool:
    """
    Check if a treatment name corresponds to a thinking/reasoning model.

    Treatment names may have suffixes like "_caps_S2" or "_typos_S4",
    so we extract the base model name first.

    Args:
        treatment_name: Treatment name (may include suffixes like "_caps_S2")

    Returns:
        True if the base model is a thinking model, False otherwise
    """
    # Extract base model name by removing treatment suffixes
    # Treatment names can be: "model_name", "model_name_caps_S2", "model_name_typos_S4"
    base_name = treatment_name
    if "_caps_" in base_name:
        base_name = base_name.split("_caps_")[0]
    elif "_typos_" in base_name:
        base_name = base_name.split("_typos_")[0]

    # Check if it's a known thinking model
    # All thinking models now have "-thinking" suffix for consistent notation
    # Also check for OpenAI o-series and gpt-5 models (they're always thinking-capable,
    # but can be used with or without the -thinking suffix)
    return (
        "-thinking" in base_name
        or base_name
        == "deepseek-r1_fw"  # Fireworks DeepSeek-R1 (always thinking, no suffix variant)
        or base_name.startswith("o1")
        or base_name.startswith("o3")
        or base_name.startswith("o4")
        or base_name.startswith("gpt-5")
        or base_name.startswith(
            "grok-3-mini"
        )  # XAI Grok-3-mini requires reasoning_effort
    )


def get_base_model_name(model_name: str) -> str:
    """
    Get the base model name by removing the "-thinking" suffix.

    For models like "qwen-3.0-80b-thinking", this returns the full name
    since the thinking variant maps to a different endpoint.
    For models like "gemini-2.5-pro-thinking", this returns "gemini-2.5-pro"
    since they use the same endpoint with different API parameters.

    Args:
        model_name: Model name (may include "-thinking" suffix)

    Returns:
        Base model name without "-thinking" suffix
    """
    if model_name.endswith("-thinking"):
        return model_name[:-9]  # Remove "-thinking" suffix
    return model_name


def needs_reasoning_params(model_name: str) -> bool:
    """
    Check if a model needs reasoning API parameters (vs a different endpoint).

    Models like OpenAI o-series, GPT-5, Anthropic Claude 3.7/4, Google Gemini 2.5
    use the same endpoint but need reasoning_tokens/reasoning_effort parameters.

    Models like "qwen-3.0-80b-thinking" and "deepseek-r1-thinking" use a different endpoint entirely.

    Args:
        model_name: Model name (may include "-thinking" suffix)

    Returns:
        True if the model needs reasoning API parameters, False if it uses a different endpoint
    """
    if not model_name.endswith("-thinking"):
        return False

    base_name = get_base_model_name(model_name)

    # Models that use different endpoints (Together AI, etc.)
    # These map to different inspect model names
    # Even if the -thinking variant exists in INSPECT_MODEL_NAMES, they use different endpoints
    different_endpoint_models = [
        "qwen-3.0-80b",  # Maps to different endpoint
        "qwen-3.0-235b",  # Maps to different endpoint
        "deepseek-r1",  # Maps to different endpoint (DeepSeek-R1)
        "ll-3.3-70b-dsR1",  # DeepSeek-R1 distill uses separate endpoint
        "gpt-oss-20b",  # Together OpenAI OSS models use separate endpoint
        "gpt-oss-120b",  # Together OpenAI OSS models use separate endpoint
    ]

    if base_name in different_endpoint_models:
        return False

    # Check if the full model name exists in INSPECT_MODEL_NAMES
    # If it does, check the provider to determine if it needs reasoning params
    if model_name in INSPECT_MODEL_NAMES:
        inspect_name = INSPECT_MODEL_NAMES[model_name]
        # Together AI models use different endpoints, don't need reasoning params
        if "together" in inspect_name.lower():
            return False
        # OpenAI, Anthropic, Google models use same endpoint with reasoning params
        # (even though the -thinking variant exists in the mapping)
        return True

    # Models that use same endpoint but need API parameters
    # OpenAI o-series, GPT-5, Anthropic Claude 3.7/4, Google Gemini 2.5
    return True
