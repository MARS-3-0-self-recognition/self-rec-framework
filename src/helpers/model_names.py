INSPECT_MODEL_NAMES: dict = {
    # OpenAI
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1-mini": "openai/gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "openai/gpt-4.1-2025-04-14",
    "gpt-5": "openai/gpt-5",
    # Anthropic
    "sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "sonnet-3.7": "anthropic/claude-3-7-sonnet-20250219",
    "haiku-3.5": "anthropic/claude-3-5-haiku-20241022",
    "haiku-4.5": "anthropic/claude-4-5-haiku-20251001",
    "opus-4.1": "anthropic/claude-opus-4-1-20250805",
    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    ## Together
    # Llama models
    "ll-3.1-8b": "together/meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
    "ll-3.1-70b": "together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
    "ll-3.1-405b": "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    # Qwen models (text-only, non-thinking)
    "qwen-2.5-7b": "together/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen-2.5-72b": "together/Qwen/Qwen2.5-72B-Instruct-Turbo",
    "qwen-3.0-80b": "together/Qwen/Qwen3-Next-80B-A3B-Instruct",
    # DeepSeek models (non-reasoning)
    "deepseek-3.0": "together/deepseek-ai/DeepSeek-V3",
    "deepseek-3.1": "together/deepseek-ai/DeepSeek-V3.1",
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

SHORT_MODEL_NAMES: dict = {v: k for k, v in INSPECT_MODEL_NAMES.items()}


def inspect_model_name(short_model_name: str) -> str:
    """
    Read from hard-coded INSPECT_MODEL_NAMES dict.
    """
    return INSPECT_MODEL_NAMES[short_model_name]


def short_model_name(model: str) -> str:
    """
    Read from inverse of hard-coded INSPECT_MODEL_NAMES dict.
    """
    return SHORT_MODEL_NAMES[model]
