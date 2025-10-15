INSPECT_MODEL_NAMES: dict = {
    "4o-mini": "openai/gpt-4o-mini",
    # Anthropic
    "3-5-sonnet": "anthropic/claude-3-5-sonnet-20241022",
    "3-5-haiku": "anthropic/claude-3-5-haiku-20241022",
    # Llama
    "ll-3-1-8b": "fireworks/accounts/fireworks/models/llama-v3p1-8b-instruct",
    "ll-3-1-70b": "fireworks/accounts/fireworks/models/llama-v3p1-70b-instruct",
    "ll-3-1-405b": "fireworks/accounts/fireworks/models/llama-v3p1-405b-instruct",
    # Qwen
    "qwen3-30b-a3b": "fireworks/accounts/fireworks/models/qwen3-30b-a3b",
    # DeepSeek
    "deepseek-v3": "fireworks/accounts/fireworks/models/deepseek-v3",
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
    return SHORT_MODEL_NAMES.inverse[model]
