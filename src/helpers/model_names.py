INSPECT_MODEL_NAMES: dict = {
    "4o-mini": "openai/gpt-4o-mini",
    "3-5-sonnet": "anthropic/claude-3-5-sonnet-20241022",
    "Qwen3-8B": "fireworks/accounts/fireworks/models/Qwen3-72B-Instruct",
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
