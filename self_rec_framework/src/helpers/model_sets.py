_MODEL_SETS: dict[str, list[str]] = {
    "test": [
        "ll-3.1-8b",
        "qwen-2.5-7b",
    ],
    "tutorial": [
        "ll-3.1-8b",
        "qwen-2.5-7b",
    ],
    "sgtr-training-judges": [
        "gpt-oss-20b-thinking",
        "ll-3.1-8b",
        "qwen-3.0-30b-thinking",
    ],
    "sgtr-training-judges-adversarial": [
        "gpt-oss-20b-thinking",
        "qwen-3.0-30b-thinking",
    ],
    "sgtr-training-generators": [
        "qwen-2.5-7b",
        "qwen-3.0-30b-thinking",
        "opus-4.1-thinking",
        "ll-3.1-8b",
        "gpt-oss-20b-thinking",
        "gpt-oss-120b-thinking",
    ],
    "gen_cot": [
        "gpt-oss-20b-thinking",
        "gpt-oss-120b-thinking",
        "glm-4.5-air-thinking",
        "glm-4.7-thinking",
        "minimax-m2.5-thinking",
        "qwen-3.0-235b-thinking",
        "deepseek-3.1-thinking",
        "deepseek-r1-0528-thinking",
        "kimi-k2.5-thinking",
    ],
    "gen_cot_test": [
        "gpt-oss-20b-thinking",
        "gpt-oss-120b-thinking",
    ],
    "dr_colm": [
        # 13 instruct models common to ICML_01/02 and COLM_01/02 experiments
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
        "sonnet-4.5",
        "opus-4.1",
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        "ll-3.1-8b",
        "qwen-2.5-7b",
        "qwen-3.0-80b",
        "deepseek-3.1",
        "kimi-k2",
    ],
    "dr": [
        # OpenAI (weakest to strongest)
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
        # Anthropic (weakest to strongest)
        "sonnet-4.5",
        "opus-4.1",
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite",
        "gemini-2.0-flash",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b",
        #"ll-3.1-70b", # still available but not working
        # Together AI - Qwen (weakest to strongest)
        "qwen-2.5-7b",
        #"qwen-2.5-72b", # still available but not working
        "qwen-3.5-27b",
        "qwen-3.0-80b",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.1",
        # Moonshot (weakest to strongest)
        "kimi-k2",
    ],
    "test_dr": [
        # OpenAI (weakest to strongest)
        "gpt-4o-mini",
        # Anthropic (weakest to strongest)
        "haiku-3.5",
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b",
        # Together AI - Qwen (weakest to strongest)
        "qwen-2.5-7b",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.1",
        # XAI (weakest to strongest)
        "grok-4.1-fast",
        # Moonshot (weakest to strongest)
        "kimi-k2",
    ],
    "dr_prompt_cot": [
        # OpenAI (weakest to strongest)
        "gpt-4o-mini",
        "gpt-4.1-mini",
        "gpt-4o",
        "gpt-4.1",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b",
        "ll-3.1-70b",
        "ll-3.1-405b",
        # Together AI - Qwen (weakest to strongest)
        "qwen-2.5-7b",
        "qwen-2.5-72b",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.1",
    ],
    "eval_cot-r": [
        # OpenAI (weakest to strongest)
        ## OS
        "gpt-oss-120b-thinking",
        ## Not OS
        # Anthropic (weakest to strongest)
        "opus-4.1-thinking",
        # Google Gemini (weakest to strongest)
        "gemini-2.5-pro-thinking",
        # Together
        ## OS
        "qwen-3.0-80b-thinking",
        "deepseek-r1-thinking",
        "kimi-k2-thinking",
    ],
    "test_eval_cot-r": [
        # Together
        ## OS
        "qwen-3.0-80b-thinking",
        "kimi-k2-thinking",
    ],
    "eval_cot-r_and_cot-i": [
        ### CoT-R
        # OpenAI (weakest to strongest)
        ## OS
        "gpt-oss-20b-thinking",
        "gpt-oss-120b-thinking",
        ## Not OS
        "o3-thinking",
        "o3-mini-thinking",
        # Anthropic (weakest to strongest)
        "haiku-3.5-thinking",
        "sonnet-3.7-thinking",
        "sonnet-4.5-thinking",
        "opus-4.1-thinking",
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite-thinking",
        "gemini-2.0-flash-thinking",
        "gemini-2.5-flash-thinking",
        "gemini-2.5-pro-thinking",
        # Together
        ## OS
        "sonnet-3.7-thinking",
        "grok-3-mini-thinking",
        "ll-3.3-70b-dsR1-thinking",
        "qwen-3.0-80b-thinking",
        "qwen-3.0-235b-thinking",
        "deepseek-r1-thinking",
        "kimi-k2-thinking",
        ### CoT-I
        # OpenAI (weakest to strongest)
        "gpt-4o-mini-thinking",
        "gpt-4.1-mini-thinking",
        "gpt-4o-thinking",
        "gpt-4.1-thinking",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-8b-thinking",
        "ll-3.1-70b-thinking",
        "ll-3.1-405b-thinking",
        # Together AI - DeepSeek (weakest to strongest)
        "deepseek-3.1-thinking",
    ],
    "test_eval_cot-r_and_cot-i": [
        # Anthropic (weakest to strongest)
        "haiku-3.5-thinking",
        # Google Gemini (weakest to strongest)
        "gemini-2.0-flash-lite-thinking",
        # Together AI - Llama (weakest to strongest)
        "ll-3.1-70b-thinking",
    ],
}

# Computed sets (depend on other sets defined above)
_MODEL_SETS["eval_cot-r_and_dr"] = _MODEL_SETS["eval_cot-r"] + _MODEL_SETS["dr"]
_MODEL_SETS["cot-r_and_dr"] = _MODEL_SETS["eval_cot-r_and_dr"]


def get_model_set(model_set_name: str | None = None) -> list[str]:
    """
    Define the canonical order for models in the pivot table and heatmap.

    Models are organized by company/provider, then ordered from weakest to strongest.

    Args:
        model_set_name: Name of the model set. If None, returns empty list to indicate
                       that models should be used in the order they were provided.

    Returns:
        List of model names in display order, or empty list if model_set_name is None
    """
    if model_set_name is None:
        return []

    key = model_set_name.lower()
    if key not in _MODEL_SETS:
        valid = ", ".join(_MODEL_SETS.keys())
        raise ValueError(f"Unknown model set: {model_set_name}. Valid sets: {valid}")

    return _MODEL_SETS[key]
