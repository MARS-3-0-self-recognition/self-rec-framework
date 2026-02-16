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
    if model_set_name.lower() == "test":
        return [
            "ll-3.1-8b",
            "qwen-2.5-7b",
        ]
    if model_set_name.lower() == "gen_cot":
        return [
            "gpt-oss-20b-thinking",
            "gpt-oss-120b-thinking",
            "sonnet-3.7-thinking",
            "grok-3-mini-thinking",
            "ll-3.3-70b-dsR1-thinking",
            "qwen-3.0-80b-thinking",
            "qwen-3.0-235b-thinking",
            "deepseek-r1-thinking",
            "kimi-k2-thinking",
        ]
    elif model_set_name.lower() == "dr":
        return [
            # OpenAI (weakest to strongest)
            "gpt-4o-mini",
            "gpt-4.1-mini",
            "gpt-4o",
            "gpt-4.1",
            # Anthropic (weakest to strongest)
            "haiku-3.5",
            "sonnet-3.7",
            "sonnet-4.5",
            "opus-4.1",
            # Google Gemini (weakest to strongest)
            "gemini-2.0-flash-lite",
            "gemini-2.0-flash",
            # Together AI - Llama (weakest to strongest)
            "ll-3.1-8b",
            "ll-3.1-70b",
            "ll-3.1-405b",
            # Together AI - Qwen (weakest to strongest)
            "qwen-2.5-7b",
            "qwen-2.5-72b",
            "qwen-3.0-80b",
            # "qwen-3.0-235b",
            # Together AI - DeepSeek (weakest to strongest)
            "deepseek-3.1",
            # XAI (weakest to strongest)
            #"grok-4.1-fast",
            # Moonshot (weakest to strongest)
            "kimi-k2",
        ]
    elif model_set_name.lower() == "test_dr":
        return [
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
        ]
    elif model_set_name.lower() == "dr_prompt_cot":
        return [
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
        ]
    elif model_set_name.lower() == "eval_cot-r":
        return [
            # OpenAI (weakest to strongest)
            ## OS
            # "gpt-oss-20b-thinking",
            "gpt-oss-120b-thinking",
            ## Not OS
            # "o3-thinking",
            # "o3-mini-thinking",
            # Anthropic (weakest to strongest)
            # "haiku-3.5-thinking",
            # "sonnet-3.7-thinking",
            # "sonnet-4.5-thinking",
            "opus-4.1-thinking",
            # Google Gemini (weakest to strongest)
            # "gemini-2.0-flash-lite-thinking",
            # "gemini-2.0-flash-thinking",
            # "gemini-2.5-flash-thinking",
            "gemini-2.5-pro-thinking",
            # Together
            ## OS
            # "grok-3-mini-thinking",
            # "ll-3.3-70b-dsR1-thinking",
            "qwen-3.0-80b-thinking",
            # "qwen-3.0-235b-thinking",
            "deepseek-r1-thinking",
            "kimi-k2-thinking",
        ]
    elif model_set_name.lower() == "test_eval_cot-r":
        return [
            # OpenAI (weakest to strongest)
            ## OS
            # "gpt-oss-20b-thinking",
            # "gpt-oss-120b-thinking",
            ## Not OS
            # "o3-thinking",
            # "o3-mini-thinking",
            # Anthropic (weakest to strongest)
            # "haiku-3.5-thinking",
            # "sonnet-3.7-thinking",
            # "sonnet-4.5-thinking",
            # "opus-4.1-thinking",
            # Google Gemini (weakest to strongest)
            # "gemini-2.0-flash-lite-thinking",
            # "gemini-2.0-flash-thinking",
            # "gemini-2.5-flash-thinking",
            # "gemini-2.5-pro-thinking",
            # Together
            ## OS
            # "grok-3-mini-thinking",
            # "ll-3.3-70b-dsR1-thinking",
            "qwen-3.0-80b-thinking",
            # "qwen-3.0-235b-thinking",
            # "deepseek-r1-thinking",
            "kimi-k2-thinking",
        ]
    elif (
        model_set_name.lower() == "eval_cot-r_and_dr"
        or model_set_name.lower() == "cot-r_and_dr"
    ):
        return get_model_set("eval_cot-r") + get_model_set("dr")
    elif model_set_name.lower() == "eval_cot-r_and_cot-i":
        return [
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
        ]
    elif model_set_name.lower() == "test_eval_cot-r_and_cot-i":
        return [
            # Anthropic (weakest to strongest)
            "haiku-3.5-thinking",
            # Google Gemini (weakest to strongest)
            "gemini-2.0-flash-lite-thinking",
            # Together AI - Llama (weakest to strongest)
            "ll-3.1-70b-thinking",
        ]
    else:
        raise ValueError(
            f"Unknown model set: {model_set_name}. Valid sets: gen_cot, dr, eval_cot, eval_cot_and_dr"
        )
