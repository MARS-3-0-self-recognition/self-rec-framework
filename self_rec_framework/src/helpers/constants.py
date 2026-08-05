import uuid


MY_DATASET_NAMESPACE = uuid.UUID("7bae975c-d5fb-416b-a803-4582e5ef1a2c")


# --- Model reasoning-endpoint classification (used by model_names.py) ---

# Base model names whose "-thinking" variant maps to a DIFFERENT inference endpoint
# (not the same endpoint with reasoning params). Even if the -thinking variant exists
# in INSPECT_MODEL_NAMES, these use a separate endpoint.
DIFFERENT_ENDPOINT_MODELS = [
    "qwen-3.0-80b",  # Maps to different endpoint
    "qwen-3.0-235b",  # Maps to different endpoint
    "deepseek-r1",  # Maps to different endpoint (DeepSeek-R1)
    "ll-3.3-70b-dsR1",  # DeepSeek-R1 distill uses separate endpoint
    "gpt-oss-20b",  # Together OpenAI OSS models use separate endpoint
    "gpt-oss-120b",  # Together OpenAI OSS models use separate endpoint
]

# Native reasoning models (COT-R): use different endpoints, are always-reasoning, or are
# dual-mode models used with -thinking for reasoning. These have their own generated data
# with the -thinking suffix. COT-I (instruction-tuned, same endpoint) use base model data
# and are NOT in this list.
NATIVE_REASONING_BASES = [
    # Together AI models with separate thinking endpoints
    "qwen-3.0-30b",
    "qwen-3.0-80b",
    "qwen-3.0-235b",
    "qwen-3.5-27b",
    "deepseek-r1",
    "deepseek-r1-0528",
    "ll-3.3-70b-dsR1",
    "ll-70B-dsr1",
    "gpt-oss-20b",
    "gpt-oss-120b",
    "kimi-k2",
    # Together AI hybrid models (same endpoint, reasoning activated via API param)
    "deepseek-3.1",
    "kimi-k2.5",
    # Together AI models that always think (no non-thinking variant)
    "minimax-m2.5",
    "glm-4.5-air",
    "glm-4.7",
    # OpenAI o-series (always reasoning)
    "o3",
    "o3-mini",
    # OpenAI GPT-5 (dual-mode)
    "gpt-5",
    "gpt-5-mini",
    # XAI Grok with separate reasoning endpoint
    "grok-4.1-fast",
    # Dual-mode models: -thinking indicates COT-R (use own reasoning data), not COT-I
    "opus-4.1",
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "sonnet-4.5",
    "sonnet-3.7",
    "haiku-3.5",
    "haiku-4.5",
]
