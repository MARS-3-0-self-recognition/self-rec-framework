from collections import defaultdict


INSPECT_MODEL_NAMES: dict = {
    # OpenAI
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4.1-mini": "openai/gpt-4.1-mini-2025-04-14",
    "gpt-4.1": "openai/gpt-4.1-2025-04-14",
    "gpt-5-mini": "openai/gpt-5-mini",
    "gpt-5-mini-thinking": "openai/gpt-5-mini",
    "gpt-5": "openai/gpt-5",
    "gpt-5-thinking": "openai/gpt-5",
    "gpt-oss-20b-thinking": "together/openai/gpt-oss-20b",
    "gpt-oss-120b-thinking": "together/openai/gpt-oss-120b",
    "o3": "openai/o3-2025-04-16",
    "o3-thinking": "openai/o3-2025-04-16",
    "o3-mini": "openai/o3-mini-2025-01-31",
    "o3-mini-thinking": "openai/o3-mini-2025-01-31",
    # Anthropic
    "sonnet-4.5": "anthropic/claude-sonnet-4-5-20250929",
    "sonnet-4.5-thinking": "anthropic/claude-sonnet-4-5-20250929",
    "sonnet-3.7": "anthropic/claude-3-7-sonnet-20250219",
    "sonnet-3.7-thinking": "anthropic/claude-3-7-sonnet-20250219",
    "haiku-3.5": "anthropic/claude-3-5-haiku-20241022",
    "haiku-3.5-thinking": "anthropic/claude-3-5-haiku-20241022",
    "haiku-4.5": "anthropic/claude-4-5-haiku-20251001",
    "haiku-4.5-thinking": "anthropic/claude-4-5-haiku-20251001",
    "opus-4.1": "anthropic/claude-opus-4-1-20250805",
    "opus-4.1-thinking": "anthropic/claude-opus-4-1-20250805",
    # Google
    "gemini-2.0-flash": "google/gemini-2.0-flash",
    "gemini-2.0-flash-thinking": "google/gemini-2.0-flash",
    "gemini-2.0-flash-lite": "google/gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-thinking": "google/gemini-2.0-flash-lite",
    "gemini-2.5-flash": "google/gemini-2.5-flash",
    "gemini-2.5-flash-thinking": "google/gemini-2.5-flash",
    "gemini-2.5-pro": "google/gemini-2.5-pro",
    "gemini-2.5-pro-thinking": "google/gemini-2.5-pro",
    # XAI (uses OpenAI provider with custom base URL via INSPECT_MODELS_OPENAI_GROK_3_MINI_BETA_*)
    "grok-3-mini": "openai/grok-3-mini",
    "grok-3-mini-thinking": "openai/grok-3-mini",
    "grok-4.1-fast": "openai/grok-4-1-fast-non-reasoning",
    "grok-4.1-fast-thinking": "openai/grok-4-1-fast-reasoning",
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

# Model parameter counts (in billions, unless specified with 'T' for trillions)
# Values are based on model names or official documentation only.
# For MoE models, values represent total parameters (not active per token).
# Use MODEL_PARAMETER_COUNTS_ESTIMATED for estimated values.
MODEL_PARAMETER_COUNTS: dict[str, str] = {
    # OpenAI
    "gpt-4o-mini": "unknown",
    "gpt-4o": "unknown",
    "gpt-4.1-mini": "unknown",
    "gpt-4.1": "unknown",
    "gpt-5-mini": "unknown",
    "gpt-5-mini-thinking": "unknown",
    "gpt-5": "unknown",
    "gpt-5-thinking": "unknown",
    "gpt-oss-20b-thinking": "20B",  # From model name
    "gpt-oss-120b-thinking": "120B",  # From model name
    "o3": "unknown",
    "o3-thinking": "unknown",
    "o3-mini": "36B",  # Confirmed
    "o3-mini-thinking": "36B",  # Confirmed
    # Anthropic
    "sonnet-4.5": "unknown",
    "sonnet-4.5-thinking": "unknown",
    "sonnet-3.7": "unknown",
    "sonnet-3.7-thinking": "unknown",
    "haiku-3.5": "unknown",
    "haiku-3.5-thinking": "unknown",
    "haiku-4.5": "unknown",
    "haiku-4.5-thinking": "unknown",
    "opus-4.1": "unknown",
    "opus-4.1-thinking": "unknown",
    # Google
    "gemini-2.0-flash": "20B",  # Confirmed
    "gemini-2.0-flash-thinking": "20B",  # Confirmed
    "gemini-2.0-flash-lite": "unknown",
    "gemini-2.0-flash-lite-thinking": "unknown",
    "gemini-2.5-flash": "5B",  # Confirmed
    "gemini-2.5-flash-thinking": "5B",  # Confirmed
    "gemini-2.5-pro": "unknown",
    "gemini-2.5-pro-thinking": "unknown",
    # XAI
    "grok-3-mini": "unknown",
    "grok-3-mini-thinking": "unknown",
    "grok-4.1-fast": "unknown",
    "grok-4.1-fast-thinking": "unknown",
    # Together - Llama
    "ll-3.1-8b": "8B",  # From model name
    "ll-3.1-70b": "70B",  # From model name
    "ll-3.3-70b-dsR1-thinking": "70B",  # From model name (distilled)
    "ll-3.1-405b": "405B",  # From model name
    # Together - Qwen
    "qwen-2.5-7b": "7B",  # From model name
    "qwen-2.5-72b": "72B",  # From model name
    "qwen-3.0-80b": "80B",  # From model name
    "qwen-3.0-80b-thinking": "80B",  # From model name
    "qwen-3.0-235b": "235B",  # From model name
    "qwen-3.0-235b-thinking": "235B",  # From model name
    # Together - DeepSeek (MoE: 671B total, 37B active per token)
    "deepseek-3.0": "671B",  # Total parameters (MoE)
    "deepseek-3.1": "671B",  # Total parameters (MoE)
    "deepseek-r1-thinking": "671B",  # Total parameters (MoE, same architecture as V3)
    # Moonshot
    "kimi-k2": "unknown",
    "kimi-k2-thinking": "unknown",
    # Fireworks - Llama
    "ll-3.1-8b_fw": "8B",  # From model name
    "ll-3.1-70b_fw": "70B",  # From model name
    "ll-3.1-405b_fw": "405B",  # From model name
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": "30B",  # From model name
    "qwen-3.0-235b_fw": "235B",  # From model name
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": "671B",  # Total parameters (MoE)
    "deepseek-r1_fw": "671B",  # Total parameters (MoE)
}

# Estimated parameter counts for models where official values are not available.
# These are research estimates based on model performance, architecture comparisons,
# and industry analysis. Values should be treated as approximate.
MODEL_PARAMETER_COUNTS_ESTIMATED: dict[str, str] = {
    # OpenAI
    "gpt-4o-mini": "8B",  # Estimated, similar to Claude 3 Haiku
    "gpt-4o": "1.8T",  # Estimated, similar to GPT-4
    "gpt-4.1-mini": "8B",  # Estimated, similar to gpt-4o-mini
    "gpt-4.1": "1.8T",  # Estimated, similar to gpt-4o
    "gpt-5-mini": "unknown",  # No reliable estimate available
    "gpt-5-mini-thinking": "unknown",  # No reliable estimate available
    "gpt-5": "1.8T",  # Estimated, similar to GPT-4
    "gpt-5-thinking": "1.8T",  # Estimated, similar to GPT-4
    "o3": "unknown",  # No reliable estimate available (likely > o3-mini)
    "o3-thinking": "unknown",  # No reliable estimate available
    # Anthropic
    "sonnet-4.5": "unknown",  # No reliable estimate available
    "sonnet-4.5-thinking": "unknown",  # No reliable estimate available
    "sonnet-3.7": "unknown",  # No reliable estimate available
    "sonnet-3.7-thinking": "unknown",  # No reliable estimate available
    "haiku-3.5": "8B",  # Estimated, similar to GPT-4o-mini
    "haiku-3.5-thinking": "8B",  # Estimated, same as haiku-3.5
    "haiku-4.5": "8B",  # Estimated, similar to haiku-3.5
    "haiku-4.5-thinking": "8B",  # Estimated, same as haiku-4.5
    "opus-4.1": "unknown",  # No reliable estimate available
    "opus-4.1-thinking": "unknown",  # No reliable estimate available
    # Google
    "gemini-2.0-flash-lite": "10B",  # Estimated, smaller than flash (20B)
    "gemini-2.0-flash-lite-thinking": "10B",  # Estimated, same as flash-lite
    "gemini-2.5-pro": "unknown",  # No reliable estimate available (likely > flash)
    "gemini-2.5-pro-thinking": "unknown",  # No reliable estimate available
    # XAI
    "grok-3-mini": "unknown",  # No reliable estimate available
    "grok-3-mini-thinking": "unknown",  # No reliable estimate available
    "grok-4.1-fast": "unknown",  # No reliable estimate available
    "grok-4.1-fast-thinking": "unknown",  # No reliable estimate available
    # Moonshot
    "kimi-k2": "unknown",  # No reliable estimate available
    "kimi-k2-thinking": "unknown",  # No reliable estimate available
}

# Model release dates (YYYY-MM-DD format)
# Values are based on official announcements, model names with embedded dates, or confirmed release dates.
# Use MODEL_RELEASE_DATES_ESTIMATED for estimated values.
MODEL_RELEASE_DATES: dict[str, str] = {
    # OpenAI
    "gpt-4o-mini": "2024-07-18",  # Confirmed
    "gpt-4o": "2024-05",  # Confirmed (month only)
    "gpt-4.1-mini": "2025-04-14",  # From model name
    "gpt-4.1": "2025-04-14",  # From model name
    "gpt-5-mini": "unknown",
    "gpt-5-mini-thinking": "unknown",
    "gpt-5": "2025-08-07",  # Confirmed
    "gpt-5-thinking": "2025-08-07",  # Confirmed
    "gpt-oss-20b-thinking": "unknown",
    "gpt-oss-120b-thinking": "unknown",
    "o3": "2025-04-16",  # From model name
    "o3-thinking": "2025-04-16",  # From model name
    "o3-mini": "2025-01-31",  # From model name
    "o3-mini-thinking": "2025-01-31",  # From model name
    # Anthropic
    "sonnet-4.5": "2025-09-29",  # From model name
    "sonnet-4.5-thinking": "2025-09-29",  # From model name
    "sonnet-3.7": "2025-02-19",  # From model name
    "sonnet-3.7-thinking": "2025-02-19",  # From model name
    "haiku-3.5": "2024-10-22",  # From model name
    "haiku-3.5-thinking": "2024-10-22",  # From model name
    "haiku-4.5": "2025-10-01",  # From model name
    "haiku-4.5-thinking": "2025-10-01",  # From model name
    "opus-4.1": "2025-08-05",  # From model name
    "opus-4.1-thinking": "2025-08-05",  # From model name
    # Google
    "gemini-2.0-flash": "2025-02-05",  # General availability
    "gemini-2.0-flash-thinking": "2025-02-05",  # General availability
    "gemini-2.0-flash-lite": "2025-02-05",  # Preview release
    "gemini-2.0-flash-lite-thinking": "2025-02-05",  # Preview release
    "gemini-2.5-flash": "2025-06-17",  # General availability
    "gemini-2.5-flash-thinking": "2025-06-17",  # General availability
    "gemini-2.5-pro": "2025-06-17",  # General availability
    "gemini-2.5-pro-thinking": "2025-06-17",  # General availability
    # XAI
    "grok-3-mini": "2025-02-14",  # Confirmed
    "grok-3-mini-thinking": "2025-02-14",  # Confirmed
    "grok-4.1-fast": "2024-11-17",  # Confirmed
    "grok-4.1-fast-thinking": "2024-11-17",  # Confirmed
    # Together - Llama
    "ll-3.1-8b": "2024-07-23",  # Confirmed
    "ll-3.1-70b": "2024-07-23",  # Confirmed
    "ll-3.3-70b-dsR1-thinking": "unknown",
    "ll-3.1-405b": "2024-07-23",  # Confirmed
    # Together - Qwen
    "qwen-2.5-7b": "2024-09-19",  # Confirmed
    "qwen-2.5-72b": "2024-09-19",  # Confirmed
    "qwen-3.0-80b": "unknown",
    "qwen-3.0-80b-thinking": "unknown",
    "qwen-3.0-235b": "2025-07",  # Estimated from model name (2507 = July 2025)
    "qwen-3.0-235b-thinking": "2025-07",  # Estimated from model name
    # Together - DeepSeek
    "deepseek-3.0": "2024-12",  # Confirmed (month only)
    "deepseek-3.1": "2025-08-19",  # Confirmed
    "deepseek-r1-thinking": "2025-05",  # Confirmed (month only, version R1-0528)
    # Moonshot
    "kimi-k2": "2025-07-12",  # Confirmed
    "kimi-k2-thinking": "2025-11-06",  # Confirmed
    # Fireworks - Llama
    "ll-3.1-8b_fw": "2024-07-23",  # Same as Together version
    "ll-3.1-70b_fw": "2024-07-23",  # Same as Together version
    "ll-3.1-405b_fw": "2024-07-23",  # Same as Together version
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": "unknown",
    "qwen-3.0-235b_fw": "2025-07",  # Estimated, same as Together version
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": "2025-08-19",  # Same as Together version
    "deepseek-r1_fw": "2025-05-28",  # From model name (0528 = May 28)
}

# Estimated release dates for models where official dates are not available.
# These are estimates based on model naming patterns, release timelines, and industry analysis.
MODEL_RELEASE_DATES_ESTIMATED: dict[str, str] = {
    # OpenAI
    "gpt-5-mini": "2025-08-07",  # Estimated, same as GPT-5
    "gpt-5-mini-thinking": "2025-08-07",  # Estimated, same as GPT-5
    "gpt-oss-20b-thinking": "unknown",  # No reliable estimate available
    "gpt-oss-120b-thinking": "unknown",  # No reliable estimate available
    # Together - Llama
    "ll-3.3-70b-dsR1-thinking": "2025-05",  # Estimated, based on DeepSeek-R1 release
    # Together - Qwen
    "qwen-3.0-80b": "2025-07",  # Estimated, similar to qwen-3.0-235b
    "qwen-3.0-80b-thinking": "2025-07",  # Estimated, same as qwen-3.0-80b
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": "2025-07",  # Estimated, similar to other Qwen 3.0 models
}

# Model capability tiers (1 = lowest, 5 = highest)
# Tiers are based on relative capabilities considering model size, version progression,
# release date, and known performance benchmarks within the model set.
MODEL_CAPABILITY_TIERS: dict[str, int] = {
    # OpenAI
    "gpt-4o-mini": 1,  # Small model (~8B estimated)
    "gpt-4o": 4,  # Large frontier model (~1.8T estimated)
    "gpt-4.1-mini": 1,  # Small model (~8B estimated)
    "gpt-4.1": 4,  # Large frontier model (~1.8T estimated), newer than 4o
    "gpt-5-mini": 2,  # Medium model, newer generation
    "gpt-5-mini-thinking": 2,  # Medium model, newer generation
    "gpt-5": 5,  # Latest flagship frontier model (~1.8T estimated)
    "gpt-5-thinking": 5,  # Latest flagship frontier model
    "gpt-oss-20b-thinking": 2,  # Medium model (20B)
    "gpt-oss-120b-thinking": 3,  # Large model (120B)
    "o3": 5,  # Latest reasoning-focused frontier model
    "o3-thinking": 5,  # Latest reasoning-focused frontier model
    "o3-mini": 3,  # Medium-large reasoning model (36B)
    "o3-mini-thinking": 3,  # Medium-large reasoning model
    # Anthropic
    "sonnet-4.5": 5,  # Latest flagship Claude model (newest version)
    "sonnet-4.5-thinking": 5,  # Latest flagship Claude model
    "sonnet-3.7": 4,  # Large frontier model (earlier version)
    "sonnet-3.7-thinking": 4,  # Large frontier model
    "haiku-3.5": 1,  # Small model (~8B estimated)
    "haiku-3.5-thinking": 1,  # Small model
    "haiku-4.5": 1,  # Small model (~8B estimated), newer version
    "haiku-4.5-thinking": 1,  # Small model, newer version
    "opus-4.1": 5,  # Flagship Claude model (highest tier)
    "opus-4.1-thinking": 5,  # Flagship Claude model
    # Google
    "gemini-2.0-flash": 2,  # Medium model (20B)
    "gemini-2.0-flash-thinking": 2,  # Medium model
    "gemini-2.0-flash-lite": 1,  # Small model (~10B estimated)
    "gemini-2.0-flash-lite-thinking": 1,  # Small model
    "gemini-2.5-flash": 1,  # Small model (5B), newer but smaller
    "gemini-2.5-flash-thinking": 1,  # Small model
    "gemini-2.5-pro": 4,  # Large flagship model
    "gemini-2.5-pro-thinking": 4,  # Large flagship model
    # XAI
    "grok-3-mini": 1,  # Small model (mini variant)
    "grok-3-mini-thinking": 1,  # Small model
    "grok-4.1-fast": 3,  # Medium-large model (fast variant)
    "grok-4.1-fast-thinking": 3,  # Medium-large model
    # Together - Llama
    "ll-3.1-8b": 1,  # Small model (8B)
    "ll-3.1-70b": 3,  # Medium-large model (70B)
    "ll-3.3-70b-dsR1-thinking": 3,  # Medium-large model (70B, distilled)
    "ll-3.1-405b": 5,  # Very large model (405B)
    # Together - Qwen
    "qwen-2.5-7b": 1,  # Small model (7B)
    "qwen-2.5-72b": 3,  # Medium-large model (72B)
    "qwen-3.0-80b": 3,  # Medium-large model (80B), newer version
    "qwen-3.0-80b-thinking": 3,  # Medium-large model
    "qwen-3.0-235b": 4,  # Large model (235B)
    "qwen-3.0-235b-thinking": 4,  # Large model
    # Together - DeepSeek (MoE: 671B total, 37B active per token)
    "deepseek-3.0": 5,  # Very large MoE model (671B total)
    "deepseek-3.1": 5,  # Very large MoE model (671B total), newer version
    "deepseek-r1-thinking": 5,  # Very large reasoning model (671B total)
    # Moonshot
    "kimi-k2": 4,  # Large flagship model
    "kimi-k2-thinking": 4,  # Large flagship reasoning model
    # Fireworks - Llama
    "ll-3.1-8b_fw": 1,  # Small model (8B)
    "ll-3.1-70b_fw": 3,  # Medium-large model (70B)
    "ll-3.1-405b_fw": 5,  # Very large model (405B)
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": 2,  # Medium model (30B)
    "qwen-3.0-235b_fw": 4,  # Large model (235B)
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": 5,  # Very large MoE model (671B total)
    "deepseek-r1_fw": 5,  # Very large reasoning model (671B total)
}
# LM Arena rankings from https://lmarena.ai/leaderboard/text
# Rankings are based on Elo scores from the leaderboard (as of Jan 16, 2026).
# Lower rank number = higher position on leaderboard (rank 1 is best).
# Models not found on leaderboard are marked as None.
LM_ARENA_RANKINGS: dict[str, int | None] = {
    # OpenAI
    "gpt-4o-mini": 152,  # gpt-4o-mini-2024-07-18, score: 1318
    "gpt-4o": 18,  # chatgpt-4o-latest-20250326, score: 1442 (using latest version)
    "gpt-4.1-mini": 80,  # gpt-4.1-mini-2025-04-14, score: 1382
    "gpt-4.1": 47,  # gpt-4.1-2025-04-14, score: 1413
    "gpt-5-mini": None,  # Not found on leaderboard
    "gpt-5-mini-thinking": None,  # Not found on leaderboard
    "gpt-5": 21,  # gpt-5.1, score: 1435 (using closest match)
    "gpt-5-thinking": 21,  # gpt-5.1, score: 1435 (using closest match)
    "gpt-oss-20b-thinking": 151,  # gpt-oss-20b, score: 1318
    "gpt-oss-120b-thinking": 101,  # gpt-oss-120b, score: 1354
    "o3": 24,  # o3-2025-04-16, score: 1433
    "o3-thinking": 24,  # o3-2025-04-16, score: 1433
    "o3-mini": 107,  # o3-mini, score: 1348
    "o3-mini-thinking": 107,  # o3-mini, score: 1348
    # Anthropic
    "sonnet-4.5": 13,  # claude-sonnet-4-5-20250929, score: 1450
    "sonnet-4.5-thinking": 11,  # claude-sonnet-4-5-20250929-thinking-32k, score: 1451
    "sonnet-3.7": None,  # claude-3-7-sonnet-20250219 not found (thinking version at rank 73)
    "sonnet-3.7-thinking": 73,  # claude-3-7-sonnet-20250219-thinking-32k, score: 1389
    "haiku-3.5": 138,  # claude-3-5-haiku-20241022, score: 1324
    "haiku-3.5-thinking": 138,  # claude-3-5-haiku-20241022, score: 1324
    "haiku-4.5": 56,  # claude-haiku-4-5-20251001, score: 1403
    "haiku-4.5-thinking": 56,  # claude-haiku-4-5-20251001, score: 1403
    "opus-4.1": 15,  # claude-opus-4-1-20250805, score: 1445
    "opus-4.1-thinking": 14,  # claude-opus-4-1-20250805-thinking-16k, score: 1449
    # Google
    "gemini-2.0-flash": 96,  # gemini-2.0-flash-001, score: 1361
    "gemini-2.0-flash-thinking": 96,  # gemini-2.0-flash-001, score: 1361
    "gemini-2.0-flash-lite": 102,  # gemini-2.0-flash-lite-preview-02-05, score: 1353
    "gemini-2.0-flash-lite-thinking": 102,  # gemini-2.0-flash-lite-preview-02-05, score: 1353
    "gemini-2.5-flash": 53,  # gemini-2.5-flash, score: 1409
    "gemini-2.5-flash-thinking": 53,  # gemini-2.5-flash, score: 1409
    "gemini-2.5-pro": 10,  # gemini-2.5-pro, score: 1451
    "gemini-2.5-pro-thinking": 10,  # gemini-2.5-pro, score: 1451
    # XAI
    "grok-3-mini": 98,  # grok-3-mini-beta, score: 1356 (using closest match)
    "grok-3-mini-thinking": 98,  # grok-3-mini-beta, score: 1356 (using closest match)
    "grok-4.1-fast": 25,  # grok-4-1-fast-reasoning, score: 1430
    "grok-4.1-fast-thinking": 25,  # grok-4-1-fast-reasoning, score: 1430
    # Together - Llama
    "ll-3.1-8b": 225,  # llama-3.1-8b-instruct, score: 1212
    "ll-3.1-70b": 173,  # llama-3.1-70b-instruct, score: 1294
    "ll-3.3-70b-dsR1-thinking": None,  # Not found on leaderboard
    "ll-3.1-405b": 125,  # llama-3.1-405b-instruct-bf16, score: 1336 (using bf16 version)
    # Together - Qwen
    "qwen-2.5-7b": None,  # qwen2.5-7b not found (qwen1.5-7b-chat at rank 262)
    "qwen-2.5-72b": 170,  # qwen2.5-72b-instruct, score: 1303
    "qwen-3.0-80b": 59,  # qwen3-next-80b-a3b-instruct, score: 1401
    "qwen-3.0-80b-thinking": 90,  # qwen3-next-80b-a3b-thinking, score: 1369
    "qwen-3.0-235b": 33,  # qwen3-235b-a22b-instruct-2507, score: 1422
    "qwen-3.0-235b-thinking": 62,  # qwen3-235b-a22b-thinking-2507, score: 1398
    # Together - DeepSeek
    "deepseek-3.0": 97,  # deepseek-v3, score: 1358
    "deepseek-3.1": 42,  # deepseek-v3.1, score: 1417
    "deepseek-r1-thinking": 37,  # deepseek-r1-0528, score: 1418
    # Moonshot
    "kimi-k2": 38,  # kimi-k2-0905-preview, score: 1418
    "kimi-k2-thinking": 26,  # kimi-k2-thinking-turbo, score: 1429
    # Fireworks - Llama
    "ll-3.1-8b_fw": 225,  # Same as Together version
    "ll-3.1-70b_fw": 173,  # Same as Together version
    "ll-3.1-405b_fw": 125,  # Same as Together version
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": 78,  # qwen3-30b-a3b-instruct-2507, score: 1383
    "qwen-3.0-235b_fw": 45,  # qwen3-vl-235b-a22b-instruct, score: 1415
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": 42,  # Same as Together version
    "deepseek-r1_fw": 37,  # Same as Together version (deepseek-r1-0528)
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

    WARNING: When multiple short names map to the same inspect model (e.g.,
    "sonnet-4.5" and "sonnet-4.5-thinking"), this function cannot determine
    which one was originally requested. It returns the first in sorted order.

    For reliable model name resolution, use the original model name from the
    request context (e.g., models_to_generate list) rather than this function.
    """
    shorts = SHORT_MODEL_NAMES.get(model)
    if not shorts:
        raise KeyError(f"No short model name found for inspect model '{model}'")

    # Return first in sorted order (deterministic but arbitrary)
    # Note: This may not be the correct variant - caller should use context
    return sorted(shorts)[0]


def is_thinking_model(treatment_name: str) -> bool:
    """
    Check if a treatment name corresponds to a thinking/reasoning model.

    CoT reasoning is ONLY enabled when the model name explicitly has the "-thinking" suffix.
    This applies to all models, including those that are always thinking-capable (like o-series,
    gpt-5) and those with dual modes (like Gemini and Claude).

    Treatment names may have suffixes like "_caps_S2" or "_typos_S4",
    so we extract the base model name first.

    Args:
        treatment_name: Treatment name (may include suffixes like "_caps_S2")

    Returns:
        True if the base model name contains "-thinking", False otherwise
    """
    # Extract base model name by removing treatment suffixes
    # Treatment names can be: "model_name", "model_name_caps_S2", "model_name_typos_S4"
    base_name = treatment_name
    if "_caps_" in base_name:
        base_name = base_name.split("_caps_")[0]
    elif "_typos_" in base_name:
        base_name = base_name.split("_typos_")[0]

    # CoT reasoning is ONLY enabled when the model name has "-thinking" suffix
    # This applies to all models, regardless of whether they're always thinking-capable
    # Examples:
    # - gemini-2.5-pro → False (no CoT)
    # - gemini-2.5-pro-thinking → True (CoT enabled)
    # - o3 → False (no CoT, even though always thinking-capable)
    # - o3-thinking → True (CoT enabled)
    # - sonnet-4.5 → False (no CoT)
    # - sonnet-4.5-thinking → True (CoT enabled)
    return "-thinking" in base_name


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


def is_native_reasoning_model(model_name: str) -> bool:
    """
    Check if a model is a native reasoning model (uses a different endpoint for reasoning).

    Native reasoning models have their own separate data generated with reasoning.
    COT-I models (instruction-tuned models prompted to think) use the same endpoint
    and should pull data from their non-thinking counterparts.

    Args:
        model_name: Model name (may include "-thinking" suffix)

    Returns:
        True if the model is a native reasoning model, False if it's COT-I
    """
    if not model_name.endswith("-thinking"):
        return False

    base_name = get_base_model_name(model_name)

    # Native reasoning models (COT-R) that use different endpoints, are always-reasoning,
    # or are dual-mode models used with -thinking for reasoning. These have their own
    # generated data with the -thinking suffix. COT-I (instruction-tuned, same endpoint)
    # use base model data and are NOT in this list.
    native_reasoning_bases = [
        # Together AI models with separate thinking endpoints
        "qwen-3.0-80b",
        "qwen-3.0-235b",
        "deepseek-r1",
        "ll-3.3-70b-dsR1",
        "gpt-oss-20b",
        "gpt-oss-120b",
        "kimi-k2",
        # OpenAI o-series (always reasoning)
        "o3",
        "o3-mini",
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

    return base_name in native_reasoning_bases


def get_data_model_name(model_name: str) -> str:
    """
    Get the model name to use for data loading.

    For COT-I models (instruction-tuned models prompted to think step-by-step),
    returns the base model name without "-thinking" suffix since these models
    use data generated without reasoning instructions.

    For native reasoning models (different endpoints), returns the full model name
    since they have their own separately generated data.

    Args:
        model_name: Model name (may include "-thinking" suffix)

    Returns:
        Model name to use for data directory lookup
    """
    if not model_name.endswith("-thinking"):
        return model_name

    # Native reasoning models use their own data with -thinking suffix
    if is_native_reasoning_model(model_name):
        return model_name

    # COT-I models use data from non-thinking counterpart
    return get_base_model_name(model_name)
