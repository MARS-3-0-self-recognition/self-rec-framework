from collections import defaultdict
from dataclasses import dataclass

@dataclass
class ModelInfo:
    model_name: str | None
    parameter_count: str | None
    parameter_count_estimated: str | None
    release_date: str | None
    release_date_estimated: str | None
    capability_tier: int | None
    lm_arena_ranking: int | None  # Updated from https://arena.ai/leaderboard/text on 2026-03-27.
    lm_arena_score: int | None  # Updated from https://arena.ai/leaderboard/text on 2026-03-27.
    gpu_tier: str | None

INSPECT_MODELS: dict[str, ModelInfo] = {
    # OpenAI
    "gpt-4o-mini": ModelInfo(
        model_name="openai/gpt-4o-mini",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, similar to Claude 3 Haiku
        release_date="2024-07-18",  # Confirmed
        release_date_estimated=None,
        capability_tier=1,  # Small model (~8B estimated)
        lm_arena_ranking=188,  # gpt-4o-mini-2024-07-18, score: 1317
        lm_arena_score=1317,
        gpu_tier=None,
    ),
    "gpt-4o": ModelInfo(
        model_name="openai/gpt-4o",
        parameter_count="unknown",
        parameter_count_estimated="1.8T",  # Estimated, similar to GPT-4
        release_date="2024-05",  # Confirmed (month only)
        release_date_estimated=None,
        capability_tier=4,  # Large frontier model (~1.8T estimated)
        lm_arena_ranking=150,  # gpt-4o-2024-05-13, score: 1345
        lm_arena_score=1345,
        gpu_tier=None,
    ),
    "gpt-4.1-mini": ModelInfo(
        model_name="openai/gpt-4.1-mini-2025-04-14",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, similar to gpt-4o-mini
        release_date="2025-04-14",  # From model name
        release_date_estimated=None,
        capability_tier=1,  # Small model (~8B estimated)
        lm_arena_ranking=111,  # gpt-4.1-mini-2025-04-14, score: 1382
        lm_arena_score=1382,
        gpu_tier=None,
    ),
    "gpt-4.1": ModelInfo(
        model_name="openai/gpt-4.1-2025-04-14",
        parameter_count="unknown",
        parameter_count_estimated="1.8T",  # Estimated, similar to gpt-4o
        release_date="2025-04-14",  # From model name
        release_date_estimated=None,
        capability_tier=4,  # Large frontier model (~1.8T estimated), newer than 4o
        lm_arena_ranking=69,  # gpt-4.1-2025-04-14, score: 1413
        lm_arena_score=1413,
        gpu_tier=None,
    ),
    "gpt-5-mini": ModelInfo(
        model_name="openai/gpt-5-mini",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="unknown",
        release_date_estimated="2025-08-07",  # Estimated, same as GPT-5
        capability_tier=2,  # Medium model, newer generation
        lm_arena_ranking=94,  # gpt-5-mini-high, score: 1382
        lm_arena_score=1382,
        gpu_tier=None,
    ),
    "gpt-5-mini-thinking": ModelInfo(
        model_name="openai/gpt-5-mini",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="unknown",
        release_date_estimated="2025-08-07",  # Estimated, same as GPT-5
        capability_tier=2,  # Medium model, newer generation
        lm_arena_ranking=94,  # gpt-5-mini-high, score: 1382
        lm_arena_score=1382,
        gpu_tier=None,
    ),
    "gpt-5": ModelInfo(
        model_name="openai/gpt-5",
        parameter_count="unknown",
        parameter_count_estimated="1.8T",  # Estimated, similar to GPT-4
        release_date="2025-08-07",  # Confirmed
        release_date_estimated=None,
        capability_tier=5,  # Latest flagship frontier model (~1.8T estimated)
        lm_arena_ranking=38,  # gpt-5.1, score: 1439
        lm_arena_score=1439,
        gpu_tier=None,
    ),
    "gpt-5-thinking": ModelInfo(
        model_name="openai/gpt-5",
        parameter_count="unknown",
        parameter_count_estimated="1.8T",  # Estimated, similar to GPT-4
        release_date="2025-08-07",  # Confirmed
        release_date_estimated=None,
        capability_tier=5,  # Latest flagship frontier model
        lm_arena_ranking=38,  # gpt-5.1, score: 1439
        lm_arena_score=1439,
        gpu_tier=None,
    ),
    "o3": ModelInfo(
        model_name="openai/o3-2025-04-16",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available (likely > o3-mini)
        release_date="2025-04-16",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Latest reasoning-focused frontier model
        lm_arena_ranking=43,  # o3-2025-04-16, score: 1431
        lm_arena_score=1431,
        gpu_tier=None,
    ),
    "o3-thinking": ModelInfo(
        model_name="openai/o3-2025-04-16",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-04-16",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Latest reasoning-focused frontier model
        lm_arena_ranking=43,  # o3-2025-04-16, score: 1431
        lm_arena_score=1431,
        gpu_tier=None,
    ),
    "o3-mini": ModelInfo(
        model_name="openai/o3-mini-2025-01-31",
        parameter_count="36B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-01-31",  # From model name
        release_date_estimated=None,
        capability_tier=3,  # Medium-large reasoning model (36B)
        lm_arena_ranking=142,  # o3-mini, score: 1348
        lm_arena_score=1348,
        gpu_tier=None,
    ),
    "o3-mini-thinking": ModelInfo(
        model_name="openai/o3-mini-2025-01-31",
        parameter_count="36B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-01-31",  # From model name
        release_date_estimated=None,
        capability_tier=3,  # Medium-large reasoning model
        lm_arena_ranking=142,  # o3-mini, score: 1348
        lm_arena_score=1348,
        gpu_tier=None,
    ),
    # Anthropic
    "sonnet-4.5": ModelInfo(
        model_name="anthropic/claude-sonnet-4-5-20250929",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-09-29",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Latest flagship Claude model (newest version)
        lm_arena_ranking=25,  # claude-sonnet-4-5-20250929, score: 1453
        lm_arena_score=1453,
        gpu_tier=None,
    ),
    "sonnet-4.5-thinking": ModelInfo(
        model_name="anthropic/claude-sonnet-4-5-20250929",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-09-29",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Latest flagship Claude model
        lm_arena_ranking=25,  # claude-sonnet-4-5-20250929, score: 1453
        lm_arena_score=1453,
        gpu_tier=None,
    ),
    "sonnet-3.7": ModelInfo(
        model_name="anthropic/claude-3-7-sonnet-20250219",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-02-19",  # From model name
        release_date_estimated=None,
        capability_tier=4,  # Large frontier model (earlier version)
        lm_arena_ranking=115,  # claude-3-7-sonnet-20250219, score: 1386
        lm_arena_score=1386,
        gpu_tier=None,
    ),
    "sonnet-3.7-thinking": ModelInfo(
        model_name="anthropic/claude-3-7-sonnet-20250219",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-02-19",  # From model name
        release_date_estimated=None,
        capability_tier=4,  # Large frontier model
        lm_arena_ranking=97,  # claude-3-7-sonnet-20250219-thinking-32k, score: 1379
        lm_arena_score=1379,
        gpu_tier=None,
    ),
    "haiku-3.5": ModelInfo(
        model_name="anthropic/claude-3-5-haiku-20241022",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, similar to GPT-4o-mini
        release_date="2024-10-22",  # From model name
        release_date_estimated=None,
        capability_tier=1,  # Small model (~8B estimated)
        lm_arena_ranking=169,  # claude-3-5-haiku-20241022, score: 1336
        lm_arena_score=1336,
        gpu_tier=None,
    ),
    "haiku-3.5-thinking": ModelInfo(
        model_name="anthropic/claude-3-5-haiku-20241022",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, same as haiku-3.5
        release_date="2024-10-22",  # From model name
        release_date_estimated=None,
        capability_tier=1,  # Small model
        lm_arena_ranking=169,  # claude-3-5-haiku-20241022, score: 1336
        lm_arena_score=1336,
        gpu_tier=None,
    ),
    "haiku-4.5": ModelInfo(
        model_name="anthropic/claude-4-5-haiku-20251001",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, similar to haiku-3.5
        release_date="2025-10-01",  # From model name
        release_date_estimated=None,
        capability_tier=1,  # Small model (~8B estimated), newer version
        lm_arena_ranking=76,  # claude-haiku-4-5-20251001, score: 1407
        lm_arena_score=1407,
        gpu_tier=None,
    ),
    "haiku-4.5-thinking": ModelInfo(
        model_name="anthropic/claude-4-5-haiku-20251001",
        parameter_count="unknown",
        parameter_count_estimated="8B",  # Estimated, same as haiku-4.5
        release_date="2025-10-01",  # From model name
        release_date_estimated=None,
        capability_tier=1,  # Small model, newer version
        lm_arena_ranking=76,  # claude-haiku-4-5-20251001, score: 1407
        lm_arena_score=1407,
        gpu_tier=None,
    ),
    "opus-4.1": ModelInfo(
        model_name="anthropic/claude-opus-4-1-20250805",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-08-05",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Flagship Claude model (highest tier)
        lm_arena_ranking=70,  # claude-opus-4-20250514, score: 1412
        lm_arena_score=1412,
        gpu_tier=None,
    ),
    "opus-4.1-thinking": ModelInfo(
        model_name="anthropic/claude-opus-4-1-20250805",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-08-05",  # From model name
        release_date_estimated=None,
        capability_tier=5,  # Flagship Claude model
        lm_arena_ranking=70,  # claude-opus-4-20250514, score: 1412
        lm_arena_score=1412,
        gpu_tier=None,
    ),
    # Google
    "gemini-2.0-flash": ModelInfo(
        model_name="google/gemini-2.0-flash",
        parameter_count="20B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-02-05",  # General availability
        release_date_estimated=None,
        capability_tier=2,  # Medium model (20B)
        lm_arena_ranking=129,  # gemini-2.0-flash-001, score: 1360
        lm_arena_score=1360,
        gpu_tier=None,
    ),
    "gemini-2.0-flash-thinking": ModelInfo(
        model_name="google/gemini-2.0-flash",
        parameter_count="20B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-02-05",  # General availability
        release_date_estimated=None,
        capability_tier=2,  # Medium model
        lm_arena_ranking=129,  # gemini-2.0-flash-001, score: 1360
        lm_arena_score=1360,
        gpu_tier=None,
    ),
    "gemini-2.0-flash-lite": ModelInfo(
        model_name="google/gemini-2.0-flash-lite",
        parameter_count="unknown",
        parameter_count_estimated="10B",  # Estimated, smaller than flash (20B)
        release_date="2025-02-05",  # Preview release
        release_date_estimated=None,
        capability_tier=1,  # Small model (~10B estimated)
        lm_arena_ranking=112,  # gemini-2.5-flash-lite-preview-09-2025-no-thinking, score: 1380
        lm_arena_score=1380,
        gpu_tier=None,
    ),
    "gemini-2.0-flash-lite-thinking": ModelInfo(
        model_name="google/gemini-2.0-flash-lite",
        parameter_count="unknown",
        parameter_count_estimated="10B",  # Estimated, same as flash-lite
        release_date="2025-02-05",  # Preview release
        release_date_estimated=None,
        capability_tier=1,  # Small model
        lm_arena_ranking=112,  # gemini-2.5-flash-lite, score: 1380
        lm_arena_score=1380,
        gpu_tier=None,
    ),
    "gemini-2.5-flash": ModelInfo(
        model_name="google/gemini-2.5-flash",
        parameter_count="5B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-06-17",  # General availability
        release_date_estimated=None,
        capability_tier=1,  # Small model (5B), newer but smaller
        lm_arena_ranking=73,  # gemini-2.5-flash, score: 1411
        lm_arena_score=1411,
        gpu_tier=None,
    ),
    "gemini-2.5-flash-thinking": ModelInfo(
        model_name="google/gemini-2.5-flash",
        parameter_count="5B",  # Confirmed
        parameter_count_estimated=None,
        release_date="2025-06-17",  # General availability
        release_date_estimated=None,
        capability_tier=1,  # Small model
        lm_arena_ranking=73,  # gemini-2.5-flash, score: 1411
        lm_arena_score=1411,
        gpu_tier=None,
    ),
    "gemini-2.5-pro": ModelInfo(
        model_name="google/gemini-2.5-pro",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available (likely > flash)
        release_date="2025-06-17",  # General availability
        release_date_estimated=None,
        capability_tier=4,  # Large flagship model
        lm_arena_ranking=28,  # gemini-2.5-pro, score: 1448
        lm_arena_score=1448,
        gpu_tier=None,
    ),
    "gemini-2.5-pro-thinking": ModelInfo(
        model_name="google/gemini-2.5-pro",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-06-17",  # General availability
        release_date_estimated=None,
        capability_tier=4,  # Large flagship model
        lm_arena_ranking=28,  # gemini-2.5-pro, score: 1448
        lm_arena_score=1448,
        gpu_tier=None,
    ),
    # XAI
    "grok-3-mini": ModelInfo(
        model_name="openai/grok-3-mini",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-02-14",  # Confirmed
        release_date_estimated=None,
        capability_tier=1,  # Small model (mini variant)
        lm_arena_ranking=123,  # grok-3-mini-high, score: 1378
        lm_arena_score=1378,
        gpu_tier=None,
    ),
    "grok-3-mini-thinking": ModelInfo(
        model_name="openai/grok-3-mini",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-02-14",  # Confirmed
        release_date_estimated=None,
        capability_tier=1,  # Small model
        lm_arena_ranking=123,  # grok-3-mini-high, score: 1378
        lm_arena_score=1378,
        gpu_tier=None,
    ),
    "grok-4.1-fast": ModelInfo(
        model_name="openai/grok-4-1-fast-non-reasoning",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2024-11-17",  # Confirmed
        release_date_estimated=None,
        capability_tier=3,  # Medium-large model (fast variant)
        lm_arena_ranking=41,  # grok-4-1-fast-reasoning, score: 1435
        lm_arena_score=1435,
        gpu_tier=None,
    ),
    "grok-4.1-fast-thinking": ModelInfo(
        model_name="openai/grok-4-1-fast-reasoning",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2024-11-17",  # Confirmed
        release_date_estimated=None,
        capability_tier=3,  # Medium-large model
        lm_arena_ranking=41,  # grok-4-1-fast-reasoning, score: 1435
        lm_arena_score=1435,
        gpu_tier=None,
    ),
    # Together - Llama
    "ll-3.3-70b-dsR1-thinking": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        parameter_count="70B",  # From model name (distilled)
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="2025-05",  # Estimated, based on DeepSeek-R1 release
        capability_tier=3,  # Medium-large model (70B, distilled)
        lm_arena_ranking=None,  # Not found on leaderboard
        lm_arena_score=None,  # Not found on leaderboard
        gpu_tier=None,
    ),
    "ll-70B-dsr1-thinking": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=None,
        lm_arena_score=None,
        gpu_tier=None,
    ),
    "ll-3.1-405b": ModelInfo(
        model_name="together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        parameter_count="405B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Confirmed
        release_date_estimated=None,
        capability_tier=5,  # Very large model (405B)
        lm_arena_ranking=155,  # llama-3.1-405b-instruct-bf16, score: 1346
        lm_arena_score=1346,
        gpu_tier="large",
    ),
    "ll-3.1-70b": ModelInfo(
        model_name=None,
        parameter_count="70B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Confirmed
        release_date_estimated=None,
        capability_tier=3,  # Medium-large model (70B)
        lm_arena_ranking=208,  # llama-3.1-70b-instruct, score: 1293
        lm_arena_score=1293,
        gpu_tier="large",
    ),
    # Together - Qwen
    "qwen-2.5-7b": ModelInfo(
        model_name="together/Qwen/Qwen2.5-7B-Instruct-Turbo",
        parameter_count="7B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-09-19",  # Confirmed
        release_date_estimated=None,
        capability_tier=1,  # Small model (7B)
        lm_arena_ranking=None,  # Not on leaderboard
        lm_arena_score=None,  # Not on leaderboard
        gpu_tier=None,
    ),
    "qwen-2.5-72b": ModelInfo(
        model_name="together/Qwen/Qwen2.5-72B-Instruct-Turbo",
        parameter_count="72B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-09-19",  # Confirmed
        release_date_estimated=None,
        capability_tier=3,  # Medium-large model (72B)
        lm_arena_ranking=205,  # qwen2.5-72b-instruct, score: 1302
        lm_arena_score=1302,
        gpu_tier=None,
    ),
    "qwen-3.0-80b": ModelInfo(
        model_name="together/Qwen/Qwen3-Next-80B-A3B-Instruct",
        parameter_count="80B",  # From model name
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="2025-07",  # Estimated, similar to qwen-3.0-235b
        capability_tier=3,  # Medium-large model (80B), newer version
        lm_arena_ranking=85,  # qwen3-next-80b-a3b-instruct, score: 1402
        lm_arena_score=1402,
        gpu_tier=None,
    ),
    "qwen-3.0-80b-thinking": ModelInfo(
        model_name="together/Qwen/Qwen3-Next-80B-A3B-Thinking",
        parameter_count="80B",  # From model name
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="2025-07",  # Estimated, same as qwen-3.0-80b
        capability_tier=3,  # Medium-large model
        lm_arena_ranking=85,  # qwen3-next-80b-a3b-instruct, score: 1402
        lm_arena_score=1402,
        gpu_tier=None,
    ),
    "qwen-3.0-235b": ModelInfo(
        model_name="together/Qwen/Qwen3-235B-A22B-Instruct-2507",
        parameter_count="235B",  # From model name
        parameter_count_estimated=None,
        release_date="2025-07",  # Estimated from model name (2507 = July 2025)
        release_date_estimated=None,
        capability_tier=4,  # Large model (235B)
        lm_arena_ranking=54,  # qwen3-235b-a22b-instruct-2507, score: 1422
        lm_arena_score=1422,
        gpu_tier=None,
    ),
    "qwen-3.0-235b-thinking": ModelInfo(
        model_name="together/Qwen/Qwen3-235B-A22B-Thinking-2507",
        parameter_count="235B",  # From model name
        parameter_count_estimated=None,
        release_date="2025-07",  # Estimated from model name
        release_date_estimated=None,
        capability_tier=4,  # Large model
        lm_arena_ranking=89,  # qwen3-235b-a22b-thinking-2507, score: 1400
        lm_arena_score=1400,
        gpu_tier=None,
    ),
    # Together - DeepSeek
    "deepseek-3.0": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-V3",
        parameter_count="671B",  # Total parameters (MoE)
        parameter_count_estimated=None,
        release_date="2024-12",  # Confirmed (month only)
        release_date_estimated=None,
        capability_tier=5,  # Very large MoE model (671B total)
        lm_arena_ranking=95,  # deepseek-v3-0324, score: 1395
        lm_arena_score=1395,
        gpu_tier=None,
    ),
    "deepseek-3.1": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-V3.1",
        parameter_count="671B",  # Total parameters (MoE)
        parameter_count_estimated=None,
        release_date="2025-08-19",  # Confirmed
        release_date_estimated=None,
        capability_tier=5,  # Very large MoE model (671B total), newer version
        lm_arena_ranking=59,  # deepseek-v3.1, score: 1418
        lm_arena_score=1418,
        gpu_tier=None,
    ),
    "deepseek-3.1-thinking": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-V3.1",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=59,  # deepseek-v3.1, score: 1418
        lm_arena_score=1418,
        gpu_tier=None,
    ),
    "deepseek-r1-thinking": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-R1",  # reasoning model
        parameter_count="671B",  # Total parameters (MoE, same architecture as V3)
        parameter_count_estimated=None,
        release_date="2025-05",  # Confirmed (month only, version R1-0528)
        release_date_estimated=None,
        capability_tier=5,  # Very large reasoning model (671B total)
        lm_arena_ranking=56,  # deepseek-r1-0528, score: 1422
        lm_arena_score=1422,
        gpu_tier=None,
    ),
    "deepseek-r1-0528-thinking": ModelInfo(
        model_name="together/deepseek-ai/DeepSeek-R1-0528",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=56,  # deepseek-r1-0528, score: 1422
        lm_arena_score=1422,
        gpu_tier=None,
    ),
    # Moonshot
    "kimi-k2": ModelInfo(
        model_name="together/moonshotai/Kimi-K2-Instruct-0905",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-07-12",  # Confirmed
        release_date_estimated=None,
        capability_tier=4,  # Large flagship model
        lm_arena_ranking=57,  # kimi-k2-0905-preview, score: 1419
        lm_arena_score=1419,
        gpu_tier=None,
    ),
    "kimi-k2-thinking": ModelInfo(
        model_name="together/moonshotai/Kimi-K2-Thinking",
        parameter_count="unknown",
        parameter_count_estimated="unknown",  # No reliable estimate available
        release_date="2025-11-06",  # Confirmed
        release_date_estimated=None,
        capability_tier=4,  # Large flagship reasoning model
        lm_arena_ranking=42,  # kimi-k2-thinking-turbo, score: 1434
        lm_arena_score=1434,
        gpu_tier=None,
    ),
    "kimi-k2.5": ModelInfo(
        model_name="together/moonshotai/Kimi-K2.5",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=42,  # kimi-k2.5-instant, score: 1433
        lm_arena_score=1433,
        gpu_tier=None,
    ),
    "kimi-k2.5-thinking": ModelInfo(
        model_name="together/moonshotai/Kimi-K2.5",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=23,  # kimi-k2.5-thinking, score: 1454
        lm_arena_score=1454,
        gpu_tier=None,
    ),
    # MiniMax / GLM
    "minimax-m2.5-thinking": ModelInfo(
        model_name="together/MiniMaxAI/MiniMax-M2.5",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=78,  # minimax-m2.5, score: 1406
        lm_arena_score=1406,
        gpu_tier=None,
    ),
    "glm-4.5-air-thinking": ModelInfo(
        model_name="together/zai-org/GLM-4.5-Air-FP8",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=118,  # glm-4.5-air, score: 1373
        lm_arena_score=1373,
        gpu_tier=None,
    ),
    "glm-4.7-thinking": ModelInfo(
        model_name="together/zai-org/GLM-4.7",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=35,  # glm-4.7, score: 1443
        lm_arena_score=1443,
        gpu_tier=None,
    ),
    # HF local
    "ll-3.1-8b": ModelInfo(
        model_name="hf/meta-llama/Llama-3.1-8B-Instruct",
        parameter_count="8B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Confirmed
        release_date_estimated=None,
        capability_tier=1,  # Small model (8B)
        lm_arena_ranking=260,  # llama-3.1-8b-instruct, score: 1211
        lm_arena_score=1211,
        gpu_tier="small",
    ),
    "ll-3.3-70b": ModelInfo(
        model_name="hf/meta-llama/Llama-3.3-70B-Instruct",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=183,  # llama-3.3-70b-instruct, score: 1318
        lm_arena_score=1318,
        gpu_tier=None,
    ),
    "qwen-3.0-30b": ModelInfo(
        model_name="hf/Qwen/Qwen3-30B-A3B-Instruct-2507",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=109,  # qwen3-30b-a3b-instruct-2507, score: 1383
        lm_arena_score=1383,
        gpu_tier=None,
    ),
    "qwen-3.0-30b-thinking": ModelInfo(
        model_name="hf/Qwen/Qwen3-30B-A3B-Instruct-2507",  # Same model, native reasoning
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=None,
        lm_arena_score=None,
        gpu_tier=None,
    ),
    "qwen-3.5-27b": ModelInfo(
        model_name="hf/Qwen/Qwen3.5-27B",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=79,  # qwen3.5-27b, score: 1405
        lm_arena_score=1405,
        gpu_tier="medium",
    ),
    "qwen-3.5-27b-thinking": ModelInfo(
        model_name="hf/Qwen/Qwen3.5-27B",  # Same model, native reasoning
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=None,
        lm_arena_score=None,
        gpu_tier=None,
    ),
    "gpt-oss-20b": ModelInfo(
        model_name="hf/openai/gpt-oss-20b",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=184,  # gpt-oss-20b, score: 1318
        lm_arena_score=1318,
        gpu_tier=None,
    ),
    "gpt-oss-20b-thinking": ModelInfo(
        model_name="hf/openai/gpt-oss-20b",  # Same model, native reasoning
        parameter_count="20B",  # From model name
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="unknown",  # No reliable estimate available
        capability_tier=2,  # Medium model (20B)
        lm_arena_ranking=184,  # Same model in thinking mode
        lm_arena_score=1318,
        gpu_tier=None,
    ),
    "gpt-oss-120b": ModelInfo(
        model_name="hf/openai/gpt-oss-120b",
        parameter_count=None,
        parameter_count_estimated=None,
        release_date=None,
        release_date_estimated=None,
        capability_tier=None,
        lm_arena_ranking=134,  # gpt-oss-120b, score: 1354
        lm_arena_score=1354,
        gpu_tier=None,
    ),
    "gpt-oss-120b-thinking": ModelInfo(
        model_name="hf/openai/gpt-oss-120b",  # Same model, native reasoning
        parameter_count="120B",  # From model name
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="unknown",  # No reliable estimate available
        capability_tier=3,  # Large model (120B)
        lm_arena_ranking=134,  # Same model in thinking mode
        lm_arena_score=1354,
        gpu_tier=None,
    ),
    # Fireworks
    "ll-3.1-8b_fw": ModelInfo(
        model_name=None,
        parameter_count="8B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Same as Together version
        release_date_estimated=None,
        capability_tier=1,  # Small model (8B)
        lm_arena_ranking=260,  # Same as Together version
        lm_arena_score=1211,
        gpu_tier=None,
    ),
    "ll-3.1-70b_fw": ModelInfo(
        model_name=None,
        parameter_count="70B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Same as Together version
        release_date_estimated=None,
        capability_tier=3,  # Medium-large model (70B)
        lm_arena_ranking=208,  # Same as Together version
        lm_arena_score=1293,
        gpu_tier=None,
    ),
    "ll-3.1-405b_fw": ModelInfo(
        model_name=None,
        parameter_count="405B",  # From model name
        parameter_count_estimated=None,
        release_date="2024-07-23",  # Same as Together version
        release_date_estimated=None,
        capability_tier=5,  # Very large model (405B)
        lm_arena_ranking=155,  # Same as Together version
        lm_arena_score=1346,
        gpu_tier=None,
    ),
    "qwen-3.0-30b_fw": ModelInfo(
        model_name=None,
        parameter_count="30B",  # From model name
        parameter_count_estimated=None,
        release_date="unknown",
        release_date_estimated="2025-07",  # Estimated, similar to other Qwen 3.0 models
        capability_tier=2,  # Medium model (30B)
        lm_arena_ranking=109,  # Same as Together version
        lm_arena_score=1383,
        gpu_tier=None,
    ),
    "qwen-3.0-235b_fw": ModelInfo(
        model_name=None,
        parameter_count="235B",  # From model name
        parameter_count_estimated=None,
        release_date="2025-07",  # Estimated, same as Together version
        release_date_estimated=None,
        capability_tier=4,  # Large model (235B)
        lm_arena_ranking=54,  # Same as Together version
        lm_arena_score=1422,
        gpu_tier=None,
    ),
    "deepseek-3.1_fw": ModelInfo(
        model_name=None,
        parameter_count="671B",  # Total parameters (MoE)
        parameter_count_estimated=None,
        release_date="2025-08-19",  # Same as Together version
        release_date_estimated=None,
        capability_tier=5,  # Very large MoE model (671B total)
        lm_arena_ranking=59,  # Same as Together version
        lm_arena_score=1418,
        gpu_tier=None,
    ),
    "deepseek-r1_fw": ModelInfo(
        model_name=None,
        parameter_count="671B",  # Total parameters (MoE)
        parameter_count_estimated=None,
        release_date="2025-05-28",  # From model name (0528 = May 28)
        release_date_estimated=None,
        capability_tier=5,  # Very large reasoning model (671B total)
        lm_arena_ranking=56,  # Same as Together version (deepseek-r1-0528)
        lm_arena_score=1422,
        gpu_tier=None,
    ),
}

INSPECT_MODEL_NAMES: dict[str, str] = {
    k: v.model_name
    for k, v in INSPECT_MODELS.items()
    if v.model_name is not None
}

# Model parameter counts (in billions, unless specified with 'T' for trillions)
# Values are based on model names or official documentation only.
# For MoE models, values represent total parameters (not active per token).
# Use MODEL_PARAMETER_COUNTS_ESTIMATED for estimated values.
MODEL_PARAMETER_COUNTS: dict[str, str] = {
    k: v.parameter_count
    for k, v in INSPECT_MODELS.items()
    if v.parameter_count is not None
}

# Estimated parameter counts for models where official values are not available.
# These are research estimates based on model performance, architecture comparisons,
# and industry analysis. Values should be treated as approximate.
MODEL_PARAMETER_COUNTS_ESTIMATED: dict[str, str] = {
    k: v.parameter_count_estimated
    for k, v in INSPECT_MODELS.items()
    if v.parameter_count_estimated is not None
}

# Model release dates (YYYY-MM-DD format)
# Values are based on official announcements, model names with embedded dates, or confirmed release dates.
# Use MODEL_RELEASE_DATES_ESTIMATED for estimated values.
MODEL_RELEASE_DATES: dict[str, str] = {
    k: v.release_date
    for k, v in INSPECT_MODELS.items()
    if v.release_date is not None
}

# Estimated release dates for models where official dates are not available.
# These are estimates based on model naming patterns, release timelines, and industry analysis.
MODEL_RELEASE_DATES_ESTIMATED: dict[str, str] = {
    k: v.release_date_estimated
    for k, v in INSPECT_MODELS.items()
    if v.release_date_estimated is not None
}

# Model capability tiers (1 = lowest, 5 = highest)
# Tiers are based on relative capabilities considering model size, version progression,
# release date, and known performance benchmarks within the model set.
MODEL_CAPABILITY_TIERS: dict[str, int] = {
    k: v.capability_tier
    for k, v in INSPECT_MODELS.items()
    if v.capability_tier is not None
}

# LM Arena rankings from https://arena.ai/leaderboard (text)
# Rankings are based on Elo scores from the leaderboard (as of Mar 17, 2026).
# Lower rank number = higher position on leaderboard (rank 1 is best).
# Models not found on leaderboard are marked as None.
LM_ARENA_RANKINGS: dict[str, int | None] = {
    k: v.lm_arena_ranking
    for k, v in INSPECT_MODELS.items()
    if v.lm_arena_ranking is not None
}

# LM Arena Elo scores from https://arena.ai/leaderboard (text)
# Scores as of Mar 17, 2026. Higher score = better model.
# Models not found on leaderboard are marked as None.
LM_ARENA_SCORES: dict[str, int | None] = {
    k: v.lm_arena_score
    for k, v in INSPECT_MODELS.items()
    if v.lm_arena_score is not None
}

# GPU tier for hf/ models that need local GPU inference.
# Tier determines which RunPod GPU config to use:
#   "small"  — 8B models, ~16GB VRAM (RTX 3090, RTX 4090, A40, etc.)
#   "medium" — 27B-35B models, ~60GB VRAM (A100 80GB, L40S)
#   "large"  — 70B+ models, ~80GB+ VRAM (A100 80GB, H100)
MODEL_GPU_TIER: dict[str, str] = {
    k: v.gpu_tier
    for k, v in INSPECT_MODELS.items()
    if v.gpu_tier is not None
}


def temp_suffix(temperature) -> str:
    """Return temperature suffix for model/directory names.

    Returns empty string for the default temperature (1.0) or None.
    Returns '_temp_{value}' for non-default temperatures.
    """
    if temperature is None or temperature == 1.0 or temperature == 1:
        return ""
    temp_str = f"{float(temperature):.1f}".rstrip("0").rstrip(".")
    if "." not in temp_str:
        temp_str += ".0"
    return f"_temp_{temp_str}"


def get_gpu_tier(model_name: str) -> str | None:
    """Get GPU tier for a model, or None if it's not an hf/ model."""
    inspect_name = INSPECT_MODEL_NAMES.get(model_name, "")
    if not inspect_name.startswith("hf/"):
        return None
    return MODEL_GPU_TIER.get(model_name, "medium")  # default to medium if unlisted


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
    Look up the inspect model name for a given short model name.

    For trained models (containing '_sft-as_'), resolves via the base model.
    For -thinking variants not in the dict, falls back to the base name.
    """
    if short_model_name in INSPECT_MODEL_NAMES:
        return INSPECT_MODEL_NAMES[short_model_name]

    # Strip -thinking and try again
    clean = short_model_name
    if clean.endswith("-thinking"):
        base = clean.removesuffix("-thinking")
        if base in INSPECT_MODEL_NAMES:
            return INSPECT_MODEL_NAMES[base]

    # Trained models: resolve via base model
    if "_sft-as_" in clean:
        base_part = clean.split("_sft-as_")[0]
        try:
            from scripts.alpaca_eval.training_runs import REORG_MODEL_MAP
            base_short = REORG_MODEL_MAP.get(base_part, base_part)
        except ImportError:
            base_short = base_part
        if base_short in INSPECT_MODEL_NAMES:
            return INSPECT_MODEL_NAMES[base_short]

    raise KeyError(f"No inspect model name found for '{short_model_name}'")


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


def needs_together_reasoning_activation(model_name: str) -> bool:
    """
    Check if a Together AI model needs explicit reasoning activation via extra_body.

    Some Together AI models are "hybrid" — they use the same endpoint for both
    instruct and thinking modes. To enable thinking, the API request must include
    `reasoning: {"enabled": true}` in the request body.

    This is distinct from models that have separate thinking endpoints (e.g.,
    Qwen3-Next-80B-A3B-Thinking) which always produce reasoning output.

    Args:
        model_name: Model name (must include "-thinking" suffix)

    Returns:
        True if the model needs `reasoning: {"enabled": true}` in extra_body
    """
    if not model_name.endswith("-thinking"):
        return False

    base_name = get_base_model_name(model_name)

    # Together AI hybrid models that share the same endpoint for instruct/thinking
    # and need `reasoning: {"enabled": true}` to activate thinking mode
    together_hybrid_thinking_models = [
        "deepseek-3.1",  # together/deepseek-ai/DeepSeek-V3.1
        "kimi-k2.5",  # together/moonshotai/Kimi-K2.5
    ]

    return base_name in together_hybrid_thinking_models


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

    return base_name in native_reasoning_bases


def get_data_model_name(model_name: str) -> str:
    """
    Get the model name to use for data loading.

    For trained models (containing '_sft-as_'), resolves to the base model
    since trained models evaluate on the base model's generated data.

    For COT-I models (instruction-tuned models prompted to think step-by-step),
    returns the base model name without "-thinking" suffix since these models
    use data generated without reasoning instructions.

    For native reasoning models (different endpoints), returns the full model name
    since they have their own separately generated data.

    Args:
        model_name: Model name (may include "-thinking" suffix or trained name)

    Returns:
        Model name to use for data directory lookup
    """
    # Strip -thinking suffix first for trained model resolution
    clean_name = model_name.removesuffix("-thinking") if model_name.endswith("-thinking") else model_name

    # Trained models: resolve to base model for data lookup
    # Names like "gpt-oss-20b_sft-as_gpt-oss-20b_vs_..." → "gpt-oss-20b"
    if "_sft-as_" in clean_name:
        # Extract the base model (part before _sft-as_)
        base_part = clean_name.split("_sft-as_")[0]
        # Map the directory-style name back to shorthand
        try:
            from scripts.alpaca_eval.training_runs import REORG_MODEL_MAP
            base_short = REORG_MODEL_MAP.get(base_part, base_part)
        except ImportError:
            base_short = base_part
        # If the original name had -thinking and the base model's thinking
        # variant is a native reasoning model, keep -thinking for data lookup
        if model_name.endswith("-thinking"):
            thinking_name = base_short + "-thinking"
            if is_native_reasoning_model(thinking_name):
                return thinking_name
        return base_short

    if not model_name.endswith("-thinking"):
        return model_name

    # Native reasoning models use their own data with -thinking suffix
    if is_native_reasoning_model(model_name):
        return model_name

    # COT-I models use data from non-thinking counterpart
    return get_base_model_name(model_name)
