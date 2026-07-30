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
    #"gpt-oss-20b-thinking": "together/openai/gpt-oss-20b",
    #"gpt-oss-120b-thinking": "together/openai/gpt-oss-120b",
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
    "ll-3.3-70b-dsR1-thinking": "together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "ll-70B-dsr1-thinking": "together/deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
    "ll-3.1-405b": "together/meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
    "ll-3-8b-lite": "together/meta-llama/Meta-Llama-3-8B-Instruct-Lite",  # cheap small instruct (Llama 3, 8k ctx)
    "gemma-3n-e4b": "together/google/gemma-3n-E4B-it",  # cheap small instruct, 32k ctx, serverless (tutorial set)
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
    "deepseek-3.1-thinking": "together/deepseek-ai/DeepSeek-V3.1",
    "deepseek-r1-thinking": "together/deepseek-ai/DeepSeek-R1",  # reasoning model
    "deepseek-r1-0528-thinking": "together/deepseek-ai/DeepSeek-R1-0528",
    # Moonshot
    "kimi-k2": "together/moonshotai/Kimi-K2-Instruct-0905",
    "kimi-k2-thinking": "together/moonshotai/Kimi-K2-Thinking",
    "kimi-k2.5": "together/moonshotai/Kimi-K2.5",
    "kimi-k2.5-thinking": "together/moonshotai/Kimi-K2.5",
    #MiniMax
    "minimax-m2.5-thinking": "together/MiniMaxAI/MiniMax-M2.5",
    #GLM
    "glm-4.5-air-thinking": "together/zai-org/GLM-4.5-Air-FP8",
    "glm-4.7-thinking": "together/zai-org/GLM-4.7",
    ## Local HF models (require GPU — dispatched to RunPod when run locally)
    "ll-3.1-8b": "hf/meta-llama/Llama-3.1-8B-Instruct",
    "ll-3.3-70b": "hf/meta-llama/Llama-3.3-70B-Instruct",
    #"qwen-2.5-7b": "hf/Qwen/Qwen2.5-7B-Instruct-Turbo",
    "qwen-3.0-30b": "hf/Qwen/Qwen3-30B-A3B-Instruct-2507",
    "qwen-3.0-30b-thinking": "hf/Qwen/Qwen3-30B-A3B-Instruct-2507",  # Same model, native reasoning
    "qwen-3.5-27b": "hf/Qwen/Qwen3.5-27B",
    "qwen-3.5-27b-thinking": "hf/Qwen/Qwen3.5-27B",  # Same model, native reasoning
    "gpt-oss-20b": "hf/openai/gpt-oss-20b",
    "gpt-oss-20b-thinking": "hf/openai/gpt-oss-20b",  # Same model, native reasoning
    "gpt-oss-120b": "hf/openai/gpt-oss-120b",
    "gpt-oss-120b-thinking": "hf/openai/gpt-oss-120b",  # Same model, native reasoning
    # Together-served gpt-oss (provider-tagged variants). Bare names above stay
    # hf/ (local/tinker for training); eval sets use these tg: routes.
    "tg:gpt-oss-20b": "together/openai/gpt-oss-20b",
    "tg:gpt-oss-20b-thinking": "together/openai/gpt-oss-20b",
    "tg:gpt-oss-120b": "together/openai/gpt-oss-120b",
    "tg:gpt-oss-120b-thinking": "together/openai/gpt-oss-120b",
    ## SGTR-trained Tinker LoRA samplers (served via Tinker OAI proxy when gpu_dispatch=tinker)
    ## Keys match data/training_reorganized/05_MSJ/ dir names so get_data_model_name()
    ## can resolve them to their base model for response-data lookup.
    # Standard SFT (sft-as self, vs alt)
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT_PW_ShareGPT":   "openai/tinker://a38689ef-03ff-531e-8c5f-59572b8d1122:train:0/sampler_weights/final",
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT_PW_PKU":        "openai/tinker://b0240aa7-1095-5571-97e1-963418a510aa:train:0/sampler_weights/final",
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT_IND_ShareGPT":  "openai/tinker://0ed53fab-55a6-5b90-8f2d-4f8985fdf73b:train:0/sampler_weights/final",
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_PW_ShareGPT":   "openai/tinker://6c9972db-2f82-5462-a08c-66513b25388c:train:0/sampler_weights/final",
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_AT_IND_ShareGPT":  "openai/tinker://0dfa5052-ffe0-52ab-b0ec-93540e8a1e1d:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_ShareGPT":     "openai/tinker://607be73d-fa6b-5347-8846-3338083886f2:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_PW_PKU":          "openai/tinker://81e7f29d-2a51-5f7b-8065-131a87f79c6b:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT_IND_ShareGPT":    "openai/tinker://1afda46a-7a09-56db-92f2-733031e9e0f1:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_PW_ShareGPT":     "openai/tinker://9fe5dba4-831f-5e3a-a27f-5e5238a6d8b9:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_AT_IND_ShareGPT":    "openai/tinker://c9bf3aee-32dd-5d08-935f-a94495ec7be1:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_ShareGPT":     "openai/tinker://8d2643e3-f1ad-5250-bcbd-3c18573183e9:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_PW_PKU":          "openai/tinker://a33cb57f-d20e-5d71-a0b9-b35ebb825a86:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT_IND_ShareGPT":    "openai/tinker://03be12f4-9b92-5939-88f1-601c5a7fc1ee:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_PW_ShareGPT":     "openai/tinker://74bc20ce-b275-5053-a116-e782cbce1a1a:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_AT_IND_ShareGPT":    "openai/tinker://20d9cf5c-5043-572d-9b60-389dcac93820:train:0/sampler_weights/final",
    # Multi-OP SFT (jointly trained on UT_PW + UT_IND + AT_PW + AT_IND, per_id_one_source)
    "llama-3-1-8b_sft-as_llama-3-1-8b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT": "openai/tinker://95c31d67-0670-57b4-bc29-d383afbb02c5:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_gpt-oss-20b_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT":   "openai/tinker://e6839bf3-0e18-5c35-b13c-e40d1d8fb82c:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_qwen-3-30b_vs_gpt-oss-120b_UT-AT_PW-IND_ShareGPT":   "openai/tinker://2d96d078-0b73-5968-bab4-15cafbcaccbb:train:0/sampler_weights/final",
    # Random-labels controls (catastrophic-forgetting check) — same multi-OP data,
    # binary targets shuffled per-ID via randomize_train_labels (seed 999).
    "llama-3-1-8b_RANDLABELS_999_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT": "openai/tinker://50c797b9-9d8d-5a90-8a56-a8340ec1a813:train:0/sampler_weights/final",
    "gpt-oss-20b_RANDLABELS_999_vs_qwen-3-30b_UT-AT_PW-IND_ShareGPT":  "openai/tinker://32003efb-2c6b-51c7-9ea5-5287994bd667:train:0/sampler_weights/final",
    # Adversarial SFT (sft-as opposite model, vs its own base)
    "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_PW_ShareGPT":     "openai/tinker://6ff54ead-4366-5cff-8873-e0f693d8cc89:train:0/sampler_weights/final",
    "gpt-oss-20b_sft-as_qwen-3-30b_vs_gpt-oss-20b_UT_IND_ShareGPT":    "openai/tinker://e6e24676-6d19-5938-ad9e-c2bb609df9a4:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_PW_ShareGPT":     "openai/tinker://c2e20b55-3946-5a3f-a768-add3ef35155d:train:0/sampler_weights/final",
    "qwen-3-30b_sft-as_gpt-oss-120b_vs_qwen-3-30b_UT_IND_ShareGPT":    "openai/tinker://2acb8fa7-3ca4-57be-b4d3-e9760cbc1909:train:0/sampler_weights/final",
}

# Maximum OUTPUT tokens per model (the API max_tokens ceiling — total tokens the
# model may emit in a single response, including reasoning for thinking models).
# Used to resolve the `"max"` sentinel in experiment configs
# (max_final_answer_tokens / max_thinking_tokens) to a concrete budget — see
# get_model_output_token_cap(). Keyed by BASE model name (without the "-thinking"
# suffix); the lookup strips "-thinking" before matching, so one entry covers
# both the instruct and thinking variant of the same underlying model.
#
# IMPORTANT: values must NOT exceed the provider's true API ceiling, or requests
# error; when uncertain, prefer a conservative value. Entries marked "# verify"
# are best-effort estimates that should be checked against current provider docs.
#
# Values below were verified against provider docs / model cards (Jun 2026).
# For closed providers these are hard API output ceilings. For open-weight models
# served on Together/HF, output is context-bound (no separate hard cap), so the
# value is the model author's recommended generation budget — set high enough for
# thinking variants (which share one budget across reasoning + answer) not to
# truncate. Entries marked "# new/unverified" are recent models whose limits come
# from aggregators rather than a primary source; confirm against your deployment.
MODEL_OUTPUT_TOKEN_CAP: dict[str, int] = {
    # OpenAI (hard output ceilings)
    "gpt-4o-mini": 16384,
    "gpt-4o": 16384,
    "gpt-4.1-mini": 32768,
    "gpt-4.1": 32768,
    "gpt-5-mini": 128000,
    "gpt-5": 128000,
    "o3": 100000,
    "o3-mini": 100000,
    # Anthropic (hard output ceilings)
    "sonnet-4.5": 64000,
    "sonnet-3.7": 8192,  # 128000 only with output-128k-2025-02-19 beta header
    "haiku-3.5": 8192,
    "haiku-4.5": 64000,
    "opus-4.1": 32000,
    # Google (hard output ceilings)
    "gemini-2.0-flash": 8192,
    "gemini-2.0-flash-lite": 8192,
    "gemini-2.5-flash": 65536,
    "gemini-2.5-pro": 65536,
    # XAI / Grok — context-bound (no fixed API output cap); reasoning headroom
    "grok-3-mini": 16384,
    "grok-4.1-fast": 32768,
    # Together — Llama / DeepSeek-R1 distills (context-bound)
    "ll-3.3-70b-dsR1": 32768,  # R1 max generation = 32768
    "ll-70B-dsr1": 32768,
    "ll-3.1-405b": 8192,
    "ll-3-8b-lite": 8192,
    # Together — Google Gemma
    "gemma-3n-e4b": 8192,
    # Together — Qwen (context-bound; thinking-budget values per Qwen cards)
    "qwen-2.5-7b": 8192,
    "qwen-2.5-72b": 8192,
    "qwen-3.0-80b": 32768,
    "qwen-3.0-235b": 32768,
    # Together — DeepSeek (context-bound)
    "deepseek-3.0": 8192,
    "deepseek-3.1": 32768,  # hybrid; thinking variant needs CoT headroom
    "deepseek-r1": 32768,  # R1 max generation = 32768
    "deepseek-r1-0528": 65536,  # R1-0528 raised max generation to 64k
    # Together — Moonshot (context-bound; covers instruct + thinking variants)
    "kimi-k2": 32768,
    "kimi-k2.5": 32768,  # new/unverified
    # Together — MiniMax
    "minimax-m2.5": 32768,  # new/unverified (docs cite up to 128k output)
    # Together — GLM (context-bound)
    "glm-4.5-air": 32768,
    "glm-4.7": 32768,  # new/unverified
    # Local HF (served on RunPod) — generation cap is configurable
    "ll-3.1-8b": 8192,
    "ll-3.3-70b": 8192,
    "qwen-3.0-30b": 32768,
    "qwen-3.5-27b": 32768,  # new/unverified
    "gpt-oss-20b": 32768,
    "gpt-oss-120b": 32768,
}

# Total CONTEXT WINDOW per model (max input + output tokens combined). A fixed
# model spec — distinct from MODEL_OUTPUT_TOKEN_CAP (the max GENERATION tokens).
# Used to resolve a "max" output budget without overflowing context: the provider
# enforces (input + max_tokens) <= context_window, so the usable output budget is
#   min(output_cap, context_window - estimated_input - margin)
# (see get_model_context_window() and tasks.py:max_output_ceiling). Keyed by BASE
# model name (the "-thinking" suffix is stripped before lookup).
#
# Verified against provider docs / model cards (Jun 2026). For Together-served
# models this is the context length AS SERVED, which can be SMALLER than the base
# model's theoretical max (e.g. Qwen2.5 is served at 32768, not its YaRN-extended
# 131072). Where a provider published only a rounded "NNK" label, the exact
# power-of-two / config integer is used. Anthropic models list their 200k default
# (1M is available only via a beta header).
MODEL_CONTEXT_WINDOW: dict[str, int] = {
    # OpenAI
    "gpt-4o-mini": 128000,
    "gpt-4o": 128000,
    "gpt-4.1-mini": 1047576,
    "gpt-4.1": 1047576,
    "gpt-5-mini": 400000,
    "gpt-5": 400000,
    "o3": 200000,
    "o3-mini": 200000,
    # Anthropic (200k default; 1M only via beta header)
    "sonnet-4.5": 200000,
    "sonnet-3.7": 200000,
    "haiku-3.5": 200000,
    "haiku-4.5": 200000,
    "opus-4.1": 200000,
    # Google
    "gemini-2.0-flash": 1048576,
    "gemini-2.0-flash-lite": 1048576,
    "gemini-2.5-flash": 1048576,
    "gemini-2.5-pro": 1048576,
    # XAI / Grok
    "grok-3-mini": 131072,
    "grok-4.1-fast": 2000000,
    # Together — Llama / DeepSeek-R1 distills
    "ll-3.3-70b-dsR1": 131072,
    "ll-70B-dsr1": 131072,
    "ll-3.1-405b": 131072,
    "ll-3-8b-lite": 8192,  # Llama 3 (NOT 3.1) — 8k base context
    # Together — Google Gemma
    "gemma-3n-e4b": 32768,  # Gemma 3n E4B — 32k ctx (tutorial set)
    # Together — Qwen (Qwen2.5 served BELOW its YaRN-extended max)
    "qwen-2.5-7b": 32768,
    "qwen-2.5-72b": 32768,
    "qwen-3.0-80b": 262144,
    "qwen-3.0-235b": 262144,
    # Together — DeepSeek (served 131072; r1-0528 served at full 163840)
    "deepseek-3.0": 131072,
    "deepseek-3.1": 131072,
    "deepseek-r1": 131072,
    "deepseek-r1-0528": 163840,
    # Together — Moonshot
    "kimi-k2": 262144,
    "kimi-k2.5": 262144,
    # Together — MiniMax (Together serves 192k)
    "minimax-m2.5": 196608,
    # Together — GLM
    "glm-4.5-air": 131072,
    "glm-4.7": 202752,
    # Local HF (served on RunPod)
    "ll-3.1-8b": 131072,
    "ll-3.3-70b": 131072,
    "qwen-3.0-30b": 262144,
    "qwen-3.5-27b": 262144,
    "gpt-oss-20b": 131072,
    "gpt-oss-120b": 131072,
}

# Sparse override for models that enforce a SEPARATE thinking-token budget cap
# below their total output ceiling. For every other model the thinking budget is
# bounded only by the shared output budget (MODEL_OUTPUT_TOKEN_CAP), so they need no
# entry here — get_model_max_thinking_tokens() falls back to MODEL_OUTPUT_TOKEN_CAP.
# Today only Gemini 2.5 needs this: its `thinkingBudget` maxes out well below
# `maxOutputTokens` (verified against ai.google.dev/gemini-api/docs/thinking).
# Keyed by BASE model name (the "-thinking" suffix is stripped before lookup).
MODEL_MAX_THINKING_TOKENS: dict[str, int] = {
    "gemini-2.5-flash": 24576,  # thinkingBudget range 0–24576
    "gemini-2.5-pro": 32768,  # thinkingBudget range 128–32768
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
# LM Arena rankings from https://arena.ai/leaderboard (text)
# Rankings are based on Elo scores from the leaderboard (as of Mar 17, 2026).
# Lower rank number = higher position on leaderboard (rank 1 is best).
# Models not found on leaderboard are marked as None.
LM_ARENA_RANKINGS: dict[str, int | None] = {
    # Updated from https://arena.ai/leaderboard/text on 2026-03-27.
    # OpenAI
    "gpt-4o-mini": 188,  # gpt-4o-mini-2024-07-18, score: 1317
    "gpt-4o": 150,  # gpt-4o-2024-05-13, score: 1345
    "gpt-4.1-mini": 111,  # gpt-4.1-mini-2025-04-14, score: 1382
    "gpt-4.1": 69,  # gpt-4.1-2025-04-14, score: 1413
    "gpt-5-mini": 94,  # gpt-5-mini-high, score: 1382
    "gpt-5-mini-thinking": 94,  # gpt-5-mini-high, score: 1382
    "gpt-5": 38,  # gpt-5.1, score: 1439
    "gpt-5-thinking": 38,  # gpt-5.1, score: 1439
    "gpt-oss-20b": 184,  # gpt-oss-20b, score: 1318
    "gpt-oss-20b-thinking": 184,  # Same model in thinking mode
    "gpt-oss-120b": 134,  # gpt-oss-120b, score: 1354
    "gpt-oss-120b-thinking": 134,  # Same model in thinking mode
    "o3": 43,  # o3-2025-04-16, score: 1431
    "o3-thinking": 43,  # o3-2025-04-16, score: 1431
    "o3-mini": 142,  # o3-mini, score: 1348
    "o3-mini-thinking": 142,  # o3-mini, score: 1348
    # Anthropic
    "sonnet-4.5": 25,  # claude-sonnet-4-5-20250929, score: 1453
    "sonnet-4.5-thinking": 25,  # claude-sonnet-4-5-20250929, score: 1453
    "sonnet-3.7": 115,  # claude-3-7-sonnet-20250219, score: 1386
    "sonnet-3.7-thinking": 97,  # claude-3-7-sonnet-20250219-thinking-32k, score: 1379
    "haiku-3.5": 169,  # claude-3-5-haiku-20241022, score: 1336
    "haiku-3.5-thinking": 169,  # claude-3-5-haiku-20241022, score: 1336
    "haiku-4.5": 76,  # claude-haiku-4-5-20251001, score: 1407
    "haiku-4.5-thinking": 76,  # claude-haiku-4-5-20251001, score: 1407
    "opus-4.1": 70,  # claude-opus-4-20250514, score: 1412
    "opus-4.1-thinking": 70,  # claude-opus-4-20250514, score: 1412
    # Google
    "gemini-2.0-flash": 129,  # gemini-2.0-flash-001, score: 1360
    "gemini-2.0-flash-thinking": 129,  # gemini-2.0-flash-001, score: 1360
    "gemini-2.0-flash-lite": 112,  # gemini-2.5-flash-lite-preview-09-2025-no-thinking, score: 1380
    "gemini-2.0-flash-lite-thinking": 112,  # gemini-2.5-flash-lite, score: 1380
    "gemini-2.5-flash": 73,  # gemini-2.5-flash, score: 1411
    "gemini-2.5-flash-thinking": 73,  # gemini-2.5-flash, score: 1411
    "gemini-2.5-pro": 28,  # gemini-2.5-pro, score: 1448
    "gemini-2.5-pro-thinking": 28,  # gemini-2.5-pro, score: 1448
    # XAI
    "grok-3-mini": 123,  # grok-3-mini-high, score: 1378
    "grok-3-mini-thinking": 123,  # grok-3-mini-high, score: 1378
    "grok-4.1-fast": 41,  # grok-4-1-fast-reasoning, score: 1435
    "grok-4.1-fast-thinking": 41,  # grok-4-1-fast-reasoning, score: 1435
    # Together - Llama
    "ll-3.1-8b": 260,  # llama-3.1-8b-instruct, score: 1211
    "ll-3.1-70b": 208,  # llama-3.1-70b-instruct, score: 1293
    "ll-3.3-70b": 183,  # llama-3.3-70b-instruct, score: 1318
    "ll-3.3-70b-dsR1-thinking": None,  # Not found on leaderboard
    "ll-3.1-405b": 155,  # llama-3.1-405b-instruct-bf16, score: 1346
    # Together - Qwen
    "qwen-2.5-7b": None,  # Not on leaderboard
    "qwen-2.5-72b": 205,  # qwen2.5-72b-instruct, score: 1302
    "qwen-3.0-30b": 109,  # qwen3-30b-a3b-instruct-2507, score: 1383
    "qwen-3.0-80b": 85,  # qwen3-next-80b-a3b-instruct, score: 1402
    "qwen-3.0-80b-thinking": 85,  # qwen3-next-80b-a3b-instruct, score: 1402
    "qwen-3.0-235b": 54,  # qwen3-235b-a22b-instruct-2507, score: 1422
    "qwen-3.0-235b-thinking": 89,  # qwen3-235b-a22b-thinking-2507, score: 1400
    "qwen-3.5-27b": 79,  # qwen3.5-27b, score: 1405
    # Together - DeepSeek
    "deepseek-3.0": 95,  # deepseek-v3-0324, score: 1395
    "deepseek-3.1": 59,  # deepseek-v3.1, score: 1418
    "deepseek-3.1-thinking": 59,  # deepseek-v3.1, score: 1418
    "deepseek-r1-thinking": 56,  # deepseek-r1-0528, score: 1422
    "deepseek-r1-0528-thinking": 56,  # deepseek-r1-0528, score: 1422
    # Moonshot
    "kimi-k2": 57,  # kimi-k2-0905-preview, score: 1419
    "kimi-k2-thinking": 42,  # kimi-k2-thinking-turbo, score: 1434
    "kimi-k2.5": 42,  # kimi-k2.5-instant, score: 1433
    "kimi-k2.5-thinking": 23,  # kimi-k2.5-thinking, score: 1454
    # Together - MiniMax
    "minimax-m2.5-thinking": 78,  # minimax-m2.5, score: 1406
    # Together - GLM (Zhipu)
    "glm-4.5-air-thinking": 118,  # glm-4.5-air, score: 1373
    "glm-4.7-thinking": 35,  # glm-4.7, score: 1443
    # Fireworks - Llama
    "ll-3.1-8b_fw": 260,  # Same as Together version
    "ll-3.1-70b_fw": 208,  # Same as Together version
    "ll-3.1-405b_fw": 155,  # Same as Together version
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": 109,  # Same as Together version
    "qwen-3.0-235b_fw": 54,  # Same as Together version
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": 59,  # Same as Together version
    "deepseek-r1_fw": 56,  # Same as Together version (deepseek-r1-0528)
}

# LM Arena Elo scores from https://arena.ai/leaderboard (text)
# Scores as of Mar 17, 2026. Higher score = better model.
# Models not found on leaderboard are marked as None.
LM_ARENA_SCORES: dict[str, int | None] = {
    # Updated from https://arena.ai/leaderboard/text on 2026-03-27.
    # OpenAI
    "gpt-4o-mini": 1317,  # rank 188
    "gpt-4o": 1345,  # rank 150 (gpt-4o-2024-05-13)
    "gpt-4.1-mini": 1382,  # rank 111
    "gpt-4.1": 1413,  # rank 69
    "gpt-5-mini": 1382,  # rank 94
    "gpt-5-mini-thinking": 1382,  # rank 94
    "gpt-5": 1439,  # rank 38
    "gpt-5-thinking": 1439,  # rank 38
    "gpt-oss-20b": 1318,  # rank 184
    "gpt-oss-20b-thinking": 1318,  # rank 184
    "gpt-oss-120b": 1354,  # rank 134
    "gpt-oss-120b-thinking": 1354,  # rank 134
    "o3": 1431,  # rank 43
    "o3-thinking": 1431,  # rank 43
    "o3-mini": 1348,  # rank 142
    "o3-mini-thinking": 1348,  # rank 142
    # Anthropic
    "sonnet-4.5": 1453,  # rank 25
    "sonnet-4.5-thinking": 1453,  # rank 25
    "sonnet-3.7": 1386,  # rank 115
    "sonnet-3.7-thinking": 1379,  # rank 97
    "haiku-3.5": 1336,  # rank 169
    "haiku-3.5-thinking": 1336,  # rank 169
    "haiku-4.5": 1407,  # rank 76
    "haiku-4.5-thinking": 1407,  # rank 76
    "opus-4.1": 1412,  # rank 70 (claude-opus-4-20250514)
    "opus-4.1-thinking": 1412,  # rank 70
    # Google
    "gemini-2.0-flash": 1360,  # rank 129 (gemini-2.0-flash-001)
    "gemini-2.0-flash-thinking": 1360,  # rank 129
    "gemini-2.0-flash-lite": 1380,  # rank 112 (gemini-2.5-flash-lite-no-thinking)
    "gemini-2.0-flash-lite-thinking": 1380,  # rank 112
    "gemini-2.5-flash": 1411,  # rank 73
    "gemini-2.5-flash-thinking": 1411,  # rank 73
    "gemini-2.5-pro": 1448,  # rank 28
    "gemini-2.5-pro-thinking": 1448,  # rank 28
    # XAI
    "grok-3-mini": 1378,  # rank 123
    "grok-3-mini-thinking": 1378,  # rank 123
    "grok-4.1-fast": 1435,  # rank 41
    "grok-4.1-fast-thinking": 1435,  # rank 41
    # Together - Llama
    "ll-3.1-8b": 1211,  # rank 260
    "ll-3.1-70b": 1293,  # rank 208
    "ll-3.3-70b": 1318,  # rank 183 (llama-3.3-70b-instruct)
    "ll-3.3-70b-dsR1-thinking": None,  # Not found on leaderboard
    "ll-3.1-405b": 1346,  # rank 155
    # Together - Qwen
    "qwen-2.5-7b": None,  # Not on leaderboard
    "qwen-2.5-72b": 1302,  # rank 205
    "qwen-3.0-30b": 1383,  # rank 109 (qwen3-30b-a3b-instruct-2507)
    "qwen-3.0-80b": 1402,  # rank 85
    "qwen-3.0-80b-thinking": 1402,  # rank 85
    "qwen-3.0-235b": 1422,  # rank 54 (qwen3-235b-a22b-instruct-2507)
    "qwen-3.0-235b-thinking": 1400,  # rank 89 (qwen3-235b-a22b-thinking-2507)
    "qwen-3.5-27b": 1405,  # rank 79 (qwen3.5-27b)
    # Together - DeepSeek
    "deepseek-3.0": 1395,  # rank 95 (deepseek-v3-0324)
    "deepseek-3.1": 1418,  # rank 59 (deepseek-v3.1)
    "deepseek-3.1-thinking": 1418,  # rank 59
    "deepseek-r1-thinking": 1422,  # rank 56 (deepseek-r1-0528)
    "deepseek-r1-0528-thinking": 1422,  # rank 56
    # Moonshot
    "kimi-k2": 1419,  # rank 57
    "kimi-k2-thinking": 1434,  # rank 42
    "kimi-k2.5": 1433,  # rank 42 (kimi-k2.5-instant)
    "kimi-k2.5-thinking": 1454,  # rank 23
    # Together - MiniMax
    "minimax-m2.5-thinking": 1406,  # rank 78
    # Together - GLM (Zhipu)
    "glm-4.5-air-thinking": 1373,  # rank 118
    "glm-4.7-thinking": 1443,  # rank 35
    # Fireworks - Llama
    "ll-3.1-8b_fw": 1211,  # Same as Together version
    "ll-3.1-70b_fw": 1293,  # Same as Together version
    "ll-3.1-405b_fw": 1346,  # Same as Together version
    # Fireworks - Qwen
    "qwen-3.0-30b_fw": 1383,  # Same as Together version
    "qwen-3.0-235b_fw": 1422,  # Same as Together version
    # Fireworks - DeepSeek
    "deepseek-3.1_fw": 1418,  # Same as Together version
    "deepseek-r1_fw": 1422,  # Same as Together version
}

# GPU tier for hf/ models that need local GPU inference.
# Tier determines which RunPod GPU config to use:
#   "small"  — 8B models, ~16GB VRAM (RTX 3090, RTX 4090, A40, etc.)
#   "medium" — 27B-35B models, ~60GB VRAM (A100 80GB, L40S)
#   "large"  — 70B+ models, ~80GB+ VRAM (A100 80GB, H100)
MODEL_GPU_TIER: dict[str, str] = {
    "ll-3.1-8b": "small",
    "qwen-3.5-27b": "medium",
    "ll-3.1-70b": "large",
    "ll-3.1-405b": "large",
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
    # Strip an optional provider tag (hf:/tg:) so base-model logic (max-tokens,
    # thinking detection, data lookup) is provider-agnostic. The tag only steers
    # the inspect route (INSPECT_MODEL_NAMES lookup), which uses the full name.
    model_name = strip_provider_tag(model_name)
    if model_name.endswith("-thinking"):
        return model_name[:-9]  # Remove "-thinking" suffix
    return model_name


# Provider tags let a config disambiguate which backend a shorthand resolves to
# when the same model is served by more than one provider (e.g. gpt-oss on both
# hf and Together). A tagged name like "tg:gpt-oss-20b-thinking" has its own
# explicit INSPECT_MODEL_NAMES entry; bare names remain the canonical default.
PROVIDER_TAGS = ("hf:", "tg:")


def strip_provider_tag(model_name: str) -> str:
    """Remove a leading provider tag (hf:/tg:) if present."""
    for tag in PROVIDER_TAGS:
        if model_name.startswith(tag):
            return model_name[len(tag):]
    return model_name


def get_model_output_token_cap(model_name: str) -> int:
    """
    Look up a model's maximum output-token ceiling from MODEL_OUTPUT_TOKEN_CAP.

    Resolves the `"max"` token sentinel used in experiment configs. The lookup
    tries the exact name first, then the base name (with the "-thinking" suffix
    stripped), so a single table entry covers both variants of a model.

    Args:
        model_name: Short model name (may include the "-thinking" suffix).

    Returns:
        The model's maximum output tokens.

    Raises:
        ValueError: If the model is not in MODEL_OUTPUT_TOKEN_CAP — either set an
            explicit token count in the config, or add the model to the table.
    """
    if model_name in MODEL_OUTPUT_TOKEN_CAP:
        return MODEL_OUTPUT_TOKEN_CAP[model_name]

    base_name = get_base_model_name(model_name)
    if base_name in MODEL_OUTPUT_TOKEN_CAP:
        return MODEL_OUTPUT_TOKEN_CAP[base_name]

    raise ValueError(
        f"No max-token entry for model '{model_name}'. Either set an explicit "
        f"token count in the experiment config instead of 'max', or add '{base_name}' "
        f"to MODEL_OUTPUT_TOKEN_CAP in self_rec_framework/src/helpers/model_names.py."
    )


def get_model_context_window(model_name: str) -> int:
    """
    Look up a model's total context window (max input + output tokens) from
    MODEL_CONTEXT_WINDOW.

    Used when resolving a "max" output budget so that input + output stays within
    the model's context window. Tries the exact name first, then the base name
    (with the "-thinking" suffix stripped).

    Args:
        model_name: Short model name (may include the "-thinking" suffix).

    Returns:
        The model's total context window in tokens.

    Raises:
        ValueError: If the model is not in MODEL_CONTEXT_WINDOW — add it to the
            table (a context window is a fixed model spec, not configurable).
    """
    if model_name in MODEL_CONTEXT_WINDOW:
        return MODEL_CONTEXT_WINDOW[model_name]

    base_name = get_base_model_name(model_name)
    if base_name in MODEL_CONTEXT_WINDOW:
        return MODEL_CONTEXT_WINDOW[base_name]

    raise ValueError(
        f"No context-window entry for model '{model_name}'. Add '{base_name}' to "
        f"MODEL_CONTEXT_WINDOW in self_rec_framework/src/helpers/model_names.py "
        f"(the context window is a fixed model spec — look it up from provider docs)."
    )


def get_model_max_thinking_tokens(model_name: str) -> int:
    """
    Look up a model's maximum thinking/CoT-token budget.

    Resolves the `"max"` sentinel for max_thinking_tokens. Most models have no
    separate thinking cap (reasoning and answer share one output budget), so this
    falls back to get_model_output_token_cap(). Only models in MODEL_MAX_THINKING_TOKENS
    (e.g. Gemini 2.5, whose `thinkingBudget` caps below `maxOutputTokens`) override.

    Args:
        model_name: Short model name (may include the "-thinking" suffix).

    Returns:
        The model's maximum thinking-token budget.

    Raises:
        ValueError: If the model is unknown (via get_model_output_token_cap fallback).
    """
    if model_name in MODEL_MAX_THINKING_TOKENS:
        return MODEL_MAX_THINKING_TOKENS[model_name]

    base_name = get_base_model_name(model_name)
    if base_name in MODEL_MAX_THINKING_TOKENS:
        return MODEL_MAX_THINKING_TOKENS[base_name]

    # No separate thinking cap — bounded by the shared output ceiling.
    return get_model_output_token_cap(model_name)


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
    # Strip any provider tag (hf:/tg:) — data identity is provider-agnostic, so a
    # tagged route (e.g. tg:gpt-oss-20b-thinking) reads the same data as the bare name.
    model_name = strip_provider_tag(model_name)

    # Strip -thinking suffix first for trained model resolution
    clean_name = model_name.removesuffix("-thinking") if model_name.endswith("-thinking") else model_name

    # Trained models: resolve to base model for data lookup.
    # Standard SGTR-trained: "<base>_sft-as_<self>_vs_..."
    # Random-labels control:  "<base>_RANDLABELS_<seed>_vs_..."
    # Both shapes use the same base-extraction rule (split on the marker).
    base_part = None
    if "_sft-as_" in clean_name:
        base_part = clean_name.split("_sft-as_")[0]
    elif "_RANDLABELS_" in clean_name:
        base_part = clean_name.split("_RANDLABELS_")[0]
    if base_part is not None:
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
