#!/bin/bash
# Batch experiment: Compare models against each other

uv run experiments/scripts/run_experiment_batch.py \
    --model_names gpt-4.1-mini qwen3-30b-a3b ll-3-1-70b deepseek-v3 sonnet-3-7 gemini-2.0-flash \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml \
    --max_workers 6
