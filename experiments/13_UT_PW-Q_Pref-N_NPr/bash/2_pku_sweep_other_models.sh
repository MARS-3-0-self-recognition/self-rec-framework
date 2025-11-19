#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names gpt-4o gpt-4.1-mini gpt-4.1 gpt-4o-mini \
                  sonnet-4.5 haiku-3.5 sonnet-3.7 opus-4.1 \
                  ll-3.1-8b ll-3.1-70b ll-3.1-405b \
                  qwen-2.5-7b qwen-2.5-72b qwen-3.0-80b \
                  deepseek-3.1 \
                  gemini-2.0-flash gemini-2.0-flash-lite gemini-2.5-flash gemini-2.5-pro \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \
    --experiment_config experiments/13_UT_PW-Q_Pref-N_NPr/config.yaml \
    --max-tasks 16 --batch
