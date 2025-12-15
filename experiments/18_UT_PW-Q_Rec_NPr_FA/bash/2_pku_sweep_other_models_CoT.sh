#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names gpt-oss-20b-thinking gpt-oss-120b-thinking \
                ll-3.3-70b-dsR1-thinking \
                qwen-3.0-80b-thinking \
                deepseek-r1-thinking \
                gemini-2.5-flash-thinking gemini-2.5-pro-thinking\
                sonnet-4.5-thinking sonnet-3.7-thinking opus-4.1-thinking \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \
    --experiment_config experiments/18_UT_PW-Q_Rec_NPr_FA/config.yaml \
    --max-tasks 24 --batch
