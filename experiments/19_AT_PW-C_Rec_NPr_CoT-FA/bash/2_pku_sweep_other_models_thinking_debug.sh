#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names deepseek-r1-thinking gpt-oss-20b-thinking ll-3.3-70b-dsR1-thinking \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/debug \
    --experiment_config experiments/19_AT_PW-C_Rec_NPr_CoT-FA/config.yaml \
    --max-tasks 2 --overwrite
