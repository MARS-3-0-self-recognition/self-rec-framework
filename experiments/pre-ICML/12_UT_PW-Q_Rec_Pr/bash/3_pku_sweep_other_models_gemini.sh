#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names gemini-2.0-flash gemini-2.0-flash-lite gemini-2.5-flash gemini-2.5-pro \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/mismatch_1-20 \
    --experiment_config experiments/12_UT_PW-Q_Rec_Pr/config.yaml \
    --max-tasks 8
