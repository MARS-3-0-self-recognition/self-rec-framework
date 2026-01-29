#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names -set test_dr \
    --treatment_type other_models \
    --dataset_dir_path data/input/pku_saferlhf/test-mismatch_10_100-200 \
    --experiment_config experiments/11_UT_PW-Q_Rec_NPr/config.yaml \
    --max-tasks 16 --batch
