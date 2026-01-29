#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names -set test_dr \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/11_UT_PW-Q_Rec_NPr/config.yaml \
    --max-tasks 16 --batch
