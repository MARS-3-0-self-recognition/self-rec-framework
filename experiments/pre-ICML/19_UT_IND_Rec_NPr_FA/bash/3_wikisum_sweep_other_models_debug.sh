#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names -set test \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/debug \
    --experiment_config experiments/19_UT_IND_Rec_NPr_FA/config.yaml \
    --max-tasks 16 --batch
