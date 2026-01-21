#!/bin/bash
# Sweep experiment: Compare models against their caps treatments

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names gpt-4o-mini gpt-4.1 haiku-3.5 \
    --treatment_type caps \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/01_AT_PW-C_Rec_Pr/config.yaml
