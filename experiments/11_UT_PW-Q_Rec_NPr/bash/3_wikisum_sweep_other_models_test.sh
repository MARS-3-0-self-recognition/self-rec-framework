#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names haiku-3.5 \
                  qwen-2.5-7b \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/11_UT_PW-Q_Rec_NPr/config.yaml \
    --max-tasks 16 --batch
