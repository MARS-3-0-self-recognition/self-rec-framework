#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names -set test_eval_cot-r_and_cot-i \
    --dataset_path=data/input/wikisum/debug \
    --dataset_config=experiments/00_data_gen/configs/config.yaml \
