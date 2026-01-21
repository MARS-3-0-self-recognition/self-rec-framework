#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names -set eval_cot-r \
    --dataset_path=data/input/sharegpt/english2_74/input.json \
    --dataset_config=experiments/00_data_gen/config.yaml \
    --batch
