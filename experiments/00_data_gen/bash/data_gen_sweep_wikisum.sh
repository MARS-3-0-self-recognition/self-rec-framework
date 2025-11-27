#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-5 \
    --dataset_path=data/input/wikisum/training_set_1-20 \
    --dataset_config=experiments/00_data_gen/configs/config.yaml \
    --batch --overwrite
