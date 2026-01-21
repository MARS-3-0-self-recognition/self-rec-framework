#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-oss-20b-thinking \
                ll-3.3-70b-dsR1-thinking \
    --dataset_path=data/input/sharegpt/debug/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
