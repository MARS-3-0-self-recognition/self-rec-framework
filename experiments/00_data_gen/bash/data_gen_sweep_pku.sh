#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gemini-2.0-flash-lite\
    --dataset_path=data/input/pku_saferlhf/mismatch_1-20/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
