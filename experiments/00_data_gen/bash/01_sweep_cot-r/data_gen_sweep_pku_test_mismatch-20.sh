#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/_scripts/generate_data_sweep.py \
    --model_names -set eval_cot-r \
    --dataset_path=data/input/pku_saferlhf/test_mismatch_1-20/input.json \
    --dataset_config=experiments/00_data_gen/config.yaml \
