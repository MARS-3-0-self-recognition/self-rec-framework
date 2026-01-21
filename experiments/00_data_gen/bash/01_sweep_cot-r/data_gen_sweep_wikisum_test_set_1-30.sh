#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/_scripts/generate_data_sweep.py \
    --model_names -set eval_cot-r \
    --dataset_path=data/input/wikisum/test_set_1-30 \
    --dataset_config=experiments/00_data_gen/config.yaml \
    --batch
