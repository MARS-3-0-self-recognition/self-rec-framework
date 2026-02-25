#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/_scripts/generate_data_sweep.py \
    --model_names -set tutorial \
    --dataset_path=data/input/wikisum/tutorial_set/input.json \
    --dataset_config=experiments/_T00_tutorial_data_gen/config.yaml \
