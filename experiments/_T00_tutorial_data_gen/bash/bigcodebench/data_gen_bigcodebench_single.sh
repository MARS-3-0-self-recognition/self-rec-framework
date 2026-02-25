#!/bin/bash
# Generate WikiSum data using current architecture

uv run experiments/scripts/generate_data.py \
    --model_name=ll-3.1-8b \
    --dataset_path=data/input/bigcodebench/instruct_1-50/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
