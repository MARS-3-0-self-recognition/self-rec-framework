#!/bin/bash
# Generate PKU SaferLHF debug data using current architecture

uv run experiments/scripts/generate_data.py \
    --model_name=haiku-3.5 \
    --dataset_path=data/input/pku_saferlhf/debug/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
