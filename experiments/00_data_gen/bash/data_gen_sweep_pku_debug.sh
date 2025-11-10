#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-4.1-mini qwen-3.0-80b ll-3.1-70b deepseek-3.1 sonnet-3.7 gemini-2.0-flash \
    --dataset_path=data/pku_saferlhf/debug/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
