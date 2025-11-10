#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-4o-mini gpt-4.1-mini gpt-4.1 qwen-3.0-80b ll-3.1-8b ll-3.1-70b ll-3.1-405b deepseek-3.1 sonnet-4.5 sonnet-3.7 haiku-3.5 gemini-2.0-flash \
    --dataset_path=data/wikisum/training_set_1-20/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
