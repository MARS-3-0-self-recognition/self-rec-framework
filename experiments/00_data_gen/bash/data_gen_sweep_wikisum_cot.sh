#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-oss-20b-thinking gpt-oss-120b-thinking \
                ll-3.3-70b-dsR1-thinking \
                qwen-3.0-80b-thinking \
                deepseek-r1-thinking \
                gemini-2.5-flash-thinking gemini-2.5-pro-thinking\
                sonnet-4.5-thinking sonnet-3.7-thinking opus-4.1-thinking \
    --dataset_path=data/input/wikisum/training_set_1-20 \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
