#!/bin/bash
# Example sweep data generation script for multiple models

uv run experiments/scripts/generate_data_sweep.py \
    --model_names gpt-oss-20b-thinking gpt-oss-120b-thinking \
                sonnet-3.7-thinking \
                grok-3-mini-thinking \
                ll-3.3-70b-dsR1-thinking \
                qwen-3.0-80b-thinking qwen-3.0-235b-thinking \
                deepseek-r1-thinking \
                kimi-k2-thinking \
    --dataset_path=data/input/pku_saferlhf/mismatch_1-20/input.json \
    --dataset_config=experiments/00_data_gen/configs/config.yaml
