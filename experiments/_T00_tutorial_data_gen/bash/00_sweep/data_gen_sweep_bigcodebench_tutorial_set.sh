#!/bin/bash
# Example sweep data generation script for multiple models

uv run srf-generate-sweep \
    --model_names -set tutorial \
    --dataset_path=data/input/bigcodebench/tutorial_set/input.json \
    --dataset_config=experiments/_T00_tutorial_data_gen/config.yaml \
