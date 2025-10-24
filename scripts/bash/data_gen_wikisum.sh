#!/bin/bash
# Generate WikiSum data using current architecture

uv run src/data_gen/gen.py \
    --model_name=3-5-haiku \
    --model_generation_string=wikisum_config \
    --dataset_name=wikisum_debug
