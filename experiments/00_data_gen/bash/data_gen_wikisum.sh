#!/bin/bash
# Generate WikiSum data using current architecture

uv run src/data_gen/gen.py \
    --model_name=haiku-3-5 \
    --model_generation_string=wikisum_config \
    --dataset_name=wikisum_train_1-20
