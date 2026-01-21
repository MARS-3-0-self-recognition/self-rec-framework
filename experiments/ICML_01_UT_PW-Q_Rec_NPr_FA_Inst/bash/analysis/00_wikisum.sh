#!/bin/bash
# Analyze combined wikisum subsets

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_DIR="$(basename "$(cd "$SCRIPT_DIR/../.." && pwd)")"

# List all wikisum subsets to combine
# Each subset adds more datapoints to the same evaluations

DATASET_PATHS=(
    "data/results/wikisum/training_set_1-20/$EXP_DIR"
    "data/results/wikisum/test_set_1-30/$EXP_DIR"
)

uv run experiments/_scripts/analyze_pairwise_results.py \
        --results_dir "${DATASET_PATHS[@]}" \
        --model_names -set dr
