#!/bin/bash
# Sweep experiment: Compare models against each other

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run experiments/_scripts/run_experiment_sweep.py \
    --model_names -set test \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/debug \
    --experiment_config "$SCRIPT_DIR/../../config.yaml" \
    --max-tasks 2 \
    --overwrite
