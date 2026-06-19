#!/bin/bash
# Tutorial sweep data generation for multiple models.
# model_names is read from config.yaml (single source of truth); overwrite is
# declared in config.yaml, so tutorials regenerate by default.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.yaml"

# Read model_names from config.yaml
MODEL_NAMES=$(python3 -c "
import yaml
with open('$CONFIG_FILE') as f:
    config = yaml.safe_load(f)
print(' '.join(config.get('model_names', [])))
")

if [ -z "$MODEL_NAMES" ]; then
    echo "Error: No model_names found in $CONFIG_FILE"
    exit 1
fi

uv run srf-generate-sweep \
    --model_names $MODEL_NAMES \
    --dataset_path=data/input/bigcodebench/tutorial_set/input.json \
    --dataset_config="$CONFIG_FILE"
