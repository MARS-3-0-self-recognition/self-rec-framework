#!/bin/bash
# Tutorial sweep data generation (sharegpt).
# model_names and gen params come from the local config.yaml (single source of
# truth), loaded by the shared scripts/utils/load_config.sh. Run from the
# self-rec-research repo root so data lands in ./data/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.yaml"

source scripts/utils/load_config.sh "$CONFIG_FILE"

uv run srf-generate-sweep \
    --model_names "${MODEL_NAMES[@]}" \
    --dataset_path=data/input/sharegpt/tutorial_set/input.json \
    --dataset_config="$CONFIG_FILE"
