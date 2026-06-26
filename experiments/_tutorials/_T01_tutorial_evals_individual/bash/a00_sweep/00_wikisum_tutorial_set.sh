#!/bin/bash
# Tutorial eval sweep — wikisum. Compares models against each other.
# All parameters come from the experiment config.yaml (single source of truth),
# loaded by the shared scripts/utils/load_config.sh. Run from the repo
# root, directly or via run_sweep.sh / scripts/utils/run.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="$SCRIPT_DIR/../../config.yaml"
DATASET_DIR_PATH="data/input/wikisum/tutorial_set"

source scripts/utils/load_config.sh "$CONFIG_FILE"

uv run srf-eval-sweep \
    --model_names "${MODEL_NAMES[@]}" \
    "${GENERATOR_MODELS_ARG[@]}" \
    --treatment_type "$TREATMENT_TYPE" \
    --dataset_dir_path "$DATASET_DIR_PATH" \
    --experiment_config "$CONFIG_FILE" \
    --max-tasks "$MAX_TASKS" \
    "${BATCH_ARG[@]}" \
    "${YES_ARG[@]}"
