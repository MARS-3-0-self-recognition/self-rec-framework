#!/bin/bash
# Auto-configured analysis script
# Script name is extracted from filename: 01-{script_name}.sh -> {script_name}.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="$(basename "${BASH_SOURCE[0]}")"

# Extract script name from filename
# Supports ##-{name}.sh
# Remove numeric prefix and separator, then remove .sh extension
SCRIPT_NAME="${SCRIPT_FILE#*-}"  # Remove prefix up to first "-"
SCRIPT_NAME="${SCRIPT_NAME%.sh}"  # Remove ".sh" extension


# Default: assume script_name matches Python file name
ENTRY_POINT="srf-${SCRIPT_NAME//_/-}"

# Auto-detect experiment directory name
# Path structure: experiments/{EXP_DIR}/bash/analysis/{dataset_name}/
EXP_DIR="$(basename "$(cd "$SCRIPT_DIR/../../.." && pwd)")"

# Auto-detect dataset name from current directory
DATASET_NAME="$(basename "$SCRIPT_DIR")"
DATASET_PATH="data/results/$DATASET_NAME"

# ============================================================================
# Load configuration from the experiment config.yaml (shared loader)
# ============================================================================

CONFIG_FILE="$SCRIPT_DIR/../../../config.yaml"
source scripts/utils/load_config.sh "$CONFIG_FILE"

# ============================================================================
# Build full dataset paths
# ============================================================================

FULL_DATASET_PATHS=()
for subset in "${DATASET_SUBSETS[@]}"; do
    FULL_DATASET_PATHS+=("$DATASET_PATH/$subset/$EXP_DIR")
done

# ============================================================================
# Run analysis script
# ============================================================================

uv run "$ENTRY_POINT" \
        --results_dir "${FULL_DATASET_PATHS[@]}" \
        --model_names "${MODEL_NAMES[@]}"
