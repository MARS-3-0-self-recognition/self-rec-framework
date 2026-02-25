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
ANALYSIS_SCRIPT="${SCRIPT_NAME}.py"

# Auto-detect experiment directory name
# Path structure: experiments/{EXP_DIR}/bash/analysis/{dataset_name}/
EXP_DIR="$(basename "$(cd "$SCRIPT_DIR/../../.." && pwd)")"

# Auto-detect dataset name from current directory
DATASET_NAME="$(basename "$SCRIPT_DIR")"
DATASET_PATH="data/results/$DATASET_NAME"

# ============================================================================
# Load configuration from shared config file
# ============================================================================

CONFIG_FILE="$SCRIPT_DIR/config.sh"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create config.sh in the same directory with DATASET_SUBSETS and MODEL_NAMES arrays."
    exit 1
fi

# Source the configuration file (loads DATASET_SUBSETS and MODEL_NAMES)
source "$CONFIG_FILE"

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

uv run "experiments/_scripts/analysis/$ANALYSIS_SCRIPT" \
        --results_dir "${FULL_DATASET_PATHS[@]}" \
        --model_names "${MODEL_NAMES[@]}"
