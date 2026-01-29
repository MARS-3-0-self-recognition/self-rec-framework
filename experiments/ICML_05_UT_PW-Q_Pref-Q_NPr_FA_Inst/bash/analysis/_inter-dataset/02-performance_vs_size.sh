#!/bin/bash
# Auto-configured analysis script
# Script name is extracted from filename: 02-{script_name}.sh -> {script_name}.py

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
# Find most recent aggregated_performance.csv file
# ============================================================================

EXP_BASE_DIR="data/analysis/_aggregated_data"
EXP_PATTERN="${EXP_DIR}"

# Find experiment directory
EXP_EXP_DIR=$(ls -d "$EXP_BASE_DIR"/$EXP_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP_EXP_DIR" ]] || [[ ! -d "$EXP_EXP_DIR" ]]; then
    echo "Error: No experiment directory found for: $EXP_DIR"
    echo "  Searched in: $EXP_BASE_DIR for pattern: $EXP_PATTERN"
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
LATEST_DIR=$(ls -td "$EXP_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$LATEST_DIR" ]] || [[ ! -d "$LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP_EXP_DIR"
    exit 1
fi

# Remove trailing slash if present
LATEST_DIR="${LATEST_DIR%/}"

AGGREGATED_FILE="$LATEST_DIR/aggregated_performance.csv"

if [[ ! -f "$AGGREGATED_FILE" ]]; then
    echo "Error: aggregated_performance.csv not found: $AGGREGATED_FILE"
    echo ""
    echo "Please run 00-performance_aggregate.sh first to generate the aggregated performance data."
    exit 1
fi

# ============================================================================
# Run analysis script
# ============================================================================

uv run "experiments/_scripts/analysis/$ANALYSIS_SCRIPT" \
        --aggregated_file "$AGGREGATED_FILE" \
        --model_names "${MODEL_NAMES[@]}"
