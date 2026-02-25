#!/bin/bash
# Auto-configured analysis script
# Script name is extracted from filename: 03-{script_name}.sh -> {script_name}.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="$(basename "${BASH_SOURCE[0]}")"

# Extract script name from filename
# Supports ##-{name}.sh
# Remove numeric prefix and separator, then remove .sh extension
SCRIPT_NAME="${SCRIPT_FILE#*-}"  # Remove prefix up to first "-"
SCRIPT_NAME="${SCRIPT_NAME%.sh}"  # Remove ".sh" extension

# Default: assume script_name matches Python file name (replacing - with _)
ANALYSIS_SCRIPT="${SCRIPT_NAME//-/_}.py"

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
# Construct input file paths (accuracy_pivot.csv for each dataset)
# ============================================================================

ACCURACY_FILES=()
DATA_BASE_DIR="data/analysis"

for subset in "${DATASET_SUBSETS[@]}"; do
    # Subset format: dataset_name/subset_name
    # Expected path: data/analysis/{dataset}/{subset}/{EXP_DIR}/recognition_accuracy/accuracy_pivot.csv
    
    FILE="$DATA_BASE_DIR/$subset/$EXP_DIR/recognition_accuracy/accuracy_pivot.csv"
    
    if [[ -f "$FILE" ]]; then
        ACCURACY_FILES+=("$FILE")
    else
        echo "Warning: accuracy_pivot.csv not found for $subset ($FILE)"
    fi
done

if [ ${#ACCURACY_FILES[@]} -eq 0 ]; then
    echo "Error: No accuracy_pivot.csv files found. Please run individual analysis steps (00-recognition_accuracy.sh) first."
    exit 1
fi

# ============================================================================
# Find output directory (most recent aggregated data dir)
# ============================================================================

EXP_BASE_DIR="data/analysis/_aggregated_data"
EXP_PATTERN="${EXP_DIR}"

# Find experiment directory
EXP_EXP_DIR=$(ls -d "$EXP_BASE_DIR"/$EXP_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP_EXP_DIR" ]] || [[ ! -d "$EXP_EXP_DIR" ]]; then
    echo "Error: No aggregated data directory found for: $EXP_DIR"
    echo "  Please run 00a-performance_aggregate.sh first."
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
LATEST_DIR=$(ls -td "$EXP_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$LATEST_DIR" ]] || [[ ! -d "$LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP_EXP_DIR"
    echo "  Please run 00a-performance_aggregate.sh first."
    exit 1
fi

# Remove trailing slash if present
LATEST_DIR="${LATEST_DIR%/}"

echo "=============================================================================="
echo "Running Rank Distance Analysis (Excluding Self-Comparisons)"
echo "=============================================================================="
echo "Output Directory: $LATEST_DIR"
echo "Datasets found: ${#ACCURACY_FILES[@]}"
echo "Model Names: ${MODEL_NAMES[@]}"

# ============================================================================
# Run analysis script
# ============================================================================

# Build arguments
ARGS=("--accuracy_files" "${ACCURACY_FILES[@]}" "--output_dir" "$LATEST_DIR" "--exclude_self")

if [ ${#MODEL_NAMES[@]} -gt 0 ]; then
    ARGS+=("--model_names" "${MODEL_NAMES[@]}")
fi

uv run "experiments/_scripts/analysis/$ANALYSIS_SCRIPT" "${ARGS[@]}"
