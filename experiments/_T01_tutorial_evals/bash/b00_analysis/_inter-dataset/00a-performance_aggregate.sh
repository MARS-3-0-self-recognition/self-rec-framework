#!/bin/bash
# Auto-configured analysis script
# Runs performance aggregation only (creates CSV files)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
# Build paths to evaluator_performance.csv files
# ============================================================================

PERFORMANCE_FILES=()
DATASET_NAMES=()

for dataset_path in "${DATASET_SUBSETS[@]}"; do
    # Construct full path to evaluator_performance.csv
    # Format: data/analysis/{dataset_path}/{experiment}/evaluator_performance/evaluator_performance.csv
    PERFORMANCE_FILE="data/analysis/$dataset_path/$EXP_DIR/evaluator_performance/evaluator_performance.csv"

    if [[ -f "$PERFORMANCE_FILE" ]]; then
        PERFORMANCE_FILES+=("$PERFORMANCE_FILE")
        DATASET_NAMES+=("$dataset_path")
    else
        echo "Warning: Performance file not found: $PERFORMANCE_FILE"
    fi
done

if [[ ${#PERFORMANCE_FILES[@]} -eq 0 ]]; then
    echo "Error: No valid performance files found!"
    exit 1
fi

# ============================================================================
# Define Output Directory
# ============================================================================

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="data/analysis/_aggregated_data/$EXP_DIR/$TIMESTAMP"
mkdir -p "$OUTPUT_DIR"

echo "=============================================================================="
echo "Running Data Aggregation"
echo "=============================================================================="

# Run Data Aggregation
uv run "experiments/_scripts/analysis/aggregate_performance_data.py" \
        --performance_files "${PERFORMANCE_FILES[@]}" \
        --dataset_names "${DATASET_NAMES[@]}" \
        --model_names "${MODEL_NAMES[@]}" \
        --output_dir "$OUTPUT_DIR"

if [[ $? -ne 0 ]]; then
    echo "Error: Data aggregation failed."
    exit 1
fi

echo ""
echo "Done! Aggregated data saved to: $OUTPUT_DIR"
