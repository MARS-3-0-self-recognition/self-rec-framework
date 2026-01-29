#!/bin/bash
# Auto-configured analysis script
# Finds most recent aggregated performance data and generates plots

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Auto-detect experiment directory name
# Path structure: experiments/{EXP_DIR}/bash/analysis/{dataset_name}/
EXP_DIR="$(basename "$(cd "$SCRIPT_DIR/../../.." && pwd)")"

# ============================================================================
# Find most recent aggregated_performance.csv file
# ============================================================================

EXP_BASE_DIR="data/analysis/_aggregated_data"
# Look for directory matching experiment name
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

echo "=============================================================================="
echo "Generating Plots from Data: $LATEST_DIR"
echo "=============================================================================="

# Check for performance data
AGGREGATED_FILE="$LATEST_DIR/aggregated_performance.csv"
if [[ -f "$AGGREGATED_FILE" ]]; then
    echo "Processing Performance Data..."
    uv run "experiments/_scripts/analysis/plot_aggregated_performance.py" \
            --aggregated_file "$AGGREGATED_FILE"
else
    echo "Warning: aggregated_performance.csv not found in $LATEST_DIR"
fi

# Check for deviation data
DEVIATION_FILE="$LATEST_DIR/aggregated_deviation.csv"
if [[ -f "$DEVIATION_FILE" ]]; then
    echo "Processing Deviation Data..."
    uv run "experiments/_scripts/analysis/plot_aggregated_performance.py" \
            --aggregated_file "$DEVIATION_FILE"
fi

echo ""
echo "Done! Plots saved to: $LATEST_DIR"
