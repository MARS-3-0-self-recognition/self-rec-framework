#!/bin/bash
# Auto-configured analysis script
# Script name is extracted from filename: 00-{script_name}.sh -> {script_name}.py

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT_FILE="$(basename "${BASH_SOURCE[0]}")"

# Extract script name from filename
# Supports ##-{name}.sh
# Remove numeric prefix and separator, then remove .sh extension
SCRIPT_NAME="${SCRIPT_FILE#*-}"  # Remove prefix up to first "-"
SCRIPT_NAME="${SCRIPT_NAME%.sh}"  # Remove ".sh" extension

# Map script name to Python file (experiment comparison uses different script)
if [[ "$SCRIPT_NAME" == "performance_contrast" ]]; then
    ANALYSIS_SCRIPT="experiment_contrast.py"
else
    ANALYSIS_SCRIPT="${SCRIPT_NAME}.py"
fi

# Extract experiment names from directory name (e.g., ICML_01-vs-ICML_02)
COMPARISON_DIR="$(basename "$SCRIPT_DIR")"
if [[ "$COMPARISON_DIR" == *"-vs-"* ]]; then
    EXP1="${COMPARISON_DIR%%-vs-*}"  # Everything before "-vs-"
    EXP2="${COMPARISON_DIR#*-vs-}"   # Everything after "-vs-"
else
    echo "Error: Directory name must be in format 'exp1-vs-exp2'"
    exit 1
fi

# ============================================================================
# Load configuration from shared config file
# ============================================================================

CONFIG_FILE="$SCRIPT_DIR/config.sh"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create config.sh in the same directory with MODEL_NAMES array."
    exit 1
fi

# Source the configuration file (loads MODEL_NAMES)
source "$CONFIG_FILE"

# ============================================================================
# Find most recent aggregated performance files for each experiment
# ============================================================================

EXP1_BASE_DIR="data/analysis/_aggregated_data"
EXP1_PATTERN="${EXP1}_*"

# Find experiment directory (e.g., ICML_01_UT_PW-Q_Rec_NPr_FA_Inst)
EXP1_EXP_DIR=$(ls -d "$EXP1_BASE_DIR"/$EXP1_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP1_EXP_DIR" ]] || [[ ! -d "$EXP1_EXP_DIR" ]]; then
    echo "Error: No experiment directory found for: $EXP1"
    echo "  Searched in: $EXP1_BASE_DIR for pattern: $EXP1_PATTERN"
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
EXP1_LATEST_DIR=$(ls -td "$EXP1_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$EXP1_LATEST_DIR" ]] || [[ ! -d "$EXP1_LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP1_EXP_DIR"
    exit 1
fi

# Remove trailing slash if present
EXP1_LATEST_DIR="${EXP1_LATEST_DIR%/}"

EXP1_FILE="$EXP1_LATEST_DIR/aggregated_performance.csv"
if [[ ! -f "$EXP1_FILE" ]]; then
    echo "Error: Performance file not found: $EXP1_FILE"
    exit 1
fi

# Find experiment directory for exp2
EXP2_PATTERN="${EXP2}_*"
EXP2_EXP_DIR=$(ls -d "$EXP1_BASE_DIR"/$EXP2_PATTERN 2>/dev/null | head -1)

if [[ -z "$EXP2_EXP_DIR" ]] || [[ ! -d "$EXP2_EXP_DIR" ]]; then
    echo "Error: No experiment directory found for: $EXP2"
    echo "  Searched in: $EXP1_BASE_DIR for pattern: $EXP2_PATTERN"
    exit 1
fi

# Find most recent timestamp subdirectory within experiment directory
EXP2_LATEST_DIR=$(ls -td "$EXP2_EXP_DIR"/*/ 2>/dev/null | head -1)

if [[ -z "$EXP2_LATEST_DIR" ]] || [[ ! -d "$EXP2_LATEST_DIR" ]]; then
    echo "Error: No timestamp directories found in: $EXP2_EXP_DIR"
    exit 1
fi

# Remove trailing slash if present
EXP2_LATEST_DIR="${EXP2_LATEST_DIR%/}"

EXP2_FILE="$EXP2_LATEST_DIR/aggregated_performance.csv"
if [[ ! -f "$EXP2_FILE" ]]; then
    echo "Error: Performance file not found: $EXP2_FILE"
    exit 1
fi

echo "Comparing experiments:"
echo "  Exp1: $EXP1 ($EXP1_FILE)"
echo "  Exp2: $EXP2 ($EXP2_FILE)"
echo ""

# ============================================================================
# Run analysis script
# ============================================================================

uv run "experiments/_scripts/analysis/$ANALYSIS_SCRIPT" \
        --exp1_file "$EXP1_FILE" \
        --exp2_file "$EXP2_FILE" \
        --exp1_name "$EXP1" \
        --exp2_name "$EXP2" \
        --model_names "${MODEL_NAMES[@]}"
