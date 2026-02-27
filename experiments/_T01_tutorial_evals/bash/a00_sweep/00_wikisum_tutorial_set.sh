#!/bin/bash
# Sweep experiment: Compare models against each other

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ============================================================================
# Load configuration from shared config file
# ============================================================================

CONFIG_FILE="$SCRIPT_DIR/config.sh"
if [[ ! -f "$CONFIG_FILE" ]]; then
    echo "Error: Configuration file not found: $CONFIG_FILE"
    echo "Please create config.sh in the same directory with sweep configuration."
    exit 1
fi

# Source the configuration file
source "$CONFIG_FILE"

# ============================================================================
# Build command arguments from config
# ============================================================================

# Build model_names argument
MODEL_NAMES_ARG=()
for model in "${MODEL_NAMES[@]}"; do
    MODEL_NAMES_ARG+=("$model")
done

# Build generator_models argument
GENERATOR_MODELS_ARG=()
if [ ${#GENERATOR_MODELS[@]} -gt 0 ]; then
    GENERATOR_MODELS_ARG+=("--generator_models")
    for generator_model in "${GENERATOR_MODELS[@]}"; do
        GENERATOR_MODELS_ARG+=("$generator_model")
    done
fi

# Build batch argument (if enabled)
BATCH_ARG=()
if [[ "$BATCH_MODE" != "false" && -n "$BATCH_MODE" ]]; then
    if [[ "$BATCH_MODE" == "true" ]]; then
        BATCH_ARG=("--batch")
    else
        # Could be an integer or a path
        BATCH_ARG=("--batch" "$BATCH_MODE")
    fi
fi

# Build yes flag (if enabled)
YES_ARG=()
if [[ "$SKIP_CONFIRMATION" == "true" ]]; then
    YES_ARG=("-y")
fi

# ============================================================================
# Run sweep experiment
# ============================================================================

uv run experiments/_scripts/eval/run_experiment_sweep.py \
    --model_names "${MODEL_NAMES_ARG[@]}" \
    "${GENERATOR_MODELS_ARG[@]}" \
    --treatment_type "$TREATMENT_TYPE" \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config "$SCRIPT_DIR/../../config.yaml" \
    --max-tasks "$MAX_TASKS" \
    "${BATCH_ARG[@]}" \
    "${YES_ARG[@]}"
