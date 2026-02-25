#!/bin/bash
# Shared configuration for sweep-dr scripts
# This file is sourced by all sweep scripts to get common arguments

# Model names to use in the sweep
# Can be individual models or model sets (e.g., "-set dr")
# Format: space-separated list
MODEL_NAMES=(
    "-set"
    "eval_cot-r"
)

# Generator models to use in the sweep
# Can be individual models or model sets (e.g., "-set dr")
# Format: space-separated list
# Leave empty to default to MODEL_NAMES
GENERATOR_MODELS=(
    "-set"
    "eval_cot-r_and_dr"
)

# Treatment type for the experiment
# Options: "other_models", "caps", "typos"
TREATMENT_TYPE="other_models"

# Maximum number of tasks to run in parallel
MAX_TASKS=20

# Skip confirmation prompt (yes flag)
# Set to "true" to enable -y flag, "false" to disable
SKIP_CONFIRMATION="true"

# Batch mode configuration
# Options:
#   "false" or empty - no batch mode
#   "true" - enable batch mode with default config
#   integer (e.g., "1000") - batch size
#   path (e.g., "config.yaml") - path to batch config file
BATCH_MODE="false"
