#!/bin/bash
# Shared configuration for sweep-dr scripts
# This file is sourced by all sweep scripts to get common arguments

# Read MODEL_NAMES / GENERATOR_MODELS from the root experiment config.yaml —
# the single source of truth. Walk up the tree to find it (same pattern as the
# COLM/ICML production sweeps).
_CONFIG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
while [[ ! -f "$_CONFIG_DIR/config.yaml" ]] && [[ "$_CONFIG_DIR" != "/" ]]; do
    _CONFIG_DIR="$(dirname "$_CONFIG_DIR")"
done
_ROOT_CONFIG="$_CONFIG_DIR/config.yaml"

MODEL_NAMES=($(python3 -c "
import yaml
with open('$_ROOT_CONFIG') as f:
    config = yaml.safe_load(f)
for item in config.get('model_names', []):
    print(item)
"))

GENERATOR_MODELS=($(python3 -c "
import yaml
with open('$_ROOT_CONFIG') as f:
    config = yaml.safe_load(f)
for item in config.get('generator_models', []):
    print(item)
"))

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
