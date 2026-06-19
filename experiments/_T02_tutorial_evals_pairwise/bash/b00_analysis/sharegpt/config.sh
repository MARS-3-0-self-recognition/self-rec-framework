#!/bin/bash
# Shared configuration for analysis scripts in this directory
# This file is sourced by all analysis scripts to get dataset subsets and model names

# Dataset subsets to combine (relative to data/results/{dataset_name}/)
# Each subset adds more datapoints to the same evaluations
DATASET_SUBSETS=(
    "tutorial_set"
)

# Model names to include in analysis
# Use -set {set_name} for model sets, or list individual models
MODEL_NAMES=(
    "-set"
    "tutorial"
)
