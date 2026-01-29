#!/bin/bash
# Shared configuration for analysis scripts in this directory
# This file is sourced by all analysis scripts to get dataset subsets and model names

# Dataset subsets to combine (relative to data/results/{dataset_name}/)
# Each subset adds more datapoints to the same evaluations
DATASET_SUBSETS=(
    "wikisum/training_set_1-20+test_set_1-30"
    "sharegpt/english_26+english2_74"
    "pku_saferlhf/mismatch_1-20+test_mismatch_1-20+test-mismatch_10_100-200"
    "bigcodebench/instruct_1-50"
)

# Model names to include in analysis
# Use -set {set_name} for model sets, or list individual models
MODEL_NAMES=(
    "-set"
    "dr"
)
