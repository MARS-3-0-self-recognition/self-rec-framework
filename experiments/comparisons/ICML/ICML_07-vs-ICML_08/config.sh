#!/bin/bash
# Shared configuration for analysis scripts in this directory
# This file is sourced by all analysis scripts to get dataset subsets and model names

# Dataset subsets to combine (relative to data/results/{dataset_name}/)
# Each subset adds more datapoints to the same evaluations
# Model names to include in analysis
# Use -set {set_name} for model sets, or list individual models
MODEL_NAMES=(
    "-set"
    "eval_cot-r_and_dr"
)

# Pair exp1 models ending with -thinking to exp2 instruct names (one row per base model)
REASONING_VS_INSTRUCT=0
