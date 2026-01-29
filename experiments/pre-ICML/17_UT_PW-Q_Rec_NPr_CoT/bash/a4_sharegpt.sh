#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/analyze_pairwise_results.py \
        --results_dir data/results/sharegpt/english_26/17_UT_PW-Q_Rec_NPr_CoT \
        --config_path experiments/17_UT_PW-Q_Rec_NPr_CoT/config.yaml
