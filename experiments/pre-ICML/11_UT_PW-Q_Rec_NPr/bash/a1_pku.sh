#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/analyze_pairwise_results.py \
        --results_dir data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr
