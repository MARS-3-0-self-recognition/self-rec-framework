#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/analyze_pairwise_results.py \
        --results_dir data/results/wikisum/training_set_1-20/13_UT_PW-Q_Pref-N_NPr
