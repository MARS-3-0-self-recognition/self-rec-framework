#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/analyze_pairwise_results.py \
        --results_dir data/results/wikisum/training_set_1-20/18_UT_PW-Q_Rec_NPr_FA \
        --config_path experiments/18_UT_PW-Q_Rec_NPr_FA/config.yaml
