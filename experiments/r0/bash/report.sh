#!/bin/bash

uv run experiments/scripts/generate_report_figures.py \
        --recognition_pivot data/analysis/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr/accuracy_pivot.csv \
        --preference_pivot data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr/accuracy_pivot.csv \
        --preference_results data/results/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr \
        --unprimed_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \
        --primed_results data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr \
        --alignment_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \
        --summary_results data/results/wikisum/training_set_1-20/11_UT_PW-Q_Rec_NPr
