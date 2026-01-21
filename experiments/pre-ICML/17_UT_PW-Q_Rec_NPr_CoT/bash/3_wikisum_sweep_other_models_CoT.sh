#!/bin/bash
# Sweep experiment: Compare models against each other

uv run experiments/scripts/run_experiment_sweep.py \
    --model_names gpt-oss-20b-thinking gpt-oss-120b-thinking \
                sonnet-3.7-thinking \
                grok-3-mini-thinking \
                ll-3.3-70b-dsR1-thinking \
                qwen-3.0-80b-thinking qwen-3.0-235b-thinking \
                deepseek-r1-thinking \
                kimi-k2-thinking \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/17_UT_PW-Q_Rec_NPr_CoT/config.yaml \
    --max-tasks 24 --batch
