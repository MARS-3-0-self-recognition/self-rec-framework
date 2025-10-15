uv run experiments/0_AT_vs_UT/run_experiment.py \
    --generation_string=simple_config \
    --dataset_name=cnn_debug \
    --pairwise_config_string=summarisation \
    --model_names 3-5-haiku 3-5-sonnet ll-3-1-8b #ll-3-1-70b deepseek-v3

# Note this needs experiment_id but does not need pairwise_config_string
# (this should be fixed for dataset - some invariance still to remove)
uv run experiments/0_AT_vs_UT/analysis.py \
    --experiment_id=0_testing \
    --generation_string=simple_config \
    --dataset_name=cnn_debug \
    --model_names 3-5-haiku 3-5-sonnet ll-3-1-8b #ll-3-1-70b deepseek-v3
