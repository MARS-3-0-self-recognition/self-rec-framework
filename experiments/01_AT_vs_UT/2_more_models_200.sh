uv run experiments/0_AT_vs_UT/run_experiment.py \
    --generation_string=simple_config \
    --dataset_name=cnn200 \
    --pairwise_config_string=summarisation \
    --model_names ll-3-1-8b ll-3-1-70b qwen3-30b-a3b 3-5-haiku 3-5-sonnet #4-5 sonnet

# Note this needs experiment_id but does not need pairwise_config_string
# (this should be fixed for dataset - some invariance still to remove)
uv run experiments/0_AT_vs_UT/analysis.py \
    --experiment_id=2_more_models_200 \
    --generation_string=simple_config \
    --dataset_name=cnn200 \
    --model_names ll-3-1-8b ll-3-1-70b qwen3-30b-a3b 3-5-haiku 3-5-sonnet #4-5 sonnet
