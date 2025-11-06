uv run experiments/scripts/run_experiment.py \
    --dataset_path_control=data/input/wikisum/debug/input/haiku-3-5/data.json \
    --dataset_path_treatment=data/input/wikisum/debug/input/haiku-3-5_typos_S2/data.json \
    --experiment_config=experiments/10_UT_PW-C_Rec_NPr/config.yaml \
    --model_name=haiku-3-5
