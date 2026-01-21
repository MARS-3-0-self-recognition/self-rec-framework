uv run experiments/scripts/run_experiment.py \
    --dataset_path_control=data/input/pku_saferlhf/debug/gemini-2.0-flash/data.json \
    --dataset_path_treatment=data/input/pku_saferlhf/debug/gemini-2.0-flash_typos_S2/data.json \
    --experiment_config=experiments/13_UT_PW-Q_Pref-N_NPr/config.yaml \
    --model_name=gemini-2.0-flash
