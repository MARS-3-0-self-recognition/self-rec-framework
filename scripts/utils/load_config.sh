#!/bin/bash
# Shared config loader — SOURCE this (do not execute), passing an experiment
# config.yaml path as $1. It populates the variables the bash sweep / analysis /
# comparison scripts use:
#   MODEL_NAMES            (array)  -> --model_names ...
#   GENERATOR_MODELS_ARG   (array)  -> --generator_models ...   (empty if none)
#   TREATMENT_TYPE, MAX_TASKS
#   BATCH_ARG              (array)  -> --batch [value]           (empty if disabled)
#   YES_ARG                (array)  -> -y                        (empty unless skip)
#   DATASET_SUBSETS        (array)  -> per-dataset analysis subsets ("tutorial_set")
#   INTER_DATASET_SUBSETS  (array)  -> {dataset}/{subset} for inter-dataset analysis
#   FIGURES_TO_PRODUCE     (array)  -> figure basenames listed in config (may be empty)
#   FIGURES_ARG            (array)  -> --figures fig1,fig2,...  (empty if list absent)
# Values come from config.yaml: the `model` / `sweep` / `analysis` sections.
# INTER_DATASET_SUBSETS is the cross product of datasets x dataset_subsets.
# When figures_to_produce is empty the analysis scripts emit every figure
# (FIGURES_ARG stays empty -> the Python default of "all").
# Lives in scripts/utils/ so scripts/utils/run.sh never executes it.

_LSC_CONFIG="${1:?load_config.sh: config.yaml path required as \$1}"
if [[ ! -f "$_LSC_CONFIG" ]]; then
    echo "Error: config.yaml not found: $_LSC_CONFIG" >&2
    return 1 2>/dev/null || exit 1
fi

eval "$(python3 - "$_LSC_CONFIG" <<'PY'
import yaml, shlex, sys
c = yaml.safe_load(open(sys.argv[1])) or {}
q = shlex.quote
def arr(name, items):
    print(name + '=(' + ' '.join(q(str(x)) for x in items) + ')')
arr('MODEL_NAMES', c.get('model_names') or [])
arr('GENERATOR_MODELS', c.get('generator_models') or [])
# Exported (not just set) so analysis Python scripts inherit it via `uv run`.
# Authoritative IND vs PW signal for the analysis (e.g. evaluator_performance,
# rank_distance) instead of guessing from the experiment folder name.
print('export SRF_EXPERIMENT_FORMAT=' + q(str(c.get('format', ''))))
print('TREATMENT_TYPE=' + q(str(c.get('treatment_type', 'other_models'))))
print('MAX_TASKS=' + q(str(c.get('max_tasks', 15))))
print('BATCH_MODE=' + q(str(c.get('batch', False)).lower()))
print('SKIP_CONFIRMATION=' + q(str(c.get('skip_confirmation', True)).lower()))
subsets = c.get('dataset_subsets') or []
datasets = c.get('datasets') or []
arr('DATASET_SUBSETS', subsets)
arr('INTER_DATASET_SUBSETS', [f'{d}/{s}' for d in datasets for s in subsets])
arr('FIGURES_TO_PRODUCE', c.get('figures_to_produce') or [])
PY
)"

# Build the --figures argument from figures_to_produce. Empty list => omit the
# flag so each analysis script falls back to its "all figures" default.
FIGURES_ARG=()
if [ ${#FIGURES_TO_PRODUCE[@]} -gt 0 ]; then
    _figs_csv=$(IFS=,; printf '%s' "${FIGURES_TO_PRODUCE[*]}")
    FIGURES_ARG=("--figures" "$_figs_csv")
fi

GENERATOR_MODELS_ARG=()
if [ ${#GENERATOR_MODELS[@]} -gt 0 ]; then
    GENERATOR_MODELS_ARG=("--generator_models" "${GENERATOR_MODELS[@]}")
fi

BATCH_ARG=()
if [[ "$BATCH_MODE" != "false" && -n "$BATCH_MODE" ]]; then
    if [[ "$BATCH_MODE" == "true" ]]; then
        BATCH_ARG=("--batch")
    else
        BATCH_ARG=("--batch" "$BATCH_MODE")
    fi
fi

YES_ARG=()
if [[ "$SKIP_CONFIRMATION" == "true" ]]; then
    YES_ARG=("-y")
fi
