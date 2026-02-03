# Generating figures for `copy_figures.sh`

`scripts/generate_figures.sh` runs the analysis scripts that produce the figures listed in `scripts/copy_figures.sh`. It parses the `FIGURES` array from `copy_figures.sh`, maps each `(experiment_path, filename)` to a bash script, deduplicates, and runs those scripts in order. Outputs go under `data/analysis/_aggregated_data/<experiment>/<YYYYMMDD_HHMMSS>/`. Then `scripts/copy_figures.sh` copies the latest run of each figure into `AIWILD_figures/` with the names used in the paper.

Below: which **script** produces which **figure filename** (and thus which **copy_figures.sh** output), and **which commands each bash file runs**.

---

## Scripts called when each bash file runs

When the bash scripts are run, they invoke the following Python scripts (from project root, via `uv run`).

### Inter-dataset analysis (single-experiment)

**`00a-performance_aggregate.sh`**

- **Calls:** `experiments/_scripts/analysis/aggregate_performance_data.py`
- **Arguments:** `--performance_files` (from `config.sh` DATASET_SUBSETS), `--dataset_names`, `--model_names`, `--output_dir` (new timestamped dir under `data/analysis/_aggregated_data/<EXP>/`).
- **Creates (AIWILD_figures):** None. Produces aggregated CSVs used by 00b, 02, 03. Must be run before those scripts.

**`00b-plot_performance_aggregate.sh`**

- **Calls:** `experiments/_scripts/analysis/plot_aggregated_performance.py`
- **Arguments:** `--aggregated_file` (latest `.../aggregated_performance.csv` in that experiment’s aggregated dir).
- **Creates (AIWILD_figures):** `figure_1a_ICML_07_aggregated_performance_grouped.png`, `figure_1b_ICML_08_aggregated_performance_grouped.png` (one per experiment run).

**`02-performance_vs_size.sh`**

- **Calls:** `experiments/_scripts/analysis/performance_vs_size.py`
- **Arguments:** `--aggregated_file` (latest `aggregated_performance.csv`), `--model_names` (from `config.sh`).
- **Creates (AIWILD_figures):** `figure_2c_ICML_07_performance_vs_arena_ranking.png`, `figure_2d_ICML_08_performance_vs_arena_ranking.png` (one per experiment run).

**`03-rank-distance.sh`**

- **Calls:** `experiments/_scripts/analysis/rank_distance.py`
- **Arguments:** `--accuracy_files` (one `accuracy_pivot.csv` per DATASET_SUBSETS entry under `data/analysis/<subset>/<EXP>/recognition_accuracy/`), `--output_dir` (latest timestamped aggregated dir), `--model_names` (if set in config). For IND experiments (e.g. ICML_08), the bash script also passes `--exclude_self`.
- **Creates (AIWILD_figures):** `figure_2a_ICML_07_rank_distance_grouped_bar_chart.png`, `figure_2b_ICML_08_rank_distance_grouped_bar_chart.png`, `figure_2e_ICML_07_rank_distance_aggregated.png`, `figure_2f_ICML_08_rank_distance_adjusted.png`, `figure_3a_ICML_07_rank_distance_filtered_evaluator_rank.png`, `figure_3b_ICML_08_rank_distance_filtered_evaluator_rank.png`, `figure_3c_ICML_07_rank_distance_filtered_evaluator_rank_positive.png`, `figure_3d_ICML_08_rank_distance_filtered_evaluator_rank_positive.png` (subset per experiment run).

### Comparison (two-experiment)

**`experiments/comparisons/ICML/<EXP1>-vs-<EXP2>/00-performance_contrast.sh`**

- **Calls:** `experiments/_scripts/analysis/experiment_contrast.py`
- **Arguments:** `--exp1_file`, `--exp2_file` (latest `aggregated_performance.csv` for each experiment), `--exp1_name`, `--exp2_name`, `--model_names`. Optional `--reasoning_vs_instruct` if `REASONING_VS_INSTRUCT` is set in that comparison’s `config.sh`.
- **Creates (AIWILD_figures):** Depends on which comparison is run: `figure_1c_ICML_07_vs_ICML_08_performance_contrast_grouped.png` (ICML_07-vs-ICML_08), `figure_4a_ICML_01_vs_ICML_05_performance_scatter.png` (ICML_01-vs-ICML_05), `figure_4b_ICML_02_vs_ICML_06_performance_scatter.png` (ICML_02-vs-ICML_06).
