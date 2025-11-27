# Report Figures

This directory contains generated figures for the SGTR AISI Challenge Fund Performance Report #1.

## Generating Figures

The figures in this directory are generated automatically using the `generate_report_figures.py` script. To regenerate all figures:

### Option 1: Using the Bash Script

```bash
./experiments/r0/bash/report.sh
```

### Option 2: Running the Python Script Directly

```bash
uv run experiments/scripts/generate_report_figures.py \
    --recognition_pivot data/analysis/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr/accuracy_pivot.csv \
    --preference_pivot data/analysis/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr/accuracy_pivot.csv \
    --preference_results data/results/pku_saferlhf/mismatch_1-20/14_UT_PW-Q_Pref-Q_NPr \
    --unprimed_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \
    --primed_results data/results/pku_saferlhf/mismatch_1-20/12_UT_PW-Q_Rec_Pr \
    --alignment_results data/results/pku_saferlhf/mismatch_1-20/11_UT_PW-Q_Rec_NPr \
    --summary_results data/results/wikisum/training_set_1-20/11_UT_PW-Q_Rec_NPr
```

## Figure Descriptions

- **figure_1_color_categorization.png**: Recognition vs Preference comparison with 4-color categorization
- **figure_2_agreement_heatmap.png**: Preference agreement heatmap between models
- **figure_3_evaluator_agreement_performance.png**: Evaluator performance on preference agreement
- **figure_4_priming_impact.png**: Impact of task priming on self-recognition
- **figure_5_semantic_variance.png**: Comparison of recognition accuracy across semantic variance (alignment vs summarization tasks)

## Note

PNG files in this directory are excluded from version control (see `.gitignore`). They should be regenerated as needed rather than committed to the repository.
