# 2T Experiment Scripts

This directory contains scripts for running Two-Turn (2T) self-recognition experiments using the inspect-ai framework. These scripts integrate the experimental design from `forced_recog` with the new `protocols/pairwise` structure.

## Overview

The 2T experiments test whether models can identify their own outputs when presented in a conversational context with two turns. This is analogous to the AT_2T (Assist Tag 2-Turn) experiments from the forced_recog pipeline.

### Key Differences from forced_recog

| Aspect | forced_recog (Old) | protocols/pairwise (New) |
|--------|-------------------|------------------------|
| Framework | Custom homemade tools | inspect-ai |
| Model wrappers | Custom (anthropic.py, etc.) | inspect-ai built-in |
| Data format | CSV (control/treatment) | JSON (model outputs) |
| Conversation handling | Manual formatting | inspect-ai ChatMessage |
| Logprobs | Manual extraction | Built-in with GenerateConfig |
| Scoring | Custom CSV output | inspect-ai scorer framework |

## Files

- **`run_2t_experiment.py`**: Main script to run a single 2T experiment
- **`run_all_2t_experiments.py`**: Batch runner for multiple experiments
- **`README.md`**: This file

## Quick Start

### 1. Prepare Your Data

Data should be organized as follows:

```
data/
└── {dataset_name}/
    ├── articles.json                                    # Source articles
    ├── {model_name}/
    │   └── {generation_string}_summaries.json          # Model outputs
    └── {alternative_model_name}/
        └── {alternative_generation_string}_summaries.json  # Alternative model outputs
```

Example:
```
data/
└── wikisum/
    ├── articles.json
    ├── claude-3-5-sonnet/
    │   └── control_summaries.json
    └── gpt-4/
        └── typo_summaries.json
```

**JSON Format:**
- `articles.json`: `{"uuid1": "article text...", "uuid2": "article text...", ...}`
- `*_summaries.json`: `{"uuid1": "summary text...", "uuid2": "summary text...", ...}`

### 2. Create a Configuration File

Configs are stored in `protocols/pairwise/config/experiments/`.

You can use existing configs or create new ones:

**Option A: Use existing config**
```bash
# Recognition mode
protocols/pairwise/config/experiments/two_turn_rec_config.yaml

# Preference mode
protocols/pairwise/config/experiments/two_turn_pref_config.yaml

# Batch mode
protocols/pairwise/config/experiments/two_turn_rec_config_batch.yaml

# Mock/test mode
protocols/pairwise/config/experiments/two_turn_rec_config_mock.yaml
```

**Option B: Create custom config**

Create `protocols/pairwise/config/experiments/my_experiment.yaml`:

```yaml
# Experiment Type
experiment_type: "two_turn"
paradigm: "rec"  # or "pref"

# Prompt Configuration (references prompt file)
prompt_file: "prompts/two_turn_prompts.yaml"
prompt_set: true

# Model Configuration
model: "anthropic/claude-3-5-sonnet-20241022"
model_name: "claude-3-5-sonnet"
alternative_model_name: "gpt-4"

# Dataset Configuration
dataset_name: "wikisum"
model_generation_string: "control"
alternative_model_generation_string: "typo"

# Processing
max_samples: null
max_connections: 10
log_dir: "./logs/my_experiment"
```

**Prompts are separate!** They're in `protocols/pairwise/config/prompts/two_turn_prompts.yaml`

### 3. Run an Experiment

**From IDE (using hardcoded config path):**
```bash
python scripts/run_2t_experiment.py
```

**From CLI (specify config):**
```bash
python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

**Using uv:**
```bash
uv run python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

### 4. Run Multiple Experiments

To run multiple experiments, you can use the existing configs directory or create a custom one:

```bash
# Option 1: Run all configs in the default directory
python scripts/run_all_2t_experiments.py

# Option 2: Run configs from a custom directory
# First, create your custom directory
mkdir -p protocols/pairwise/config/experiments/batch_run

# Add multiple config files
# protocols/pairwise/config/experiments/batch_run/experiment1.yaml
# protocols/pairwise/config/experiments/batch_run/experiment2.yaml
# etc.

# Run all experiments sequentially
python scripts/run_all_2t_experiments.py --config-dir protocols/pairwise/config/experiments/batch_run

# Run in parallel with 4 workers
python scripts/run_all_2t_experiments.py --config-dir protocols/pairwise/config/experiments/batch_run --parallel --workers 4

# Dry run (see what would execute)
python scripts/run_all_2t_experiments.py --config-dir protocols/pairwise/config/experiments/batch_run --dry-run
```

## Available Tasks

The integration provides several task variants in `protocols/pairwise/tasks.py`:

1. **`two_turn_summary_recognition`**: Standard 2T task
2. **`two_turn_summary_recognition_deterministic`**: Temperature=0.0 variant

### Task Parameters

All tasks accept these parameters:
- `model_name`: Name of the model being evaluated (for data lookup)
- `alternative_model_name`: Name of the alternative model
- `dataset_name`: Dataset directory name under `data/`
- `model_generation_string`: Generation identifier for model (e.g., "control")
- `alternative_model_generation_string`: Generation identifier for alternative (e.g., "typo")
- `system_prompt`: (Optional) System prompt for the conversation

## How It Works

### Conversation Structure

The 2T task creates a conversation with the following structure:

```
[System Message]: "You are an expert WikiHow article summarizer..."

[User]: "Summarize the following article: {article_text}"
[Assistant]: {output1}  # Could be from model or alternative

[User]: "Summarize the following article: {article_text}"  # Same article
[Assistant]: {output2}  # Could be from model or alternative

[User]: "In the chat history, one of the two responses you provided was edited..."
```

The model must then respond with "1" or "2" to indicate which output it originally produced.

### Data Loading

The loader (`protocols/pairwise/data.py`) creates **two samples per UUID**:
1. Model output first (position 1) - correct answer is "1"
2. Alternative output first (position 1) - correct answer is "2"

This ensures both presentation orders are tested.

### Scoring

The `logprob_scorer` from `protocols/pairwise/scorer.py`:
- Extracts the model's choice ("1" or "2")
- Compares to the correct answer
- Reports accuracy metrics

## Configuration Options

### Required Parameters

- `model`: Full model identifier for inspect-ai (e.g., "anthropic/claude-3-5-sonnet-20241022")
- `model_name`: Short name for data directory (e.g., "claude-3-5-sonnet")
- `alternative_model_name`: Alternative model name (e.g., "gpt-4")
- `dataset_name`: Dataset directory name (e.g., "wikisum")

### Optional Parameters

- `task_name`: Task variant to use (default: "two_turn_summary_recognition")
- `model_generation_string`: Generation identifier (default: "control")
- `alternative_model_generation_string`: Alternative generation identifier (default: "treatment")
- `system_prompt`: Custom system prompt (uses default if not specified)
- `log_dir`: Directory for inspect-ai logs (default: "./logs")
- `max_samples`: Maximum samples to evaluate (default: null = all)
- `max_connections`: Max concurrent API connections (default: 10)

## Migration from forced_recog

If you have data from the forced_recog pipeline:

### Data Conversion

You'll need to convert from CSV to JSON format:

```python
import pandas as pd
import json

# Load control and treatment CSVs
control_df = pd.read_csv('control.csv')
treatment_df = pd.read_csv('treatment.csv')

# Convert to JSON format
articles = {row['uuid']: row['passage'] for _, row in control_df.iterrows()}
control_summaries = {row['uuid']: row['response'] for _, row in control_df.iterrows()}
treatment_summaries = {row['uuid']: row['response'] for _, row in treatment_df.iterrows()}

# Save
with open('data/wikisum/articles.json', 'w') as f:
    json.dump(articles, f, indent=2)

with open('data/wikisum/model_name/control_summaries.json', 'w') as f:
    json.dump(control_summaries, f, indent=2)

with open('data/wikisum/model_name/treatment_summaries.json', 'w') as f:
    json.dump(treatment_summaries, f, indent=2)
```

### Prompt Mapping

Old AT_2T prompts → New configuration:

| forced_recog | protocols/pairwise |
|--------------|-------------------|
| `system` | `system_prompt` parameter |
| `user` | `conversational_generation_prompt` in config |
| `rec_detection` | `conversational_verification_prompt` in config |

## Results

Results are stored by inspect-ai in the log directory:
- Location: `{log_dir}/` (default: `./logs/`)
- Format: JSON files with full experiment details
- Metrics: Automatically computed by inspect-ai

To analyze results, use inspect-ai's built-in tools or parse the JSON logs.

## Troubleshooting

### "Config file not found"
- Make sure you've created a config YAML file
- Check the path is correct (relative to where you run the script)

### "No config files found in directory"
- Ensure your config files have `.yaml` or `.yml` extension
- Check you're pointing to the correct directory

### "Data file not found"
- Verify data structure matches expected format
- Check model names and generation strings match directory/file names
- Ensure UUIDs match across articles and summaries

### Import errors
- Make sure you're in the project root directory
- Try using `uv run` instead of plain `python`

## Examples

See `configs/2t_experiment_config.yaml` for a complete example configuration.
