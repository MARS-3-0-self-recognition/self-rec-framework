# 2T Experiments - Quick Start Guide

## TL;DR

```bash
# 1. Use an existing config or create your own
# Configs are in: protocols/pairwise/config/experiments/

# 2. Run the experiment
uv run python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml

# 3. Check results
# Results are in ./logs/ directory
```

## Minimal Example

### 1. Prepare Data (5 minutes)

Create this structure:
```
data/wikisum/
├── articles.json                           # {"uuid1": "article text", ...}
├── claude-3-5-sonnet/
│   └── control_summaries.json             # {"uuid1": "summary text", ...}
└── gpt-4/
    └── typo_summaries.json                # {"uuid1": "summary text", ...}
```

### 2. Use or Create Config (2 minutes)

**Option A: Use existing config**
```bash
# Just use one of the pre-made configs
protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

**Option B: Create custom config**

`protocols/pairwise/config/experiments/my_experiment.yaml`:
```yaml
experiment_type: "two_turn"
paradigm: "rec"
prompt_file: "prompts/two_turn_prompts.yaml"
prompt_set: true

model: "anthropic/claude-3-5-sonnet-20241022"
model_name: "claude-3-5-sonnet"
alternative_model_name: "gpt-4"
dataset_name: "wikisum"
model_generation_string: "control"
alternative_model_generation_string: "typo"
log_dir: "./logs"
```

### 3. Run (1 command)

```bash
uv run python scripts/run_2t_experiment.py --config protocols/pairwise/config/experiments/my_experiment.yaml
```

Done! Results in `./logs/`

## Running Multiple Experiments

```bash
# Option 1: Run all configs in default directory
python scripts/run_all_2t_experiments.py --parallel --workers 4

# Option 2: Create custom batch directory
mkdir -p protocols/pairwise/config/experiments/my_batch
# Add: experiment1.yaml, experiment2.yaml, ...

# Run all in parallel
python scripts/run_all_2t_experiments.py \
    --config-dir protocols/pairwise/config/experiments/my_batch \
    --parallel \
    --workers 4
```

## Common Tasks

### Test with Fewer Samples
Add to config:
```yaml
max_samples: 10  # Only run 10 samples
```

### Use Deterministic Generation
Change task name:
```yaml
task_name: "two_turn_summary_recognition_deterministic"  # temp=0.0
```

### Change Prompts
Edit the prompt file (`protocols/pairwise/config/prompts/two_turn_prompts.yaml`):
```yaml
system: "Your custom system prompt..."
user: "Your custom user prompt with {passage} placeholder..."
rec_detection: "Your custom detection question..."
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Can't find config | Check path is relative to where you run the script |
| Can't find data | Verify `data/{dataset_name}/...` structure |
| API errors | Check `max_connections` in config, try reducing |
| Out of memory | Reduce `max_samples` or run sequentially |

## What Gets Created

```
logs/
└── {timestamp}/
    ├── eval_log.json          # Full experiment results
    ├── results.json           # Aggregated metrics
    └── samples.json           # Individual sample results
```

## Next Steps

- See `scripts/README.md` for full documentation
- See `INTEGRATION_SUMMARY.md` for technical details
- Check example configs in `configs/`
