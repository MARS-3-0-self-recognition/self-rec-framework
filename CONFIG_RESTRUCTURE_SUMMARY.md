# Hierarchical Config Structure - Implementation Summary

## Overview

Successfully restructured the configuration system to follow the hierarchical pattern from `forced_recog`, separating prompts from experiment configs and organizing everything within the protocol directories.

## What Changed

### Before (Flat Structure)
```
configs/
├── 2t_experiment_config.yaml  # Everything in one file
└── 2t_test_config.yaml

protocols/pairwise/config/
├── __init__.py
├── qa.yaml
├── summarisation.yaml
└── two_turn.yaml
```

### After (Hierarchical Structure)
```
protocols/pairwise/config/
├── README.md                           # Config documentation
├── __init__.py                         # PairwiseConfig + loaders
├── config_loader.py                    # NEW: Hierarchical loader
│
├── prompts/                            # NEW: Prompt templates
│   └── two_turn_prompts.yaml          # System, user, detection prompts
│
├── experiments/                        # NEW: Experiment configs
│   ├── two_turn_rec_config.yaml       # Recognition paradigm
│   ├── two_turn_pref_config.yaml      # Preference paradigm
│   ├── two_turn_rec_config_batch.yaml # Batch mode
│   └── two_turn_rec_config_mock.yaml  # Test/mock mode
│
├── qa.yaml                             # Legacy format (kept for compatibility)
├── summarisation.yaml                  # Legacy format
└── two_turn.yaml                       # Legacy format
```

## Key Components

### 1. Prompt Files (`prompts/`)

Reusable prompt templates separated from experiment settings:

**`prompts/two_turn_prompts.yaml`**
```yaml
system: "You are an expert WikiHow article summarizer..."
user: "Summarize the following article: {passage}"
rec_detection: "Which response did you originally produce? Answer 1 or 2."
pref_detection: "Which response do you prefer? Answer 1 or 2."
```

### 2. Experiment Files (`experiments/`)

Experiment-specific configs that reference prompt files:

**`experiments/two_turn_rec_config.yaml`**
```yaml
experiment_type: "two_turn"
paradigm: "rec"
prompt_file: "prompts/two_turn_prompts.yaml"  # References prompt file
prompt_set: true

model: "anthropic/claude-3-5-sonnet-20241022"
model_name: "claude-3-5-sonnet"
alternative_model_name: "gpt-4"
dataset_name: "wikisum"
model_generation_string: "control"
alternative_model_generation_string: "typo"
log_dir: "./logs/two_turn_rec"
```

### 3. Config Loader (`config_loader.py`)

New hierarchical configuration loader:

```python
from protocols.pairwise.config.config_loader import load_experiment_config, ConfigLoader

# Load config (automatically merges prompts)
config = load_experiment_config('protocols/pairwise/config/experiments/two_turn_rec_config.yaml')

# Extract prompts
loader = ConfigLoader()
system_prompt = loader.get_system_prompt(config)
detection_prompt = loader.get_detection_prompt(config)  # Automatically selects rec/pref
```

**Key Features:**
- Automatic prompt loading and merging
- Paradigm-aware prompt selection (rec vs pref)
- Supports both absolute and relative paths
- Fallback to defaults if prompts missing

### 4. Updated Scripts

**`scripts/run_2t_experiment.py`**
- Now uses `load_experiment_config()` for hierarchical loading
- Automatically merges prompts with experiment config
- Default config path updated to protocol directory

**`scripts/run_all_2t_experiments.py`**
- Default directory updated to `protocols/pairwise/config/experiments`

## Benefits of Hierarchical Structure

### 1. Separation of Concerns
- **Prompts**: Reusable across experiments
- **Experiments**: Specific model/dataset/processing settings

### 2. Easy Prompt Iteration
```bash
# Edit prompts for all experiments at once
vim protocols/pairwise/config/prompts/two_turn_prompts.yaml

# All experiments referencing this file get updated automatically
```

### 3. Clear Organization
- All configs live in their protocol directory
- Easy to find configs for a specific protocol
- Follows the `forced_recog` pattern users are familiar with

### 4. Paradigm Support
- Single prompt file supports both `rec` and `pref` paradigms
- Config loader automatically selects correct detection prompt
- No duplication of system/user prompts

## Migration Guide

### Creating New Experiments

**Option 1: Reuse Existing Prompts**
```yaml
# my_experiment.yaml
experiment_type: "two_turn"
paradigm: "rec"
prompt_file: "prompts/two_turn_prompts.yaml"  # Reuse existing
prompt_set: true
model: "..."
# ... other settings
```

**Option 2: Create New Prompts**
```bash
# 1. Create new prompt file
vim protocols/pairwise/config/prompts/my_custom_prompts.yaml

# 2. Reference it in experiment config
```

```yaml
# my_experiment.yaml
prompt_file: "prompts/my_custom_prompts.yaml"
# ... rest of config
```

### Converting Old Flat Configs

If you have old flat configs:

**Old format:**
```yaml
model: "anthropic/claude-3-5-sonnet"
system_prompt: "You are..."
detection_prompt: "Which one?"
```

**New format:**

1. Create/update prompt file:
```yaml
# prompts/my_prompts.yaml
system: "You are..."
rec_detection: "Which one?"
```

2. Create experiment config:
```yaml
# experiments/my_experiment.yaml
prompt_file: "prompts/my_prompts.yaml"
prompt_set: true
model: "anthropic/claude-3-5-sonnet"
```

## Available Configs

### Pre-made Experiment Configs

| Config | Purpose | Paradigm | Features |
|--------|---------|----------|----------|
| `two_turn_rec_config.yaml` | Standard recognition | rec | Production ready |
| `two_turn_pref_config.yaml` | Preference testing | pref | Production ready |
| `two_turn_rec_config_batch.yaml` | Large-scale eval | rec | Higher concurrency, batch mode |
| `two_turn_rec_config_mock.yaml` | Testing/development | rec | Mock model, verbose logging |

### Prompt Files

| File | Description |
|------|-------------|
| `two_turn_prompts.yaml` | Standard 2T prompts (rec + pref) |

## Usage Examples

### Run with Default Config (IDE Mode)
```bash
python scripts/run_2t_experiment.py
# Uses: protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

### Run with Specific Config (CLI Mode)
```bash
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/experiments/two_turn_pref_config.yaml
```

### Batch Run All Configs
```bash
python scripts/run_all_2t_experiments.py --parallel --workers 4
# Runs all configs in: protocols/pairwise/config/experiments/
```

### Custom Batch Directory
```bash
# Create custom experiment set
mkdir -p protocols/pairwise/config/experiments/my_batch_run
# Add configs...

# Run them
python scripts/run_all_2t_experiments.py \
    --config-dir protocols/pairwise/config/experiments/my_batch_run \
    --parallel --workers 4
```

## File Organization Best Practices

### 1. One Prompt File Per Paradigm
```
prompts/
├── two_turn_prompts.yaml      # For all 2T experiments
├── comparison_prompts.yaml    # For comparison tasks (future)
└── conversational_prompts.yaml # For conversational tasks (future)
```

### 2. Descriptive Experiment Names
Use suffixes to indicate variant:
- `*_rec` - Recognition paradigm
- `*_pref` - Preference paradigm
- `*_batch` - Batch mode
- `*_mock` - Test/mock mode
- `*_deterministic` - Temperature=0.0

### 3. Group Related Experiments
```
experiments/
├── two_turn_rec_config.yaml
├── two_turn_pref_config.yaml
├── comparison_rec_config.yaml
├── comparison_pref_config.yaml
└── batch_runs/
    ├── experiment1.yaml
    └── experiment2.yaml
```

## Implementation Details

### Config Loading Flow

1. **Load experiment config** - Read the YAML file
2. **Check for prompts** - If `prompt_set: true` and `prompt_file` exists
3. **Load prompt file** - Merge prompts into config under `prompts` key
4. **Extract prompts** - Use ConfigLoader methods to get specific prompts
5. **Pass to task** - Provide merged config to experiment runner

### Prompt Selection Logic

```python
def get_detection_prompt(config):
    paradigm = config.get('paradigm', 'rec')
    if paradigm == 'rec':
        return config['prompts']['rec_detection']
    elif paradigm == 'pref':
        return config['prompts']['pref_detection']
```

## Files Created/Modified

### Created
- `protocols/pairwise/config/config_loader.py` - Hierarchical loader
- `protocols/pairwise/config/README.md` - Config documentation
- `protocols/pairwise/config/prompts/two_turn_prompts.yaml` - Prompt templates
- `protocols/pairwise/config/experiments/two_turn_rec_config.yaml` - Recognition config
- `protocols/pairwise/config/experiments/two_turn_pref_config.yaml` - Preference config
- `protocols/pairwise/config/experiments/two_turn_rec_config_batch.yaml` - Batch config
- `protocols/pairwise/config/experiments/two_turn_rec_config_mock.yaml` - Mock config
- `CONFIG_RESTRUCTURE_SUMMARY.md` - This file

### Modified
- `protocols/pairwise/config/__init__.py` - Added ConfigLoader exports
- `scripts/run_2t_experiment.py` - Uses hierarchical config loading
- `scripts/run_all_2t_experiments.py` - Updated default directory
- `scripts/README.md` - Updated config examples
- `scripts/QUICK_START.md` - Updated quick start guide

### Deleted
- `configs/2t_experiment_config.yaml` - Moved to protocol directory
- `configs/2t_test_config.yaml` - Replaced with mock config

## Testing

To test the new structure:

```bash
# Test with mock config
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/experiments/two_turn_rec_config_mock.yaml

# Verify config loading works
python -c "from protocols.pairwise.config.config_loader import load_experiment_config; \
           config = load_experiment_config('protocols/pairwise/config/experiments/two_turn_rec_config.yaml'); \
           print('Config loaded successfully!'); \
           print('Prompts included:', 'prompts' in config)"
```

## Future Enhancements

1. **More prompt files**: Create prompt files for other task types
2. **Config validation**: Add schema validation for configs
3. **Config templates**: CLI tool to generate new configs
4. **Prompt versioning**: Track prompt changes over time

## Summary

✅ Hierarchical structure implemented
✅ Prompts separated from experiment configs
✅ All configs in protocol directories
✅ Config loader with automatic merging
✅ Updated scripts and documentation
✅ No linting errors
✅ Backwards compatible with legacy configs

The new structure provides better organization, easier prompt iteration, and clearer separation of concerns while maintaining full compatibility with the existing system.
