# Configuration Structure Quick Reference

## Complete Directory Tree

```
protocols/pairwise/
├── config/
│   ├── README.md                               # Full config documentation
│   ├── CONFIG_STRUCTURE.md                     # This file (quick reference)
│   ├── __init__.py                             # PairwiseConfig class + loaders
│   ├── config_loader.py                        # ConfigLoader class
│   │
│   ├── prompts/                                # 🎯 PROMPT TEMPLATES
│   │   └── two_turn_prompts.yaml              # 2T: system, user, rec, pref
│   │
│   ├── experiments/                            # 🔬 EXPERIMENT CONFIGS
│   │   ├── two_turn_rec_config.yaml           # Recognition (production)
│   │   ├── two_turn_pref_config.yaml          # Preference (production)
│   │   ├── two_turn_rec_config_batch.yaml     # Batch mode
│   │   └── two_turn_rec_config_mock.yaml      # Testing/mock
│   │
│   ├── qa.yaml                                 # Legacy: QA task config
│   ├── summarisation.yaml                      # Legacy: Summary task config
│   └── two_turn.yaml                           # Legacy: 2T task config
│
├── data.py                                     # Data loading
├── scorer.py                                   # Scoring logic
├── task.py                                     # Task builders
└── tasks.py                                    # Task definitions
```

## Visual Flow: How Configs Work Together

```
┌─────────────────────────────────────────────────────────────────┐
│                    Experiment Config File                        │
│           experiments/two_turn_rec_config.yaml                   │
├─────────────────────────────────────────────────────────────────┤
│ experiment_type: "two_turn"                                     │
│ paradigm: "rec"                                                 │
│ prompt_file: "prompts/two_turn_prompts.yaml" ← References      │
│ prompt_set: true                                                │
│                                                                 │
│ model: "anthropic/claude-3-5-sonnet-20241022"                  │
│ model_name: "claude-3-5-sonnet"                                │
│ dataset_name: "wikisum"                                        │
│ ...                                                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            │ load_experiment_config()
                            ↓
                    ┌───────────────┐
                    │ ConfigLoader  │
                    │  merges with  │
                    │    prompts    │
                    └───────┬───────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                      Prompt File                                │
│             prompts/two_turn_prompts.yaml                       │
├─────────────────────────────────────────────────────────────────┤
│ system: "You are an expert WikiHow summarizer..."              │
│ user: "Summarize the following article: {passage}"             │
│ rec_detection: "Which response did you produce? 1 or 2?"       │
│ pref_detection: "Which response do you prefer? 1 or 2?"        │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│                     Merged Config                               │
│                  (returned to script)                           │
├─────────────────────────────────────────────────────────────────┤
│ experiment_type: "two_turn"                                     │
│ paradigm: "rec"                                                 │
│ model: "anthropic/claude-3-5-sonnet-20241022"                  │
│ ...                                                             │
│                                                                 │
│ prompts:  ← Added automatically                                │
│   system: "You are an expert WikiHow summarizer..."            │
│   user: "Summarize the following article: {passage}"           │
│   rec_detection: "Which response did you produce? 1 or 2?"     │
│   pref_detection: "Which response do you prefer? 1 or 2?"      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Usage Patterns

### Pattern 1: Use Existing Config
```bash
# Just run with a pre-made config
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

### Pattern 2: Create New Experiment (Same Prompts)
```yaml
# protocols/pairwise/config/experiments/my_new_experiment.yaml
experiment_type: "two_turn"
paradigm: "rec"
prompt_file: "prompts/two_turn_prompts.yaml"  # Reuse existing prompts
prompt_set: true

model: "different-model"
dataset_name: "different-dataset"
# ... customize other settings
```

### Pattern 3: Create New Prompts
```yaml
# Step 1: Create prompts/my_custom_prompts.yaml
system: "My custom system prompt..."
user: "My custom user prompt: {passage}"
rec_detection: "My custom detection prompt..."

# Step 2: Reference in experiment config
prompt_file: "prompts/my_custom_prompts.yaml"
```

## File Naming Conventions

### Prompt Files
- `{paradigm}_prompts.yaml` - e.g., `two_turn_prompts.yaml`
- Located in: `prompts/`

### Experiment Files
- `{paradigm}_{variant}_config.yaml` - e.g., `two_turn_rec_config.yaml`
- Variants: `rec`, `pref`, `batch`, `mock`
- Located in: `experiments/`

## Config Keys Reference

### Experiment Config Required Keys
```yaml
experiment_type: "two_turn"        # Type of task
paradigm: "rec"                    # or "pref"
task_name: "two_turn_summary_recognition"
model: "anthropic/..."             # Full model ID
model_name: "claude-3-5-sonnet"    # Short name for data dirs
alternative_model_name: "gpt-4"
dataset_name: "wikisum"
model_generation_string: "control"
alternative_model_generation_string: "typo"
```

### Prompt Config Keys
```yaml
system: "System prompt text..."
user: "User prompt with {passage} placeholder..."
rec_detection: "Recognition detection question..."
pref_detection: "Preference detection question..."
```

### Optional Experiment Keys
```yaml
max_samples: 100                   # Limit samples (null = all)
max_connections: 10                # Concurrent API calls
batch_size: 50                     # For batch configs
log_dir: "./logs/experiment_name"
logging:
  enabled: true
  level: "INFO"                    # DEBUG, INFO, WARNING, ERROR
show_sample_data: false
show_conversation_breakdown: true
```

## Paradigm Selection

The ConfigLoader automatically selects the right detection prompt:

| Paradigm | Detection Prompt Used |
|----------|----------------------|
| `rec` | `prompts['rec_detection']` |
| `pref` | `prompts['pref_detection']` |

Example:
```python
# Config has paradigm: "rec"
loader.get_detection_prompt(config)  # Returns rec_detection

# Config has paradigm: "pref"
loader.get_detection_prompt(config)  # Returns pref_detection
```

## Common Workflows

### Workflow 1: Test Before Production
```bash
# 1. Test with mock config
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/experiments/two_turn_rec_config_mock.yaml

# 2. If successful, run production config
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/experiments/two_turn_rec_config.yaml
```

### Workflow 2: Iterate on Prompts
```bash
# 1. Edit prompts
vim protocols/pairwise/config/prompts/two_turn_prompts.yaml

# 2. Test with small dataset
# (edit config to set max_samples: 5)
python scripts/run_2t_experiment.py --config ...

# 3. Run full experiment when satisfied
```

### Workflow 3: Batch Multiple Models
```bash
# 1. Create configs for each model
# experiments/model1_rec.yaml
# experiments/model2_rec.yaml
# experiments/model3_rec.yaml
# (all reference same prompt file)

# 2. Run all in parallel
python scripts/run_all_2t_experiments.py --parallel --workers 3
```

## Quick Reference Card

```
┌─────────────────────────────────────────────────────────────┐
│                  CONFIG QUICK REFERENCE                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Prompts:  protocols/pairwise/config/prompts/               │
│           ↳ two_turn_prompts.yaml                          │
│                                                             │
│ Experiments: protocols/pairwise/config/experiments/        │
│              ↳ two_turn_rec_config.yaml (production)       │
│              ↳ two_turn_pref_config.yaml (preference)      │
│              ↳ two_turn_rec_config_batch.yaml (batch)      │
│              ↳ two_turn_rec_config_mock.yaml (testing)     │
│                                                             │
│ Load Config:                                               │
│   from protocols.pairwise.config.config_loader import \    │
│       load_experiment_config                               │
│   config = load_experiment_config('path/to/config.yaml')   │
│                                                             │
│ Extract Prompts:                                           │
│   from protocols.pairwise.config.config_loader import \    │
│       ConfigLoader                                         │
│   loader = ConfigLoader()                                  │
│   system = loader.get_system_prompt(config)                │
│   detection = loader.get_detection_prompt(config)          │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## See Also

- `README.md` - Full configuration documentation
- `../../scripts/README.md` - How to run experiments
- `../../scripts/QUICK_START.md` - Quick start guide
- `CONFIG_RESTRUCTURE_SUMMARY.md` - Implementation details
