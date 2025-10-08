# Pairwise Protocol Configuration Structure

This directory contains the hierarchical configuration system for pairwise self-recognition experiments, following the structure from `forced_recog`.

## Directory Structure

```
protocols/pairwise/config/
├── README.md                    # This file
├── __init__.py                  # PairwiseConfig dataclass and loaders
├── config_loader.py             # Hierarchical config loader
│
├── prompts/                     # Prompt templates (separated from experiment configs)
│   └── two_turn_prompts.yaml   # 2T prompts (system, user, rec_detection, pref_detection)
│
├── experiments/                 # Experiment configurations
│   ├── two_turn_rec_config.yaml           # Recognition paradigm
│   ├── two_turn_pref_config.yaml          # Preference paradigm
│   ├── two_turn_rec_config_batch.yaml     # Batch mode
│   └── two_turn_rec_config_mock.yaml      # Test/mock mode
│
├── qa.yaml                      # QA task config (legacy format)
├── summarisation.yaml           # Summarisation task config (legacy format)
└── two_turn.yaml                # 2T task config (legacy format)
```

## Configuration Philosophy

### Hierarchical Structure

Following the `forced_recog` pattern, configurations are split into:

1. **Prompt Files** (`prompts/`): Reusable prompt templates
2. **Experiment Files** (`experiments/`): Experiment-specific settings that reference prompts

This separation allows:
- Reusing prompts across multiple experiments
- Easy prompt iteration without changing experiment configs
- Clear separation of concerns

### Config File Format

#### Prompt Files (`prompts/*.yaml`)

Contains prompt templates:

```yaml
# System prompt
system: "You are an expert..."

# User prompt (can include placeholders)
user: "Summarize the following article: {passage}"

# Detection prompts (recognition vs preference)
rec_detection: "Which response did you originally produce?"
pref_detection: "Which response do you prefer?"
```

#### Experiment Files (`experiments/*.yaml`)

References a prompt file and specifies experiment parameters:

```yaml
# Experiment Type
experiment_type: "two_turn"
paradigm: "rec"  # or "pref"

# Prompt Configuration (reference to prompt file)
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

# Processing & Logging
max_samples: null
log_dir: "./logs/two_turn_rec"
```

## Usage

### Loading Configs in Scripts

```python
from protocols.pairwise.config.config_loader import load_experiment_config, ConfigLoader

# Load experiment config (automatically merges with prompts)
config = load_experiment_config('protocols/pairwise/config/experiments/two_turn_rec_config.yaml')

# Access prompts
loader = ConfigLoader()
system_prompt = loader.get_system_prompt(config)
user_prompt = loader.get_user_prompt_template(config)
detection_prompt = loader.get_detection_prompt(config)
```

### Creating New Experiments

1. **Reuse existing prompts**: Just create a new experiment config referencing an existing prompt file
2. **Create new prompts**: Add a new prompt file in `prompts/`, then reference it in your experiment config

Example experiment config:

```yaml
experiment_type: "two_turn"
paradigm: "rec"
prompt_file: "prompts/two_turn_prompts.yaml"
prompt_set: true
model: "anthropic/claude-3-5-sonnet-20241022"
model_name: "claude-3-5-sonnet"
alternative_model_name: "gpt-4"
dataset_name: "my_dataset"
model_generation_string: "control"
alternative_model_generation_string: "treatment"
log_dir: "./logs/my_experiment"
```

## Config Types

### Recognition (`rec`) vs Preference (`pref`)

- **Recognition**: Model identifies which output it originally produced
  - Uses `rec_detection` prompt
  - Tests self-recognition ability

- **Preference**: Model indicates which output it prefers
  - Uses `pref_detection` prompt
  - Tests preference alignment

### Batch Modes

Batch configs optimize for large-scale evaluation:
- Higher `max_connections`
- `batch_size` parameter
- Less verbose logging

### Mock/Test Modes

Test configs for pipeline validation:
- Use mock models or small datasets
- Very low `max_samples`
- Verbose logging (`DEBUG` level)

## Config Loader API

The `ConfigLoader` class provides methods to:

- `load_experiment_config(path)`: Load and merge experiment + prompts
- `get_system_prompt(config)`: Extract system prompt
- `get_user_prompt_template(config)`: Extract user prompt
- `get_detection_prompt(config)`: Extract detection prompt (rec/pref aware)

## Migration from Old Format

Old flat config:
```yaml
model: "anthropic/claude-3-5-sonnet"
system_prompt: "You are an expert..."
detection_prompt: "Which one did you write?"
```

New hierarchical config:

**prompts/my_prompts.yaml**:
```yaml
system: "You are an expert..."
rec_detection: "Which one did you write?"
```

**experiments/my_experiment.yaml**:
```yaml
prompt_file: "prompts/my_prompts.yaml"
prompt_set: true
model: "anthropic/claude-3-5-sonnet"
```

## Best Practices

1. **One prompt file per experimental paradigm**: E.g., `two_turn_prompts.yaml` for all 2T experiments
2. **Descriptive experiment names**: Use suffixes like `_rec`, `_pref`, `_batch`, `_mock`
3. **Relative paths in configs**: Always use paths relative to this directory
4. **Document prompts**: Add comments in prompt files explaining their purpose

## See Also

- `scripts/README.md` - How to run experiments
- `INTEGRATION_SUMMARY.md` - Technical integration details
- `forced_recog/configs/operationalizations/` - Original config structure
