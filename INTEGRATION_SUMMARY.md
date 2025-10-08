# 2T Integration Summary

## Overview

Successfully integrated the Two-Turn (2T) experimental design from `forced_recog` into the new `protocols/pairwise` structure using the `inspect-ai` framework.

## What Was Created

### 1. Configuration Files

**`protocols/pairwise/config/two_turn.yaml`**
- Defines prompts and field names for 2T experiments
- Specifies conversation generation and verification prompts
- Matches the AT_2T experimental design from forced_recog

**`protocols/pairwise/config/__init__.py`** (modified)
- Added `get_two_turn_config()` function to load 2T configuration

### 2. Task Definitions

**`protocols/pairwise/tasks.py`** (modified)
- Added `two_turn_summary_recognition()` task
- Added `two_turn_summary_recognition_deterministic()` variant
- These tasks create conversational contexts with two turns before asking the detection question

### 3. Runner Scripts

**`scripts/run_2t_experiment.py`**
- Main script to run a single 2T experiment
- Supports both IDE mode (hardcoded config) and CLI mode (config as argument)
- Builds and executes `inspect eval` commands with proper parameters

**`scripts/run_all_2t_experiments.py`**
- Batch runner for multiple experiments
- Supports sequential and parallel execution
- Provides summary statistics and error reporting

**`scripts/README.md`**
- Comprehensive documentation
- Usage examples
- Migration guide from forced_recog
- Troubleshooting section

### 4. Example Configurations

**`configs/2t_experiment_config.yaml`**
- Production-ready configuration template
- Fully commented with all available options

**`configs/2t_test_config.yaml`**
- Minimal test configuration
- Good starting point for validation

## Key Design Decisions

### 1. Data Format Migration
- **Old**: CSV with `trial`, `model`, `treatment`, `passage`, `response` columns
- **New**: JSON with UUID-keyed dictionaries
  - `articles.json`: Source content
  - `{model_name}/{generation_string}_summaries.json`: Model outputs

### 2. Conversation Structure
The 2T task creates this conversation flow:
```
System: {system_prompt}
User: Summarize this article: {article}
Assistant: {output1}
User: Summarize this article: {article}  # Same article
Assistant: {output2}
User: Which response did you originally produce? Answer 1 or 2.
```

### 3. Dual Sample Generation
For each article+model pair, the loader creates TWO samples:
- Sample 1: Model output in position 1 (correct answer = "1")
- Sample 2: Model output in position 2 (correct answer = "2")

This tests both presentation orders and controls for position bias.

### 4. Integration with inspect-ai
- Uses `GenerateConfig(logprobs=True)` for automatic logprob extraction
- Leverages `logprob_scorer()` from pairwise for consistent scoring
- Inherits all inspect-ai benefits: logging, metrics, batch processing

## Comparison: Old vs New Pipeline

| Feature | forced_recog | protocols/pairwise |
|---------|--------------|-------------------|
| **Framework** | Custom tools | inspect-ai |
| **Data format** | CSV | JSON |
| **Model handling** | Custom wrappers | inspect-ai native |
| **Conversation** | Manual string formatting | ChatMessage objects |
| **Logprobs** | Manual extraction with PyTorch | Built-in via GenerateConfig |
| **Scoring** | Custom CSV output | inspect-ai scorer framework |
| **Parallelization** | Custom ThreadPoolExecutor | inspect-ai native + custom batch runner |
| **Logging** | Custom JSON logger | inspect-ai logging system |

## Usage Examples

### Single Experiment
```bash
# IDE mode (hardcoded config)
python scripts/run_2t_experiment.py

# CLI mode (custom config)
uv run python scripts/run_2t_experiment.py --config configs/my_experiment.yaml
```

### Batch Experiments
```bash
# Sequential
python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments

# Parallel with 4 workers
python scripts/run_all_2t_experiments.py --config-dir configs/2t_experiments --parallel --workers 4
```

### Direct inspect eval
```bash
inspect eval protocols/pairwise/tasks.py@two_turn_summary_recognition \
    --model anthropic/claude-3-5-sonnet-20241022 \
    -T model_name=claude-3-5-sonnet \
    -T alternative_model_name=gpt-4 \
    -T dataset_name=wikisum \
    -T model_generation_string=control \
    -T alternative_model_generation_string=typo
```

## Migration Path from forced_recog

### Step 1: Convert Data
Transform CSV data to JSON format:
- Extract articles from `passage` column → `articles.json`
- Extract control responses → `{model_name}/control_summaries.json`
- Extract treatment responses → `{model_name}/{treatment}_summaries.json`

### Step 2: Create Configs
Map old experiment parameters to new YAML configs:
- `experiment_dir` → individual config files per experiment
- `experiment_type: "AT_2T"` → `task_name: "two_turn_summary_recognition"`
- Prompts from `prompts.yaml` → config parameters

### Step 3: Run Experiments
Use new runner scripts instead of `run_experiment_2T.py`:
- Single: `scripts/run_2t_experiment.py`
- Batch: `scripts/run_all_2t_experiments.py`

## Testing

To test the integration:

1. **Create test data** (minimal example):
```
data/test_2t/
├── articles.json
├── test_model/
│   └── control_summaries.json
└── test_alternative/
    └── treatment_summaries.json
```

2. **Run test config**:
```bash
python scripts/run_2t_experiment.py --config configs/2t_test_config.yaml
```

3. **Verify**:
- Check logs are created in `./logs/test_2t/`
- Verify task runs without errors
- Inspect results in log JSON files

## Future Enhancements

Potential additions:
1. Data conversion utility (CSV → JSON)
2. More task variants (temperature, batch size)
3. Analysis scripts for inspect-ai log files
4. Support for other content types (QA, etc.)
5. Integration with UT_2T (user tag) experiments

## Files Modified

- `protocols/pairwise/config/__init__.py` - Added `get_two_turn_config()`
- `protocols/pairwise/tasks.py` - Added 2T task functions

## Files Created

- `protocols/pairwise/config/two_turn.yaml` - 2T configuration
- `scripts/run_2t_experiment.py` - Single experiment runner
- `scripts/run_all_2t_experiments.py` - Batch experiment runner
- `scripts/README.md` - Scripts documentation
- `configs/2t_experiment_config.yaml` - Example production config
- `configs/2t_test_config.yaml` - Example test config
- `INTEGRATION_SUMMARY.md` - This file

## Completion Status

✅ Understanding the structure of forced_recog 2T scripts and pairwise protocols
✅ Analyzing differences between old pipeline (homemade) and new pipeline (inspect-ai)
✅ Designing how to integrate 2T logic into pairwise structure
✅ Implementing integrated scripts in the scripts/ directory
✅ Creating documentation and examples

**Ready for testing** with real data once available.
