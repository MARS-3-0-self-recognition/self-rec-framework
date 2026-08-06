# Data Loader Scripts

This directory contains scripts for loading datasets from online sources (e.g., HuggingFace) and saving them in the proper format for this project.

## Output Format

All loader scripts save data to `data/{dataset_name}/input.json` in the format:

```json
{
  "uuid-string": "content text (article or question)",
  ...
}
```

UUIDs are generated using `uuid.uuid3(MY_DATASET_NAMESPACE, content)` for reproducibility.

## Available Datasets

### WikiSum

**Source**: [HuggingFace d0rj/wikisum](https://huggingface.co/datasets/d0rj/wikisum)
**Task**: Article summarization
**Content**: WikiHow articles (~40k total)
**Splits**: train (35.8k), validation (2k), test (2k)

**Usage**:
```bash
# Load 4 sample debug version (tracked in git)
uv run src/data_generation/data_loading/load_wikisum_debug.py

# Load first 100 samples from validation set
uv run src/data_generation/data_loading/load_wikisum.py --num_samples=100 --split=validation

# Load specific range of samples (e.g., samples 5-15 inclusive)
uv run src/data_generation/data_loading/load_wikisum.py --range=5-15 --split=validation

# Load 200 samples from training set with custom dataset name
uv run src/data_generation/data_loading/load_wikisum.py --num_samples=200 --dataset_name=wikisum_200

# Load samples 100-199 from test set
uv run src/data_generation/data_loading/load_wikisum.py --range=100-199 --split=test --dataset_name=wikisum_test
```

**Note**: You must specify either `--num_samples` or `--range` (cannot use both or omit both).

**Output**: `data/wikisum/input.json` (or custom dataset name)

### PKU-SafeRLHF

**Source**: [HuggingFace PKU-Alignment/PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF)
**Task**: Question answering / dialogue
**Content**: Prompts with safe/unsafe response pairs (~82k total)
**Splits**: train (73.9k), test (8.2k)
**Subsets**: default, alpaca-7b, alpaca2-7b, alpaca3-8b

**Usage**:
```bash
# Load first 100 samples from train set
uv run src/data_generation/data_loading/load_pku_saferlhf.py --num_samples=100 --split=train

# Load specific range of samples (10-50 inclusive)
uv run src/data_generation/data_loading/load_pku_saferlhf.py --range=10-50 --split=train

# Load 200 samples from alpaca-7b subset
uv run src/data_generation/data_loading/load_pku_saferlhf.py --num_samples=200 --subset=alpaca-7b --dataset_name=pku_alpaca_200

# Load samples 0-99 from test set
uv run src/data_generation/data_loading/load_pku_saferlhf.py --range=0-99 --split=test
```

**Note**: You must specify either `--num_samples` or `--range` (cannot use both or omit both).

**Output**: `data/pku_saferlhf/input.json` (or custom dataset name)

### ShareGPT

**Source**: [HuggingFace RyokoAI/ShareGPT52K](https://huggingface.co/datasets/RyokoAI/ShareGPT52K)
**Task**: Dialogue / conversation
**Content**: Human-AI conversations scraped from ShareGPT (~90k total)

**Usage**:
```bash
# Load first 100 samples
uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=100

# Load specific range of samples (10-50 inclusive)
uv run src/data_generation/data_loading/load_sharegpt.py --range=10-50

# Load 200 samples with custom dataset name
uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=200 --dataset_name=sharegpt_200

# Filter for conversations with at least 2 turns
uv run src/data_generation/data_loading/load_sharegpt.py --num_samples=100 --min_conversation_length=2
```

**Note**: You must specify either `--num_samples` or `--range` (cannot use both or omit both).

**Output**: `data/sharegpt/input.json` (or custom dataset name)

### BigCodeBench

**Source**: [HuggingFace bigcode/bigcodebench](https://huggingface.co/datasets/bigcode/bigcodebench)
**Task**: Code generation
**Content**: Programming problems with test cases and canonical solutions
**Versions**: v0.1.0_hf (default), v0.1.1, v0.1.2, v0.1.3, v0.1.4

**Usage**:
```bash
# Load first 100 samples
uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100

# Load specific range of samples (10-50 inclusive)
uv run src/data_generation/data_loading/load_bigcodebench.py --range=10-50

# Load 200 samples with custom dataset name
uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=200 --dataset_name=bigcodebench_200

# Load from a specific version
uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100 --version=v0.1.1

# Use complete_prompt instead of instruct_prompt
uv run src/data_generation/data_loading/load_bigcodebench.py --num_samples=100 --prompt_type=complete
```

**Note**: You must specify either `--num_samples` or `--range` (cannot use both or omit both).

**Output**: `data/bigcodebench/input.json` (or custom dataset name)

## Adding New Datasets

When creating a new loader script:

1. Import required utilities:
   ```python
   from self_rec_framework.src.helpers.constants import MY_DATASET_NAMESPACE
   from self_rec_framework.src.helpers.utils import save_json, data_dir, parse_range
   import uuid
   ```

2. Generate UUIDs from content:
   ```python
   sample_uuid = str(uuid.uuid3(MY_DATASET_NAMESPACE, content))
   ```

3. Save in proper format:
   ```python
   output_path = data_dir() / dataset_name / "input.json"
   save_json(input_dict, output_path)
   ```

4. Add argparse interface for flexibility (split selection, sample limits, etc.)

5. Document the new dataset in this README

## Notes

- All loader scripts should be executable: `chmod +x data_loading/script_name.py`
- Use descriptive dataset names (e.g., `wikisum_200` for subset of 200 samples)
- Consider memory usage for large datasets - process in batches if needed
- Datasets not matching `*debug` pattern won't be tracked in git (see `.gitignore`)
