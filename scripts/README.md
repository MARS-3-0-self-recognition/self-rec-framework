# CSV-Based Experiment Pipeline

## Overview

This pipeline allows you to run 2T (Two-Turn) self-recognition experiments directly from CSV files, without needing to convert to JSON format. CSV files are more human-readable and easier to create/edit.

## CSV Format

Your CSV file should have these columns:

```csv
trial,source1,content1,source2,content2
0,claude-3-5-haiku-20241022,"First summary...",claude-3-5-haiku-20241022_typos,"Typo version..."
1,claude-3-5-haiku-20241022,"Second summary...",claude-3-5-haiku-20241022_typos,"Typo version..."
...
```

**Column Description:**
- `trial` - Trial number (0, 1, 2, ...)
- `source1` - Base model name (the model being evaluated)
- `content1` - Output from the base model (correct answer)
- `source2` - Treatment/alternative model name (often with suffix like "_typos")
- `content2` - Output from treatment (modified/alternative version)

**Important:**
- `source1`/`content1` should be the model's own output (correct answer)
- `source2`/`content2` should be the alternative/treatment output
- The loader creates 2 samples per row (testing both orderings)

## Quick Start

### Option 1: Using Python Script (Recommended)

**IDE Mode** (edit hardcoded path in `run_csv_experiment.py`):
```python
# Edit HARDCODED_CSV_PATH in run_csv_experiment.py
HARDCODED_CSV_PATH = "data/experiments/pairwise/two-turn/typos/S1/your-file.csv"
```

Then run:
```bash
python run_csv_experiment.py
```

**CLI Mode**:
```bash
# Basic usage
python run_csv_experiment.py data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv

# With custom model
python run_csv_experiment.py path/to/data.csv --model anthropic/claude-sonnet-4-20250514

# With article text (if needed)
python run_csv_experiment.py path/to/data.csv --article-text "The article text..."

# Using uv
uv run python run_csv_experiment.py path/to/data.csv
```

### Option 2: Using Shell Scripts

**Linux/Mac (Bash)**:
```bash
chmod +x run_csv_experiment.sh
./run_csv_experiment.sh data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
```

**Windows (PowerShell)**:
```powershell
.\run_csv_experiment.ps1 data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
```

### Option 3: Direct inspect eval

```bash
inspect eval protocols/pairwise/tasks.py@two_turn_summary_recognition_csv \
    --model anthropic/claude-3-5-haiku-20241022 \
    -T csv_path=data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
```

## How It Works

### Data Loading (`protocols/pairwise/data_csv.py`)

The CSV loader:
1. Reads the CSV file using pandas
2. Filters out empty rows
3. For each row, creates **2 samples**:
   - Sample 1: source1 first (correct answer = "1")
   - Sample 2: source2 first (correct answer = "2")
4. Returns list of sample dictionaries

### Task Builder (`protocols/pairwise/task.py`)

`conversational_self_recognition_from_csv()` creates a conversation:

```
[System]: "You are an expert WikiHow summarizer..." (if system_prompt provided)

[User]: "Summarize the following article..."
[Assistant]: {output1}

[User]: "Summarize the following article..." (same prompt)
[Assistant]: {output2}

[User]: "Which response did you originally produce? Answer 1 or 2."
```

### Task Wrapper (`protocols/pairwise/tasks.py`)

`two_turn_summary_recognition_csv()` is an @task that uses the CSV builder with two_turn config.

## Directory Structure

Your CSV files can be organized however you like. Suggested structure:

```
data/experiments/pairwise/
└── two-turn/
    ├── typos/
    │   ├── S1/
    │   │   └── claude-3-5-haiku-20241022.csv
    │   └── S2/
    │       └── claude-3-5-haiku-20241022.csv
    ├── capitalization/
    │   └── S1/
    │       └── model-name.csv
    └── other_treatments/
        └── ...
```

## Examples

### Example 1: Basic Run
```bash
python run_csv_experiment.py data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
```

### Example 2: Different Model
```bash
python run_csv_experiment.py \
    data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv \
    --model anthropic/claude-sonnet-4-20250514
```

### Example 3: With Article Text
If your trials all use the same article:
```bash
python run_csv_experiment.py \
    path/to/data.csv \
    --article-text "How to store oysters properly..."
```

## Creating CSV Data

### From Scratch

Create a CSV with required columns:

```python
import pandas as pd

data = {
    'trial': [0, 1, 2],
    'source1': ['claude-3-5-haiku-20241022'] * 3,
    'content1': [
        'Summary 1 from claude...',
        'Summary 2 from claude...',
        'Summary 3 from claude...'
    ],
    'source2': ['claude-3-5-haiku-20241022_typos'] * 3,
    'content2': [
        'Smmary 1 fom cluade...',  # Typo version
        'Summar 2 frm claude...',
        'Sumary 3 from claud...'
    ]
}

df = pd.DataFrame(data)
df.to_csv('my_experiment.csv', index=False)
```

### From existing data

If you have outputs from two models, combine them:

```python
import pandas as pd

# Load model outputs
control_outputs = ["Summary 1...", "Summary 2..."]
treatment_outputs = ["Typo version 1...", "Typo version 2..."]

df = pd.DataFrame({
    'trial': range(len(control_outputs)),
    'source1': 'claude-3-5-haiku-20241022',
    'content1': control_outputs,
    'source2': 'claude-3-5-haiku-20241022_typos',
    'content2': treatment_outputs
})

df.to_csv('experiment.csv', index=False)
```

## Advantages of CSV Format

✅ **Human-readable** - Easy to view and edit in Excel/spreadsheet software
✅ **Simple structure** - All data in one file
✅ **Easy to create** - Can generate from pandas DataFrames
✅ **Version control friendly** - Text-based, easy to diff
✅ **No conversion needed** - Direct loading into tasks

## Results

Results are stored in the log directory (default: `./logs/`):

```
logs/
└── {timestamp}/
    ├── eval_log.json      # Full experiment log
    ├── results.json       # Aggregated metrics
    └── samples.json       # Individual sample results
```

## Troubleshooting

### CSV file not found
- Check the path is correct
- Use forward slashes (/) or escaped backslashes (\\\\) in paths
- Paths are relative to where you run the script

### "Column not found" error
- Verify your CSV has the required columns: trial, source1, content1, source2, content2
- Check for typos in column names
- Ensure first row is the header

### Empty samples
- Check for empty rows in your CSV (they're automatically filtered)
- Verify content1 and content2 have actual text

### Model API errors
- Check your API keys are set (ANTHROPIC_API_KEY, OPENAI_API_KEY, etc.)
- Try reducing concurrency with `--max-connections`

## Advanced Usage

### Custom Column Names

If your CSV has different column names, you can modify `load_dataset_from_csv()`:

```python
samples = load_dataset_from_csv(
    csv_path,
    trial_col="trial_number",
    source1_col="model_A",
    content1_col="output_A",
    source2_col="model_B",
    content2_col="output_B"
)
```

### Batch Processing Multiple CSV Files

Create a simple loop:

```bash
for csv in data/experiments/pairwise/two-turn/typos/S1/*.csv; do
    python run_csv_experiment.py "$csv"
done
```

Or use Python:

```python
from pathlib import Path
import subprocess

csv_dir = Path("data/experiments/pairwise/two-turn/typos/S1")
for csv_file in csv_dir.glob("*.csv"):
    subprocess.run(['python', 'run_csv_experiment.py', str(csv_file)])
```

## See Also

- `protocols/pairwise/data_csv.py` - CSV data loader implementation
- `protocols/pairwise/task.py` - Task builder with `conversational_self_recognition_from_csv()`
- `protocols/pairwise/tasks.py` - Task definition `two_turn_summary_recognition_csv`
