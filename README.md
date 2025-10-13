# Self-Recognition

## Setup

### Install `uv`
Follow instructions on [uv docs](https://docs.astral.sh/uv/getting-started/installation/).
If you have a mac, it is probably easiest to use homebrew.

### Clone the repository
And navigate to the directory
```
git clone git@github.com:W-L-W/EvalAwareness.git
cd EvalAwareness
```

### Sync uv
Sync dependencies and create virtual environment with development tools (includes ipykernel for Jupyter notebooks, pre-commit, etc):
```
uv sync --extra dev
```

### Install pre-commit hooks
(this ensures code quality checks run automatically on every commit):
```
uv run pre-commit install
```

### Inspect plug-in
Install the inspect plug-in for vscode to easily view eval-logs.

## Usage

To run commands in the virtual environment, use `uv run`:
```
uv run your_script.py
```

To activate the virtual environment manually (if needed):
```
source .venv/bin/activate
```

### Pre-commit Hooks
Once installed, pre-commit hooks will run automatically on every `git commit`. You can also run them manually:
```
uv run pre-commit run --all-files
```

To update pre-commit hook versions:
```
uv run pre-commit autoupdate
```

# Git Hygiene

- Please make any edits on a separate branch in format `<your_name>/<branch_name>`
- Any changes to main must go via PR


# Structure
```
self-recognition/
│
├── run.py                          # Hydra entry point (python run.py ...)
├── tasks.py                        # Inspect task definitions (inspect eval tasks.py@...)
├── README.md                       # Full documentation
├── QUICKSTART.md                   # Quick start guide
├── pyproject.toml                  # Python dependencies and config
├── .gitignore                      # Git ignore rules
│
├── src/                            # Source code
│   ├── __init__.py
│   │
│   ├── protocols/                  # Protocol implementations
│   │   ├── __init__.py
│   │   ├── pairwise/              # Pairwise recognition protocol
│   │   │   ├── __init__.py
│   │   │   ├── solvers.py         # Protocol solvers (comparison, conversational)
│   │   │   ├── scorers.py         # Scorers (logprob, match)
│   │   │   └── data_loader.py     # Dataset loading with UUID alignment
│   │   └── individual/            # TODO: Individual recognition (MVP+)
│   │
│   ├── datasets/                   # Base generation scripts
│   │   ├── __init__.py
│   │   └── base_generation.py     # TODO: Inspect batch generation tasks
│   │
│   └── utils/                      # Utilities
│       ├── __init__.py
│       └── config.py               # Model name resolution
│
├── configs/                        # Hydra configuration files
│   ├── model/                      # Model name mappings
│   │   ├── claude.yaml            # short_name -> full provider/model
│   │   ├── gpt4.yaml
│   │   └── ...
│   │
│   ├── dataset/                    # Dataset configurations
│   │   ├── cnn.yaml               # Summarization dataset
│   │   ├── qa.yaml                # QA dataset (MVP+)
│   │   └── ...
│   │
│   ├── protocol/                   # Protocol configurations
│   │   ├── comparison.yaml        # Single message protocol
│   │   ├── conversational.yaml    # Fake conversation protocol
│   │   ├── comparison_qa.yaml     # QA variant (MVP+)
│   │   └── ...
│   │
│   ├── scorer/                     # Scorer configurations
│   │   ├── logprob.yaml           # Logprob-based scoring
│   │   └── match.yaml             # Text extraction scoring
│   │
│   └── experiment/                 # Composed experiment configs
│       ├── pairwise_base.yaml     # Base pairwise experiment
│       └── hypothesis1_assistant_tags.yaml  # Example hypothesis config
│
├── prompts/                        # Prompt templates
│   └── pairwise/
│       ├── summarization/          # Summarization prompts
│       │   ├── comparison.txt     # Comparison protocol template
│       │   ├── generation.txt     # Conversational generation request
│       │   └── verification.txt   # Conversational verification question
│       └── qa/                     # QA prompts (MVP+)
│           └── comparison.txt     # QA comparison template
│
├── data/                           # Dataset storage (not in git)
│   └── {dataset_name}/            # e.g., cnn, wikisum
│       ├── articles.json          # {uuid: article_text}
│       └── {model_name}/          # Full model name (e.g., anthropic--claude-3-5-sonnet)
│           └── {gen_string}_summaries.json  # {uuid: summary_text}
│
└── logs/                           # Evaluation logs (not in git)
    └── {timestamp}/               # Inspect eval outputs

```
