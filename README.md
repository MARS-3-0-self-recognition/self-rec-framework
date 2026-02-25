# Self-Recognition Framework

A research framework for investigating **self-recognition in large language models** -- whether models can identify their own generated outputs when compared against text from other sources (other models, humans, or procedurally edited text).

Built on the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evaluation framework, with multi-provider support across OpenAI, Anthropic, Google, Together AI, and more.

## Research Overview

**Core Question:** Can LLMs recognize their own outputs?

This touches on fundamental questions about model self-awareness, internal representations vs. output patterns, and whether models track conversation context meaningfully.

### Experiment Design

The framework tests self-recognition across a comprehensive matrix of experimental conditions:

| Dimension | Options | Description |
|-----------|---------|-------------|
| **Tags** | AT (Assistant Tags), UT (User Tags) | Whether text to identify appears in assistant or user message roles |
| **Format** | PW (Pairwise), IND (Individual) | Side-by-side comparison vs. individual evaluation |
| **Interaction** | C (Conversation), Q (Query) | Chat history format vs. single-query format |
| **Task** | Rec (Recognition), Pref (Preference) | Identify own output vs. express preference |
| **Priming** | Pr (Primed), NPr (Not Primed) | Whether model receives priming context |
| **Generator Output** | FA (Final Answer), RT-FA (Reasoning+Final) | What portion of generation is shown |
| **Evaluator Reasoning** | DR (Direct), Rsn (Reasoning/CoT), Rsn-Inst (Both) | How the evaluator model reasons |

**Position Bias Control:** Every comparison is tested in both orderings (model output in position 1 and position 2), eliminating systematic positional preference.

### AT vs. UT: Why Two Tagging Modes?

- **AT (Assistant Tags):** Text to identify is presented in assistant message roles (conversation history). Models may be biased to claim assistant-tagged messages as "theirs" regardless of actual authorship.
- **UT (User Tags):** Text appears in user messages (standard prompting). Tests whether recognition relies on content/style analysis rather than role tags.

Comparing AT vs. UT reveals whether self-recognition is robust across message role contexts or driven by superficial role tag cues.

---

## Supported Models

The framework supports 50+ models across multiple providers:

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-5, gpt-5-mini, o3, o3-mini |
| **Anthropic** | haiku-3.5, haiku-4.5, sonnet-3.7, sonnet-4.5, opus-4.1 |
| **Google** | gemini-2.0-flash, gemini-2.0-flash-lite, gemini-2.5-flash, gemini-2.5-pro |
| **Together AI (Llama)** | ll-3.1-8b, ll-3.1-70b, ll-3.1-405b |
| **Together AI (Qwen)** | qwen-2.5-7b, qwen-2.5-72b, qwen-3.0-80b, qwen-3.0-235b |
| **Together AI (DeepSeek)** | deepseek-3.0, deepseek-3.1, deepseek-r1 |
| **Moonshot** | kimi-k2 |
| **XAI** | grok-3-mini, grok-4.1-fast |

Most models also have `-thinking` variants for chain-of-thought reasoning evaluations.

Models are organized into predefined **model sets** (e.g., `tutorial`, `dr`, `eval_cot-r`) for convenient batch operations. Use `-set <name>` in commands to reference a set.

---

## Project Structure

```
self-rec-framework/
├── self_rec_framework/              # Core Python package
│   └── src/
│       ├── inspect/                 # Inspect AI integration
│       │   ├── tasks.py             # Task definitions (comparison, conversational)
│       │   ├── config.py            # ExperimentConfig dataclass
│       │   ├── scorer.py            # Scoring logic (text parsing + logprobs)
│       │   └── data.py              # Dataset loading with position swapping
│       ├── data_generation/
│       │   ├── data_loading/        # Dataset loaders (WikiSum, PKU, ShareGPT, BigCodeBench)
│       │   └── procedural_editing/  # Treatment framework (typos, capitalization)
│       ├── helpers/
│       │   ├── model_names.py       # Model name mappings (short name <-> API name)
│       │   ├── model_sets.py        # Predefined model groupings
│       │   ├── constants.py         # UUID namespace, other constants
│       │   └── utils.py             # File I/O, path helpers
│       └── core_prompts/            # Prompt management
│
├── experiments/                     # Experiment definitions and tutorials
│   ├── _T00_tutorial_data_gen/      # Tutorial: data generation
│   ├── _T01_tutorial_evals/         # Tutorial: evaluation & analysis
│   ├── _T02_tutorial_comparisons/   # Tutorial: experiment comparison
│   ├── _scripts/                    # Shared scripts (generation, eval, analysis)
│   │   ├── gen/                     # Data generation utilities
│   │   ├── eval/                    # Evaluation sweep runner
│   │   └── analysis/                # Analysis & visualization scripts
│   └── ...                          # Additional experiments
│
├── data/
│   ├── input/                       # Input datasets (WikiSum, PKU, ShareGPT, etc.)
│   ├── results/                     # Evaluation results (Inspect eval logs)
│   └── analysis/                    # Generated analysis outputs (plots, CSVs)
│
├── scripts/                         # Utility scripts
│   ├── list_active_batches.py       # Monitor batch API jobs
│   ├── cancel_all_batches.py        # Cancel orphaned batch jobs
│   └── README_BATCH.md              # Batch mode documentation
│
├── pyproject.toml                   # Dependencies and package config
├── .env                             # API keys (not in git)
└── CLAUDE.md                        # AI assistant reference guide
```

---

## Setup

### Prerequisites

- Python >= 3.11
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (modern Python package manager)
- API keys for at least one LLM provider

### Installation

```bash
# Clone the repository
git clone git@github.com/MARS-3-0-self-recognition/self-rec-framework
cd self-rec-framework

# Sync dependencies and create virtual environment
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

### Environment Variables

Create a `.env` file in the project root with your API keys:

```bash
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
FIREWORKS_API_KEY=...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
```

You only need keys for the providers you intend to use.

### Optional: Inspect AI VSCode Extension

Install the [Inspect AI extension](https://marketplace.visualstudio.com/items?itemName=aisi-inspect.inspect-ai) for VSCode to browse `.eval` log files with a graphical viewer.

---

## Usage

All commands use `uv run` to execute within the virtual environment.

### Loading Datasets

Download datasets from HuggingFace to create standardized `input.json` files:

```bash
# WikiSum (article summarization)
uv run self_rec_framework/src/data_generation/data_loading/load_wikisum.py \
    --num_samples=100 --split=validation --dataset_name=wikisum/my_set

# PKU-SafeRLHF (question answering)
uv run self_rec_framework/src/data_generation/data_loading/load_pku_saferlhf.py \
    --num_samples=100 --split=train --dataset_name=pku_saferlhf/my_set

# ShareGPT (general conversation)
uv run self_rec_framework/src/data_generation/data_loading/load_sharegpt.py \
    --num_samples=100 --dataset_name=sharegpt/my_set

# BigCodeBench (code generation)
uv run self_rec_framework/src/data_generation/data_loading/load_bigcodebench.py \
    --num_samples=100 --dataset_name=bigcodebench/my_set
```

All loaders support `--num_samples` or `--range` (e.g., `--range=0-19`) for sample selection.

### Data Generation

Generate model outputs across a set of models:

```bash
uv run experiments/_scripts/generate_data_sweep.py \
    --model_names -set tutorial \
    --dataset_path=data/input/wikisum/tutorial_set/input.json \
    --dataset_config=experiments/_T00_tutorial_data_gen/config.yaml
```

### Running Evaluations

Run pairwise evaluation sweeps (all model pairs tested automatically):

```bash
uv run experiments/_scripts/eval/run_experiment_sweep.py \
    --model_names -set tutorial \
    --generator_models -set tutorial \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/tutorial_set \
    --experiment_config experiments/_T01_tutorial_evals/config.yaml \
    --max-tasks 20 -y
```

### Analysis

Run analysis scripts on evaluation results:

```bash
# Recognition accuracy heatmap
uv run experiments/_scripts/analysis/recognition_accuracy.py \
    --results_dir data/results/wikisum/tutorial_set/_T01_tutorial_evals \
    --model_names -set tutorial

# Evaluator performance
uv run experiments/_scripts/analysis/evaluator_performance.py \
    --results_dir data/results/wikisum/tutorial_set/_T01_tutorial_evals \
    --model_names -set tutorial
```

### Batch Mode

For large-scale experiments, enable batch API mode for ~50% cost savings:

```bash
uv run experiments/_scripts/eval/run_experiment_sweep.py \
    --model_names -set dr \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/ICML_01.../config.yaml \
    --batch -y
```

> **Important:** Always use `tmux` for batch runs. Batch jobs continue on provider servers even if your terminal closes, but results cannot be retrieved without the active process. See [scripts/README_BATCH.md](scripts/README_BATCH.md) for details.

---

## Procedural Editing

In addition to comparing outputs from different models, the framework supports **procedural editing** as an alternative source of comparison text:

| Treatment | Description | Strength Levels |
|-----------|-------------|-----------------|
| **Typos** | Keyboard-adjacent character substitutions | S1 - S4 |
| **Capitalization** | Systematic capitalization changes | S1 - S4 |

This enables controlled experiments testing whether models can distinguish their own output from slightly modified versions of it.

---

## Git Workflow

- Create branches in the format `<your_name>/<branch_name>`
- All changes to `main` must go via pull request
- Pre-commit hooks run `ruff` automatically on every commit

```bash
# Run pre-commit hooks manually
uv run pre-commit run --all-files

# Update hook versions
uv run pre-commit autoupdate
```

---

## Tutorial

The tutorials walk through the full experimental workflow in three phases: **data generation**, **evaluation**, and **analysis**. Each phase has pre-built bash scripts in the `experiments/` directory covering four datasets (WikiSum, BigCodeBench, PKU-SafeRLHF, ShareGPT).

Below, we demonstrate the full pipeline using **WikiSum** as the example. The process is identical for the other datasets -- just swap the script name.

### Prerequisites

Before running tutorials, ensure you have:
1. Completed the [Setup](#setup) steps above
2. API keys configured for at least one provider
3. Downloaded the tutorial dataset (or use one already in `data/input/`)

The tutorial model set uses two small, inexpensive models (`ll-3.1-8b` and `qwen-2.5-7b` via Together AI), so you only need a Together AI API key to follow along.

### Phase 1: Data Generation (`_T00_tutorial_data_gen`)

Generate model outputs for the tutorial model set on the WikiSum dataset:

```bash
bash experiments/_T00_tutorial_data_gen/bash/00_sweep/data_gen_sweep_wikisum_tutorial_set.sh
```

**What this does:**
- Calls `experiments/_scripts/generate_data_sweep.py` with the tutorial model set
- Generates outputs from each model in the set for every sample in the dataset
- Saves results to `data/input/wikisum/tutorial_set/` alongside the input data
- Uses the generation config in `experiments/_T00_tutorial_data_gen/config.yaml` (temperature=0.0, with S2-strength typo treatments)

**Other datasets:** Replace `wikisum` with `sharegpt`, `pku`, or `bigcodebench` in the script name:
```bash
bash experiments/_T00_tutorial_data_gen/bash/00_sweep/data_gen_sweep_sharegpt_tutorial_set.sh
bash experiments/_T00_tutorial_data_gen/bash/00_sweep/data_gen_sweep_pku_tutorial_set.sh
bash experiments/_T00_tutorial_data_gen/bash/00_sweep/data_gen_sweep_bigcodebench_tutorial_set.sh
```

**Sweeps:** To run all scripts in a sweep:
```bash
bash run.sh experiments/_T00_tutorial_data_gen/bash/00_sweep
```

### Phase 2: Evaluation (`_T01_tutorial_evals`)

Run self-recognition evaluations for all model pairs:

```bash
bash experiments/_T01_tutorial_evals/bash/a00_sweep/00_wikisum_tutorial_set.sh
```

**What this does:**
- Runs pairwise evaluations: each model judges whether it can identify its own output vs. another model's output
- Tests all permutations of model pairs in the tutorial set
- Uses the experiment config in `experiments/_T01_tutorial_evals/config.yaml`:
  - `tags: UT` (User Tags paradigm)
  - `format: PW-Q` (Pairwise-Query)
  - `task: Rec` (Recognition)
  - `evaluator_reasoning: DR` (Direct Response, no chain-of-thought)
- Saves Inspect eval logs to `data/results/`
- The shared configuration in `bash/a00_sweep/config.sh` controls model sets, parallelism, and batch settings

**Other datasets and sweeps:**
Protocol is the same as above for running other datasets and sweeps.

### Phase 3: Analysis (`_T01_tutorial_evals`)

Analyze evaluation results with per-dataset and cross-dataset scripts.

#### Per-dataset analysis

Each dataset has three analysis scripts:

```bash
# Recognition accuracy (generates heatmaps of model self-recognition scores)
bash experiments/_T01_tutorial_evals/bash/b00_analysis/wikisum/00-recognition_accuracy.sh

# Evaluator performance (how well each model performs as an evaluator)
bash experiments/_T01_tutorial_evals/bash/b00_analysis/wikisum/01-evaluator_performance.sh

# Recognition disagreement (where models disagree on self-recognition)
bash experiments/_T01_tutorial_evals/bash/b00_analysis/wikisum/02-recognition_disagreement.sh
```

#### Cross-dataset analysis

Compare results across all four datasets:

```bash
# Aggregate performance data
bash experiments/_T01_tutorial_evals/bash/b00_analysis/_inter-dataset/00a-performance_aggregate.sh

# Plot aggregated performance
bash experiments/_T01_tutorial_evals/bash/b00_analysis/_inter-dataset/00b-plot_performance_aggregate.sh

# Performance contrast across datasets
bash experiments/_T01_tutorial_evals/bash/b00_analysis/_inter-dataset/01-performance_contrast.sh

# Performance vs. model size
bash experiments/_T01_tutorial_evals/bash/b00_analysis/_inter-dataset/02-performance_vs_size.sh

# Rank distance analysis
bash experiments/_T01_tutorial_evals/bash/b00_analysis/_inter-dataset/03-rank-distance.sh
```

#### Run all analyses at once

```bash
bash experiments/_T01_tutorial_evals/bash/b00_analysis/run_all_analyses.sh
```

This runs every per-dataset and cross-dataset analysis script in sequence.

Analysis outputs (plots, CSVs, statistics) are saved to `data/analysis/`.

### Phase 4 (Optional): Experiment Comparison (`_T02_tutorial_comparisons`)

Compare results between two different experiments:

```bash
bash experiments/_T02_tutorial_comparisons/_T01_tutorial_evals-vs-_T01_tutorial_evals/00-performance_contrast.sh
```

**What this does:**
- Extracts the two experiment names from the directory name (`{exp1}-vs-{exp2}`)
- Finds the most recent aggregated performance data for each experiment
- Generates a comparative analysis

To compare different experiments, create a new directory following the naming convention `{exp1}-vs-{exp2}/` with a `config.sh` and `00-performance_contrast.sh` script.

### Summary

| Phase | Tutorial Directory | Purpose |
|-------|-------------------|---------|
| **1. Data Gen** | `_T00_tutorial_data_gen/` | Generate model outputs for each dataset |
| **2. Evaluation** | `_T01_tutorial_evals/` (a00_sweep) | Run self-recognition evaluations |
| **3. Analysis** | `_T01_tutorial_evals/` (b00_analysis) | Visualize and analyze results |
| **4. Comparison** | `_T02_tutorial_comparisons/` | Compare experiments against each other |

---

## License

MIT License. See [LICENSE](LICENSE) for details.
