# Self-Recognition Framework

A research framework for investigating **self-recognition in large language models** -- whether models can identify their own generated outputs when compared against text from other sources (other models, humans, or procedurally edited text).

Built on the [Inspect AI](https://inspect.ai-safety-institute.org.uk/) evaluation framework, with multi-provider support across OpenAI, Anthropic, Google, Together AI, and more.

## Table of Contents

- [Research Overview](#research-overview)
- [Setup](#setup)
- [Usage](#usage)
- [Tutorial](#tutorial)
- [Procedural Editing](#procedural-editing)
- [Git Workflow](#git-workflow)
- [Supported Providers](#supported-providers)
- [Project Structure](#project-structure)
- [License](#license)

---

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
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
FIREWORKS_API_KEY=...
```

You only need keys for the providers you intend to use. See [Supported Providers](#supported-providers) for the full list.

### Optional: Inspect AI VSCode Extension

Install the [Inspect AI extension](https://marketplace.visualstudio.com/items?itemName=aisi-inspect.inspect-ai) for VSCode to browse `.eval` log files with a graphical viewer.

---

## Usage

The framework installs a set of `srf-*` console commands (see `pyproject.toml`). All commands run inside the uv environment via `uv run`.

| Command | Purpose |
|---------|---------|
| `srf-generate` / `srf-generate-sweep` | Generate model outputs (single model / model sweep) |
| `srf-eval` / `srf-eval-sweep` | Run self-recognition evaluations (single / sweep) |
| `srf-recognition-accuracy` | Per-dataset recognition-accuracy heatmaps |
| `srf-evaluator-performance` | Per-evaluator performance scores |
| `srf-aggregate-performance-data` | Aggregate per-dataset performance into one table |
| `srf-plot-aggregated-performance` | Plot aggregated performance figures |
| `srf-performance-contrast` / `srf-performance-vs-size` / `srf-rank-distance` | Cross-dataset analyses |
| `srf-experiment-contrast` | Contrast two experiments |
| `srf-list-models` / `srf-list-batches` / `srf-cancel-batches` | Admin helpers |

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
uv run srf-generate-sweep \
    --model_names -set tutorial \
    --dataset_path=data/input/wikisum/tutorial_set/input.json \
    --dataset_config=experiments/_tutorials/_T00_tutorial_data_gen/config.yaml
```

### Running Evaluations

Run evaluation sweeps (all model pairs tested automatically):

```bash
uv run srf-eval-sweep \
    --model_names -set tutorial \
    --generator_models -set tutorial \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/tutorial_set \
    --experiment_config experiments/_tutorials/_T01_tutorial_evals_individual/config.yaml \
    --max-tasks 20 -y
```

### Analysis

Run analysis scripts on evaluation results:

```bash
# Recognition accuracy heatmap
uv run srf-recognition-accuracy \
    --results_dir data/results/wikisum/tutorial_set/_T01_tutorial_evals_individual \
    --model_names -set tutorial

# Evaluator performance
uv run srf-evaluator-performance \
    --results_dir data/results/wikisum/tutorial_set/_T01_tutorial_evals_individual \
    --model_names -set tutorial
```

In practice, the tutorial bash scripts wrap these `srf-*` commands and read all parameters from the experiment `config.yaml` -- see the [Tutorial](#tutorial).

### Batch Mode

For large-scale experiments, enable batch API mode for ~50% cost savings:

```bash
uv run srf-eval-sweep \
    --model_names -set dr \
    --treatment_type other_models \
    --dataset_dir_path data/input/wikisum/training_set_1-20 \
    --experiment_config experiments/.../config.yaml \
    --batch -y
```

> **Important:** Always use `tmux` for batch runs. Batch jobs continue on provider servers even if your terminal closes, but results cannot be retrieved without the active process. Use `srf-list-batches` / `srf-cancel-batches` to monitor and clean up jobs.

---

## Tutorial

The tutorials walk through the full experimental workflow: **data generation -> evaluation -> analysis -> comparison**. They live under `experiments/_tutorials/` and cover four datasets (WikiSum, BigCodeBench, PKU-SafeRLHF, ShareGPT).

Each experiment's `config.yaml` is the **single source of truth** -- model sets, sweep parameters, and the figures to produce all come from there. The bash scripts read it through the shared loader, so you rarely pass CLI flags directly.

> **Where to run from:** run the tutorial entry-point scripts (`run_sweep.sh`, `run_all_analyses.sh`, `run_comparison.sh`) from the repository root. They use the bundled `scripts/utils/run.sh` runner and `scripts/utils/load_config.sh` config loader.

The tutorial model set uses two small, inexpensive models -- `gemma-3n-e4b` and `qwen-2.5-7b`, both served by **Together AI** -- so a single `TOGETHER_API_KEY` is enough to follow along.

### Prerequisites

1. Complete the [Setup](#setup) steps above.
2. Configure a `TOGETHER_API_KEY` (or keys for whichever providers your model set uses).
3. Have the tutorial datasets in `data/input/` (or download them via the [loaders](#loading-datasets)).

### Phase 1: Data Generation (`_T00_tutorial_data_gen`)

Generate the tutorial model set's outputs for each dataset. Run every sweep script in the directory at once:

```bash
bash experiments/_tutorials/_T00_tutorial_data_gen/bash/00_sweep/run_sweep.sh
```

Or generate a single dataset:

```bash
bash experiments/_tutorials/_T00_tutorial_data_gen/bash/00_sweep/data_gen_sweep_wikisum_tutorial_set.sh
# ...or _sharegpt_, _pku_, _bigcodebench_
```

**What this does:**
- Calls `srf-generate-sweep` with the model set from `config.yaml`.
- Generates outputs for every sample in each dataset and saves them alongside the input data in `data/input/<dataset>/tutorial_set/`.
- Uses `_T00_tutorial_data_gen/config.yaml` (temperature 0.0; also produces S2-strength `caps` and `typos` treatment variants).

### Phase 2: Evaluation

There are two evaluation experiments, differing only in their **format**:

| Experiment | Directory | Format |
|-----------|-----------|--------|
| Individual | `_T01_tutorial_evals_individual` | `IND-Q` -- each output judged on its own |
| Pairwise | `_T02_tutorial_evals_pairwise` | `PW-Q` -- two outputs compared side by side |

Run the full sweep for an experiment (all datasets):

```bash
bash experiments/_tutorials/_T01_tutorial_evals_individual/bash/a00_sweep/run_sweep.sh
bash experiments/_tutorials/_T02_tutorial_evals_pairwise/bash/a00_sweep/run_sweep.sh
```

Or a single dataset:

```bash
bash experiments/_tutorials/_T01_tutorial_evals_individual/bash/a00_sweep/00_wikisum_tutorial_set.sh
```

**What this does:**
- Runs self-recognition evaluations: each model judges whether it can identify its own output.
- Reads all parameters from the experiment `config.yaml` (e.g. `tags: UT`, `task: Rec`, `evaluator_reasoning: DR`, and the format above).
- Saves Inspect eval logs under `data/results/`.

### Phase 3: Analysis

Run every per-dataset and cross-dataset analysis for an experiment:

```bash
bash experiments/_tutorials/_T01_tutorial_evals_individual/bash/b00_analysis/run_all_analyses.sh
bash experiments/_tutorials/_T02_tutorial_evals_pairwise/bash/b00_analysis/run_all_analyses.sh
```

This runs, in sequence:
- **Per-dataset** (`<dataset>/`): recognition accuracy, evaluator performance, recognition disagreement.
- **Cross-dataset** (`_inter-dataset/`): aggregate performance, plot aggregated performance, performance contrast, performance-vs-size, rank distance.

**Controlling which figures are produced.** The `figures_to_produce` list in `config.yaml` names the figures (by output filename, no extension) that the analysis should emit; anything not listed is skipped. The tutorial keeps the output minimal:

```yaml
# _T01.../config.yaml and _T02.../config.yaml
figures_to_produce: ["aggregated_performance_grouped"]
```

So each experiment's analysis produces `aggregated_performance_grouped.png` (plus the intermediate CSVs that feed it and the comparison). Remove or empty the list to produce every figure.

Analysis outputs are written to `data/analysis/_aggregated_data/<experiment>/<timestamp>/`.

### Phase 4: Comparison (`_T03_tutorial_comparisons`)

Contrast two experiments -- here, individual vs. pairwise:

```bash
bash experiments/_tutorials/_T03_tutorial_comparisons/bash/_T01_tutorial_evals_individual-vs-_T02_tutorial_evals_pairwise/run_comparison.sh
```

**What this does:**
- Reads the two experiment names from the directory name (`{exp1}-vs-{exp2}`).
- Finds the most recent aggregated performance data for each and contrasts them with propagated error bars.
- Emits the figure(s) named in `_T03_tutorial_comparisons/config.yaml`:

```yaml
figures_to_produce: ["performance_contrast_grouped"]
```

To compare different experiments, add a directory named `{exp1}-vs-{exp2}/` with a `00-performance_contrast.sh` (and a `run_comparison.sh` entry point).

### Summary

| Phase | Tutorial Directory | Entry point | Purpose |
|-------|-------------------|-------------|---------|
| **1. Data Gen** | `_T00_tutorial_data_gen/` | `00_sweep/run_sweep.sh` | Generate model outputs per dataset |
| **2a. Eval (IND)** | `_T01_tutorial_evals_individual/` | `a00_sweep/run_sweep.sh` | Individual-format recognition |
| **2b. Eval (PW)** | `_T02_tutorial_evals_pairwise/` | `a00_sweep/run_sweep.sh` | Pairwise-format recognition |
| **3. Analysis** | `_T01.../`, `_T02.../` | `b00_analysis/run_all_analyses.sh` | Visualize and analyze results |
| **4. Comparison** | `_T03_tutorial_comparisons/` | `bash/<exp1>-vs-<exp2>/run_comparison.sh` | Contrast two experiments |

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

## Supported Providers

The framework reaches 50+ models through the providers below. Models are referenced by short names and organized into predefined **model sets** (e.g. `tutorial`, `dr`, `eval_cot-r`) -- use `-set <name>` in any command to reference a set. Run `srf-list-models` to see everything available.

| Provider | Env var | Notes |
|----------|---------|-------|
| **OpenAI** | `OPENAI_API_KEY` | GPT-4o / 4.1 / 5 families, o3 reasoning models |
| **Anthropic** | `ANTHROPIC_API_KEY` | Claude Haiku / Sonnet / Opus |
| **Google** | `GOOGLE_API_KEY` | Gemini 2.0 / 2.5 (flash, flash-lite, pro) |
| **Together AI** | `TOGETHER_API_KEY` | Open-weight hosting: Llama, Qwen, DeepSeek, Kimi, GLM, gpt-oss, MiniMax, Gemma |
| **Fireworks** | `FIREWORKS_API_KEY` | Additional open-weight hosting |

xAI's Grok models are reached through an OpenAI-compatible endpoint (configured via `INSPECT_MODELS_OPENAI_GROK_*` environment variables) rather than a dedicated provider key.

Many models also have `-thinking` variants for chain-of-thought reasoning evaluations.

---

## Project Structure

```
self-rec-framework/
├── self_rec_framework/              # Core Python package
│   ├── src/
│   │   ├── inspect/                 # Inspect AI integration
│   │   │   ├── tasks.py             # Task definitions (comparison, conversational)
│   │   │   ├── config.py            # ExperimentConfig dataclass
│   │   │   ├── scorer.py            # Scoring logic (text parsing + logprobs)
│   │   │   └── data.py              # Dataset loading with position swapping
│   │   ├── data_generation/
│   │   │   ├── data_loading/        # Dataset loaders (WikiSum, PKU, ShareGPT, BigCodeBench)
│   │   │   └── procedural_editing/  # Treatment framework (typos, capitalization)
│   │   ├── helpers/
│   │   │   ├── model_names.py       # Model name / token-cap / context-window mappings
│   │   │   ├── model_sets.py        # Predefined model groupings
│   │   │   ├── constants.py         # UUID namespace, other constants
│   │   │   └── utils.py             # File I/O, path helpers
│   │   └── core_prompts/            # Prompt management
│   └── scripts/                     # srf-* entry points
│       ├── gen/                     # Data generation
│       ├── eval/                    # Evaluation sweep runner
│       ├── analysis/                # Analysis & visualization
│       └── admin/                   # Batch / model admin helpers
│
├── experiments/
│   └── _tutorials/                  # Tutorial experiments
│       ├── _T00_tutorial_data_gen/          # Phase 1: data generation
│       ├── _T01_tutorial_evals_individual/  # Phase 2: IND-format evaluation + analysis
│       ├── _T02_tutorial_evals_pairwise/    # Phase 2: PW-format evaluation + analysis
│       └── _T03_tutorial_comparisons/       # Phase 4: experiment comparison
│
├── data/
│   ├── input/                       # Input datasets (+ generated outputs)
│   ├── results/                     # Evaluation results (Inspect eval logs)
│   └── analysis/                    # Generated analysis outputs (plots, CSVs)
│
├── scripts/
│   └── utils/
│       ├── run.sh                   # Runner: execute every script in a directory
│       └── load_config.sh           # Loader: export config.yaml values for bash scripts
│
├── pyproject.toml                   # Dependencies, package config, srf-* entry points
├── .env                             # API keys (not in git)
└── CLAUDE.md                        # AI assistant reference guide
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
