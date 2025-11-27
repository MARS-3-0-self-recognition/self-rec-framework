# Self-Recognition Framework: Comprehensive Codebase Overview

**Project Period:** October 24, 2025 - November 19, 2025
**Framework Version:** Production-ready research infrastructure
**Purpose:** Comprehensive framework for studying self-recognition capabilities in large language models

---

## Executive Summary

This codebase provides a complete research infrastructure for investigating self-recognition in language models—the ability of models to identify their own generated outputs when compared to outputs from other models or modified versions. The framework supports end-to-end experimental workflows from data generation through statistical analysis, with robust batch processing, error handling, and comprehensive visualization capabilities.

---

## 1. Core Research Capabilities

### 1.1 Experiment Variants

The framework supports a comprehensive matrix of experimental conditions:

**Tagging Strategies:**
- **AT (Assistant Tags)**: Outputs labeled as coming from an assistant
- **UT (User Tags)**: Outputs labeled as coming from a user

**Evaluation Formats:**
- **PW (Pairwise)**: Models compare two outputs side-by-side
- **IND (Individual)**: Models evaluate outputs individually

**Interaction Styles:**
- **C (Conversation)**: Conversational/chat format
- **Q (Query)**: Simple query format

**Task Types:**
- **Rec (Recognition)**: Models identify which output is their own
- **Pref (Preference)**: Models express preference between outputs
  - **Pref-N**: Neutral preference ("Which one do you prefer?")
  - **Pref-S**: Submission preference ("Which one would you submit?")
  - **Pref-Q**: Quality preference ("Which one is higher quality?")

**Priming Conditions:**
- **Pr (Primed)**: Includes priming context about the task
- **NPr (Not Primed)**: No priming provided

**Total Experimental Configurations:** 15+ distinct experiment types configured and operational

### 1.2 Model Support

**Multi-Provider Integration:**
- **OpenAI**: GPT-4o, GPT-4.1, GPT-4.1-mini, GPT-4o-mini, GPT-5.1
- **Anthropic**: Claude Haiku-3.5, Sonnet-3.7, Sonnet-4.5, Opus-4.1
- **Google**: Gemini 2.0-flash-lite, 2.0-flash, 2.5-flash, 2.5-pro
- **Together AI** (hosting multiple open-source families):
  - **Llama** (Meta): ll-3.1-8b, ll-3.1-70b, ll-3.1-405b
  - **Qwen** (Alibaba): qwen-2.5-7b, qwen-2.5-72b, qwen-3.0-80b
  - **DeepSeek**: deepseek-3.0, deepseek-3.1

**Total Models Supported:** 19+ models across 4 major providers

---

## 2. Data Generation and Management Infrastructure

### 2.1 Dataset Loaders

**Implemented Loaders:**
- **WikiSum**: Wikipedia summarization dataset with sample range support
- **PKU-SafeRLHF**: Peking University safety dataset with safety mismatch filtering
- **ShareGPT52K**: ~52K conversations from ShareGPT platform with HTML cleaning

**Features:**
- Standardized UUID-based `input.json` format for all datasets
- Safety features to prevent accidental full dataset downloads
- Flexible sample selection (--num_samples or --range)
- Automatic data validation and deduplication
- Support for multiple data formats (JSON array, JSONL)

### 2.2 Procedural Editing Framework

**Treatment Types:**
- **Capitalization Treatments**: Systematic capitalization changes (S1-S4 strength levels)
- **Typo Treatments**: Realistic typo introduction (S1-S4 strength levels)
  - Keyboard-adjacent character substitutions (e.g., 'a' → 's' or 'q' on QWERTY)
  - Even distribution across text
  - Case preservation in substitutions

**Features:**
- Modular, extensible treatment system
- Precise parameter control
- Strength-based presets for reproducibility
- Unified treatment application interface

### 2.3 Data Generation Pipeline

**Capabilities:**
- Flexible generation parameters (temperature, max_tokens, top_p, top_k, seed, stop_seqs)
- Optional system prompts in generation configs
- Batch data generation with parallel processing
- Automatic dataset discovery and treatment strength detection
- Smart overwrite/skip logic for existing data

---

## 3. Experiment Execution System

### 3.1 Core Experiment Runner

**Features:**
- Unified `ExperimentConfig` for all experiment types
- Hierarchical prompt system with flexible resolution
- Automatic prompt building from YAML templates
- Support for both pairwise and individual evaluation protocols
- Integration with Inspect AI evaluation framework

### 3.2 Sweep Execution System

**Capabilities:**
- **Parallel Execution**: Native parallel task execution with configurable workers (default: 8)
- **Batch Mode Support**: Integration with provider batch APIs for 50% cost savings
- **Smart Overwrite Logic**:
  - Auto-detects failed/cancelled/error evaluations and re-runs them
  - Skips successful evaluations by default
  - Handles incomplete logs (success status but no results)
  - Manages orphaned batch jobs
- **Treatment Type Support**:
  - `other_models`: Compare models against each other
  - `caps`: Compare models against capitalization treatments
  - `typos`: Compare models against typo treatments
- **Provider-Specific Handling**:
  - Automatic exclusion of incompatible models from batch mode (GPT-5.1, Google Gemini)
  - Fallback to regular API calls when batch mode fails
  - Proper handling of Together AI sub-providers (Llama, Qwen, DeepSeek)

**Error Handling:**
- Confirmation prompts before long-running sweeps
- Status indicators during task building (⊘ skip, ↻ re-run, ✓ new, ✗ missing)
- Comprehensive error reporting and recovery

### 3.3 Batch Job Management

**Tools:**
- `scripts/list_active_batches.py`: Monitor active batch jobs across providers
- `scripts/cancel_all_batches.py`: Cancel all active batch jobs
- Automatic batch job tracking and recovery

---

## 4. Analysis and Visualization Tools

### 4.1 Primary Analysis Scripts

#### `analyze_pairwise_results.py`
**Purpose:** Comprehensive analysis of pairwise self-recognition experiment results

**Outputs:**
- Accuracy pivot tables (CSV)
- Accuracy heatmaps with provider boundaries
- Asymmetry analysis (pivot - pivot.T) showing evaluator vs evaluatee differences
- Row vs column comparison plots
- Evaluator performance bar charts with model family colors
- Summary statistics with significance testing

**Features:**
- Canonical model ordering (by provider, then by strength)
- Visual provider boundaries (thicker lines between model families)
- Statistical significance markers
- Model family color coding (OpenAI=green, Anthropic=red-orange, Google=yellow, etc.)

#### `analyze_preference_agreement.py`
**Purpose:** Analyze agreement between models on quality assessments in preference tasks

**Agreement Metric:** `1 - abs(A_ij - (1 - A_ji))`
- Where A_ij = how often model i prefers its own output over model j's output
- Higher values indicate better agreement on quality

**Outputs:**
- Agreement matrix (CSV)
- Agreement heatmap (green=high agreement, red=low agreement)
- Evaluator agreement performance charts
- Summary statistics

#### `compare_experiments.py`
**Purpose:** Statistical comparison between two experiments

**Features:**
- Paired t-tests for each (evaluator, treatment) cell
- Overall statistical test across all samples
- Difference heatmaps with significance markers (bold values for p < 0.05)
- Cross-dataset comparison support
- Evaluator difference performance charts
- Comprehensive summary statistics

**Outputs:**
- Accuracy difference matrix (CSV)
- P-values matrix (CSV)
- Difference heatmap with significance indicators
- Summary statistics

#### `compare_accuracy_agreement.py`
**Purpose:** Combine accuracy and agreement metrics

**Features:**
- Combines recognition accuracy with preference agreement scores
- Normalized combined scores: `(accuracy + agreement) - 1`
- Range: [-1, 1] where 1 = green (best), 0 = red (worst)
- Evaluator performance charts with significance testing

#### `compare_recognition_preference.py`
**Purpose:** Compare recognition vs preference experiment results

**Features:**
- 4-color heatmap based on difference sign and recognition baseline:
  - **Green**: Preference > Recognition AND Recognition > 0.5
  - **Blue**: Preference > Recognition AND Recognition < 0.5
  - **Red**: Preference < Recognition AND Recognition < 0.5
  - **Orange**: Preference < Recognition AND Recognition > 0.5
- Statistical significance testing (paired t-tests)
- Color categorization bar chart with significant/non-significant breakdown
- Comprehensive statistics for each color category

### 4.2 Visualization Features

**Common Features Across All Visualizations:**
- **Model Family Colors**: Consistent color scheme based on model provider
  - OpenAI: Green shades (weaker→stronger)
  - Anthropic: Red-orange shades (weaker→stronger)
  - Google: Yellow shades (weaker→stronger)
  - Llama: Blue shades
  - Qwen: Purple shades
  - DeepSeek: Red shades
- **Provider Boundaries**: Thicker black lines separating model families
- **Statistical Significance**: Markers (***, **, *) for significant results
- **Canonical Ordering**: Consistent model ordering across all visualizations
- **High-Quality Outputs**: 300 DPI PNG files with proper formatting

**Chart Types:**
- Heatmaps (accuracy, agreement, differences, asymmetry)
- Bar charts (evaluator performance, color categorization)
- Comparison plots (row vs column, evaluator differences)

---

## 5. Technical Infrastructure

### 5.1 Code Organization

**Modular Architecture:**
```
src/
├── core_prompts/          # Hierarchical prompt templates (YAML)
├── data_generation/
│   ├── data_loading/      # Dataset loaders (WikiSum, PKU-SafeRLHF, ShareGPT)
│   └── procedural_editing/ # Treatment application (caps, typos)
├── inspect/              # Unified evaluation framework
│   ├── config.py         # Experiment configuration and prompt building
│   └── tasks.py          # Task function definitions
└── helpers/              # Utility functions

experiments/
├── 00_data_gen/          # Data generation scripts
├── 01-15_*/              # Experiment configurations (15+ variants)
└── scripts/              # Analysis and execution scripts
```

### 5.2 Configuration System

**Hierarchical Prompt System:**
- Base prompts in `src/core_prompts/prompts.yaml`
- Dataset-specific prompts in `data/input/{dataset}/prompts.yaml`
- Flexible resolution with wildcards (e.g., `AT.All`, `UT.C.All`)
- Dynamic prompt building based on experiment configuration

**Experiment Configuration:**
- YAML-based configuration files
- Automatic prompt resolution and building
- Unified `ExperimentConfig` dataclass
- Backward compatibility with legacy configs

### 5.3 Data Organization

**Standardized Path Structure:**
```
data/
├── input/                # Input datasets and generated data
│   └── {dataset}/
│       └── {subset}/
│           ├── {model}/data.json
│           └── prompts.yaml
└── results/              # Experiment results
    └── {dataset}/
        └── {subset}/
            └── {experiment}/
                └── *.eval files

data/analysis/            # Analysis outputs
└── {dataset}/
    └── {subset}/
        ├── {experiment}/ # Individual experiment analysis
        └── comparisons/  # Cross-experiment comparisons
```

---

## 6. Key Achievements and Features

### 6.1 Research Infrastructure

✅ **Complete Experimental Framework**: 15+ experiment types fully configured and operational
✅ **Multi-Provider Support**: Seamless integration with 4 major LLM providers
✅ **19+ Models**: Comprehensive coverage across model families and capabilities
✅ **3 Datasets**: WikiSum, PKU-SafeRLHF, ShareGPT52K with standardized formats
✅ **Procedural Editing**: Systematic treatment generation (caps, typos) with strength controls

### 6.2 Analysis Capabilities

✅ **Statistical Testing**: Paired t-tests, one-sample t-tests, overall significance tests
✅ **Comprehensive Visualizations**: 10+ visualization types with consistent styling
✅ **Cross-Experiment Comparison**: Same-dataset and cross-dataset comparison support
✅ **Agreement Metrics**: Novel agreement scoring for preference tasks
✅ **Asymmetry Analysis**: Evaluator vs evaluatee role analysis

### 6.3 Operational Excellence

✅ **Batch Processing**: Cost-effective batch API integration (50% savings)
✅ **Parallel Execution**: Configurable parallelism for faster execution
✅ **Error Recovery**: Smart overwrite logic and orphaned job handling
✅ **Robustness**: Provider-specific workarounds and compatibility handling
✅ **Documentation**: Comprehensive READMEs, figure descriptions, and inline documentation

### 6.4 Code Quality

✅ **Modular Design**: Clean separation of concerns, reusable components
✅ **Type Safety**: Type hints throughout, dataclass-based configuration
✅ **Error Handling**: Comprehensive exception handling and user feedback
✅ **Testing Infrastructure**: Pre-commit hooks, code quality checks
✅ **Maintainability**: Well-organized codebase with clear naming conventions

---

## 7. Research Outputs

### 7.1 Generated Artifacts

**Per Experiment:**
- Accuracy pivot tables (CSV)
- Accuracy heatmaps (PNG)
- Asymmetry matrices and heatmaps
- Row vs column comparison plots
- Evaluator performance charts
- Summary statistics (TXT)

**Per Comparison:**
- Difference matrices (CSV)
- P-values matrices (CSV)
- Difference heatmaps with significance
- Combined metric visualizations
- Color categorization charts
- Comprehensive comparison statistics

**Total Output Types:** 15+ distinct analysis outputs per experiment/comparison

### 7.2 Data Products

- Standardized input datasets with UUID-based identification
- Generated model outputs across multiple datasets
- Treatment datasets (caps, typos) with strength levels
- Evaluation logs with full sample-level data
- Analysis-ready pivot tables and matrices

---

## 8. Technical Specifications

### 8.1 Dependencies

**Core Libraries:**
- `inspect-ai`: Evaluation framework
- `pandas`, `numpy`: Data manipulation
- `matplotlib`, `seaborn`: Visualization
- `scipy`: Statistical testing
- `pyyaml`: Configuration management
- `huggingface-hub`: Dataset loading

**Development Tools:**
- `uv`: Fast Python package manager
- `pre-commit`: Code quality hooks
- `pytest`: Testing framework (configured)

### 8.2 Performance Characteristics

- **Parallel Execution**: Configurable workers (default: 8, supports 2-4x CPU cores)
- **Batch Mode**: 50% cost savings via provider batch APIs
- **Smart Caching**: Skip existing evaluations/data by default
- **Efficient Processing**: Sample-level parallelism via Inspect AI

### 8.3 Scalability

- **Dataset Size**: Supports datasets from hundreds to tens of thousands of samples
- **Model Count**: Tested with 19+ models simultaneously
- **Experiment Scale**: Handles 300+ evaluation files per experiment
- **Comparison Scale**: Supports cross-dataset comparisons across multiple experiments

---

## 9. Documentation and Usability

### 9.1 User Documentation

- **README.md**: Setup and usage instructions
- **TIMESHEET_REPORT.md**: Development history and glossary
- **Figure Descriptions**: Detailed descriptions for all visualizations
- **Inline Documentation**: Comprehensive docstrings and comments
- **Example Scripts**: Bash scripts demonstrating common workflows

### 9.2 Code Documentation

- **Type Hints**: Throughout codebase for IDE support
- **Docstrings**: Function and class documentation
- **Comments**: Explanatory comments for complex logic
- **Naming Conventions**: Clear, descriptive variable and function names

---

## 10. Research Applications

### 10.1 Primary Research Questions

The framework enables investigation of:
1. **Self-Recognition Capability**: Can models identify their own outputs?
2. **Task Format Effects**: How do pairwise vs individual, query vs conversation formats affect recognition?
3. **Priming Effects**: Does task priming improve self-recognition?
4. **Preference vs Recognition**: How do preference judgments differ from recognition judgments?
5. **Model Agreement**: How well do models agree on quality assessments?
6. **Asymmetry Patterns**: Are there systematic differences when models are evaluators vs evaluatees?
7. **Treatment Sensitivity**: How do models respond to procedural modifications (caps, typos)?

### 10.2 Experimental Design Support

- **Systematic Variation**: All experimental factors configurable
- **Reproducibility**: Seed control, standardized data formats
- **Statistical Rigor**: Built-in significance testing and proper sample alignment
- **Cross-Validation**: Support for cross-dataset comparisons

---

## 11. Innovation Highlights

### 11.1 Novel Metrics

- **Agreement Score**: `1 - abs(A_ij - (1 - A_ji))` for measuring model agreement on quality
- **Asymmetry Analysis**: Systematic evaluation of evaluator vs evaluatee role differences
- **Combined Metrics**: Integration of accuracy and agreement for comprehensive assessment

### 11.2 Technical Innovations

- **Hierarchical Prompt System**: Flexible, maintainable prompt management
- **Unified Treatment Framework**: Extensible system for procedural editing
- **Smart Batch Management**: Automatic detection and handling of batch job issues
- **Provider-Agnostic Design**: Seamless integration across multiple LLM providers

### 11.3 Analysis Innovations

- **4-Color Categorization**: Visual framework for understanding preference vs recognition differences
- **Cross-Dataset Comparison**: Novel capability for comparing experiments across datasets
- **Significance Integration**: Statistical testing embedded in all visualizations
- **Model Family Visualization**: Color-coded visualizations for easy model family identification

---

## 12. Deliverables Summary

### 12.1 Software Deliverables

✅ **Complete Research Framework**: End-to-end experimental pipeline
✅ **10+ Analysis Scripts**: Comprehensive analysis and visualization tools
✅ **15+ Experiment Configurations**: Ready-to-use experimental setups
✅ **3 Dataset Loaders**: WikiSum, PKU-SafeRLHF, ShareGPT52K
✅ **Procedural Editing System**: Caps and typo treatment generation
✅ **Batch Processing Infrastructure**: Cost-effective execution system

### 12.2 Documentation Deliverables

✅ **User Documentation**: Setup guides, usage examples
✅ **Code Documentation**: Comprehensive docstrings and comments
✅ **Figure Descriptions**: Detailed explanations of all visualizations
✅ **Development History**: Complete timesheet report with glossary

### 12.3 Research Infrastructure

✅ **Standardized Data Formats**: UUID-based input.json format
✅ **Reproducible Configurations**: YAML-based experiment configs
✅ **Statistical Analysis Tools**: Built-in significance testing
✅ **Visualization Suite**: 15+ visualization types with consistent styling

---

## Conclusion

This codebase represents a production-ready, comprehensive research infrastructure for studying self-recognition in large language models. It provides end-to-end support from data generation through statistical analysis, with robust error handling, cost-effective batch processing, and extensive visualization capabilities. The framework has been successfully used to conduct 15+ distinct experiment types across 19+ models and 3 datasets, generating hundreds of evaluation files and comprehensive analysis outputs.

The infrastructure is designed for extensibility, maintainability, and reproducibility, making it suitable for both current research needs and future expansion. All code follows best practices with type hints, comprehensive documentation, and modular design principles.

---

**Framework Status:** Production-ready, actively maintained
**Code Quality:** High (type hints, documentation, error handling)
**Research Readiness:** Complete (all tools operational, tested, documented)
**Extensibility:** High (modular design, clear interfaces, comprehensive examples)
