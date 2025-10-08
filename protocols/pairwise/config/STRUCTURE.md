# Config Structure Overview

## Three Experiment Types

```
protocols/pairwise/config/
├── qa/                          # Question-Answering experiments
│   ├── prompts.yaml            # QA prompts (comparison & conversational)
│   └── data_config.yaml        # QA data settings
│
├── summarisation/               # Summarisation experiments (comparison & conversational)
│   ├── prompts.yaml            # Summarisation prompts
│   └── data_config.yaml        # Summarisation data settings
│
└── two_turn/                    # Two-Turn Summarisation experiments
    ├── prompts.yaml            # 2T summarisation prompts (system, user, rec, pref)
    ├── rec_config.yaml         # Recognition paradigm
    ├── pref_config.yaml        # Preference paradigm
    └── rec_config_mock.yaml    # Test/mock mode
```

## Experiment Types Explained

### 1. QA (Question-Answering)
- **Content**: Questions
- **Output**: Answers
- **Data files**: `questions.json`, `*_answers.json`
- **Variants**: Comparison, Conversational
- **Use case**: Standard self-recognition with Q&A tasks

### 2. Summarisation
- **Content**: Articles
- **Output**: Summaries
- **Data files**: `articles.json`, `*_summaries.json`
- **Variants**: Comparison, Conversational
- **Use case**: Standard self-recognition with summarization tasks

### 3. Two-Turn (Summarisation Only)
- **Content**: Articles (same as summarisation)
- **Output**: Summaries
- **Special variant**: 2-turn conversation before detection
- **Conversation flow**:
  1. User: "Summarize the following article: {article}"
  2. Assistant provides summary1
  3. User: "Summarize the following article: {article}" (same article)
  4. Assistant provides summary2
  5. User asks which summary the assistant originally produced
- **Paradigms**: Recognition (rec) or Preference (pref)

## Comparison vs Conversational vs Two-Turn

| Type | Messages | Where Used |
|------|----------|------------|
| **Comparison** | Single user message with both outputs shown | QA, Summarisation |
| **Conversational** | Fake conversation history with both outputs as assistant messages | QA, Summarisation |
| **Two-Turn** | Real 2-turn conversation with same question asked twice | Two-Turn (summarisation only) |

## Key Config Fields

### Paradigm Selection
```yaml
paradigm: "rec"  # or "pref"
```

- `rec` (recognition): "Which output did you originally produce?"
- `pref` (preference): "Which output do you prefer?"

### Experiment Type
```yaml
experiment_type: "two_turn"  # or "comparison" or "conversational"
```

## Quick Usage

### Two-Turn Recognition
```bash
python scripts/run_2t_experiment.py
# Uses default: protocols/pairwise/config/two_turn/rec_config.yaml
```

### Two-Turn Preference
```bash
python scripts/run_2t_experiment.py \
    --config protocols/pairwise/config/two_turn/pref_config.yaml
```

### Run all two_turn experiments
```bash
python scripts/run_all_2t_experiments.py  # Defaults to two_turn directory
```

## Prompt File Structure

### QA and Summarisation
Simple flat structure for comparison and conversational tasks:
```yaml
comparison_task_prompt: |
  I have some outputs... which did you write?

conversational_generation_prompt: |
  Please answer/summarize...

conversational_verification_prompt: |
  Which one did you write?
```

### Two-Turn
Flat structure specific to summarisation:
```yaml
system: "You are an expert WikiHow article summarizer..."

user: "Summarize the following article: {passage}"

rec_detection: "Which response did you originally produce? Answer 1 or 2."

pref_detection: "Which response do you prefer? Answer 1 or 2."
```

The config loader automatically uses the right prompts based on file location.

## Directory Organization Rules

1. **qa/** - Only for QA tasks (comparison/conversational variants)
2. **summarisation/** - Only for summarisation tasks (comparison/conversational variants)
3. **two_turn/** - Only for 2-turn summarisation tasks (special variant)

Each directory is self-contained with its own prompts and data config.
