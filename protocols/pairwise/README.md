# Pairwise Self-Recognition Tasks

Tests whether models can identify their own outputs when shown two alternatives - one from the model being evaluated, one from a different model.

## Quick Start

```bash
inspect eval protocols/pairwise/tasks.py@comparison_summary_recognition \
    --model anthropic/claude-3-5-sonnet-20241022 \
    -T model_name=anthropic--claude-3-5-sonnet-20241022 \
    -T model_generation_string=simple_config \
    -T alternative_model_name=gpt4 \
    -T alternative_model_generation_string=simple_config \
    -T dataset_name=cnn

inspect eval protocols/pairwise/tasks.py@comparison_summary_recognition \
    --model fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct \
    -T model_name=gpt4 \
    -T model_generation_string=pirate_config \
    -T alternative_model_name=anthropic--claude-3-5-sonnet-20241022 \
    -T alternative_model_generation_string=pirate_config \
    -T dataset_name=cnn
```

## Task Types

**comparison**: Direct question - "Here are two outputs, which did you write?"

**Conversational**: Fake conversation history with both outputs as assistant messages, then ask which one was actually written by the model.

## Available Tasks

- `comparison_summary_recognition` - Article summarization
- `conversational_summary_recognition` - Article summarization (conversational)
- `comparison_qa_recognition` - Question answering
- `conversational_qa_recognition` - Question answering (conversational)

**Variants:**
- `comparison_summary_recognition_deterministic` - temp=0.0
- `conversational_summary_recognition_high_temp` - temp=1.0
- `comparison_summary_recognition_batch` - Batch processing
- `conversational_qa_recognition_batch` - Batch processing (conversational)

## Data Format

```
data/
└── {dataset_name}/
    ├── articles.json  # or questions.json
    ├── {model_name}/
    │   └── {generation_string}_summaries.json  # or answers.json
    └── {alternative_model_name}/
        └── {generation_string}_summaries.json
```

JSON files:
- Content: `{uuid: text}`
- Outputs: `{uuid: output_text}`

The loader creates 2 samples per UUID (swapping positions) to test both orderings.

## Usage Examples

```bash
# Standard evaluation
inspect eval protocols/pairwise/tasks.py@conversational_qa_recognition \
    --model anthropic/claude-3-5-sonnet \
    -T model_name=claude-3-5-sonnet \
    -T alternative_model_name=gpt-4 \
    -T dataset_name=open_ended_questions \
    -T model_generation_string=default \
    -T alternative_model_generation_string=temp0

# Batch mode
inspect eval protocols/pairwise/tasks.py@comparison_summary_recognition_batch \
    --model openai/gpt-4 \
    -T model_name=gpt-4 \
    -T alternative_model_name=claude-3-5-sonnet \
    -T dataset_name=news_articles \
    -T batch_size=100
```

## Adding New Content Types

1. Create `config/{type}.yaml` with prompts and field names
2. Add loader function in `config/__init__.py`
3. Add task functions in `tasks.py` using `comparison_self_recognition()` or `conversational_self_recognition()`
4. Export in `__init__.py`