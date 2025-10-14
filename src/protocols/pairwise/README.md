# Pairwise Self-Recognition Tasks

Tests whether models can identify their own outputs when shown two alternatives - one from the model being evaluated, one from a different model.

## Quick Start

```bash
inspect eval protocols/pairwise/tasks.py@comparison_self_recognition \
    --model anthropic/claude-3-5-sonnet-20241022 \
    -T model_name=anthropic--claude-3-5-sonnet-20241022 \
    -T model_generation_string=simple_config \
    -T alternative_model_name=gpt4 \
    -T alternative_model_generation_string=simple_config \
    -T dataset_name=cnn

inspect eval protocols/pairwise/tasks.py@conversational_self_recognition \
    --model fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct \
    -T model_name=gpt4 \
    -T model_generation_string=pirate_config \
    -T alternative_model_name=anthropic--claude-3-5-sonnet-20241022 \
    -T alternative_model_generation_string=pirate_config \
    -T dataset_name=cnn
```
