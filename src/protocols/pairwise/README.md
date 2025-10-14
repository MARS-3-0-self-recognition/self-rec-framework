# Pairwise Self-Recognition Tasks

Tests whether models can identify their own outputs when shown two alternatives - one from the model being evaluated, one from a different model.

## Quick Start

NOTE: `model_name`s are short, see `helpers/model_names.py`.

```bash
inspect eval src/protocols/pairwise/tasks.py@comparison_self_recognition \
    -T model_name=3-5-sonnet \
    -T model_generation_string=simple_config \
    -T alternative_model_name=3-5-sonnet \
    -T alternative_model_generation_string=pirate_style_config \
    -T pairwise_config_string=summarisation \
    -T dataset_name=cnn_debug

inspect eval protocols/pairwise/tasks.py@conversational_self_recognition \
    -T model_name=Qwen3-8B \
    -T model_generation_string=pirate_config \
    -T alternative_model_name=3-5-sonnet \
    -T alternative_model_generation_string=pirate_config \
    -T dataset_name=toy_lennie \
    -T pairwise_config_string=qa
```
