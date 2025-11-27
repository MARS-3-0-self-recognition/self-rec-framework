# Base Generation Tasks

Obtains rollouts using inspect.

## Quick Start

NOTE: `model_name`s are short, see `helpers/model_names.py`.

One can run the underlying task as follows:

```bash
uv run inspect eval src/data_gen/gen.py@base_generation \
    -T model_name=3-5-haiku \
    -T model_generation_string=simple_config \
    -T pairwise_config_string=summarisation \
    -T dataset_name=cnn_debug
```

To automatically run the eval and parse the output back to json instead run the python file as a script
```bash
uv run src/data_gen/gen.py \
    --model_name=3-5-haiku \
    --model_generation_string=simple_config \
    --pairwise_config_string=summarisation \
    --dataset_name=cnn_debug
```
