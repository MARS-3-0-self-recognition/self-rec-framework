# Send batch request
uv run -m data_gen.simple_response.send \
    --model anthropic/claude-3-5-haiku-20241022 \
    --data cnn_debug \
    --generation_config data_gen/simple_response/config/simple_config.yaml

# Check and download results
uv run -m data_gen.simple_response.receive \
    --model anthropic/claude-3-5-haiku-20241022 \
    --data cnn_debug \
    --generation_config data_gen/simple_response/config/simple_config.yaml
