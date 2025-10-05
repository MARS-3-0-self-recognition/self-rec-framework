# Data Generation

Batch inference scripts for generating model outputs across multiple APIs.

## Setup

Create a .env file in the project root:

```bash
OPENAI_API_KEY="your-key-here"
ANTHROPIC_API_KEY="your-key-here"
TOGETHER_API_KEY="your-key-here"
FIREWORKS_API_KEY="your-key-here"
FIREWORKS_ACCOUNT_ID="your-account-here"
```

Install dependencies:
```bash
pip install openai anthropic together fireworks-ai python-dotenv pyyaml requests
```


## Quick Start

```bash
# Send batch request
python -m data_gen.simple_response.send \
    --model fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct \
    --data cnn_debug \
    --generation_config data_gen/simple_response/config/simple_config.yaml

# Check and download results
python -m data_gen.simple_response.receive \
    --model fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct \
    --data cnn_debug \
    --generation_config data_gen/simple_response/config/simple_config.yaml
```

## Configuration

Create a YAML config file with generation settings:

```yaml
temperature: 1.0
output_type: summaries  # or "answers"
system_prompt: You are an article summariser.
prompt: |
  Please summarise the following article:
  
  {request}
```

**Fields:**
- `temperature`: Generation temperature
- `output_type`: `"summaries"` or `"answers"` (determines input/output filenames)
- `system_prompt`: System message
- `prompt`: User prompt template (use `{request}` placeholder)

Generation name is derived from config filename (e.g., `simple_config.yaml` → `simple_config`).

## Workflow

### 1. Send Request
Creates batch job and saves metadata:

```
data/{dataset}/gpt-4/
└── simple_config/
    ├── request_data.jsonl      # Local requests
    └── request_status.json     # Batch metadata
```

### 2. Receive Results
Downloads and transforms results:

```
data/{dataset}/gpt-4/
├── simple_config.json          # Final output {uuid: text}
└── simple_config/
    ├── request_data.jsonl
    ├── request_status.json     # Updated with received=true
    └── raw.jsonl               # Downloaded results
```

## Supported APIs

**Model format:** `{api}/{model_path}`

- OpenAI: `fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct`, `openai/gpt-3.5-turbo`
- Anthropic: `anthropic/claude-sonnet-4-5`, `anthropic/claude-opus-4`
- Fireworks: `fireworks/accounts/fireworks/models/llama-v3p3-70b-instruct`
- Together: `together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo`

Model path with `/` becomes directory name with `--` (e.g., `openai--gpt-4`).

## Error Handling

If batch fails, `request_status.json` is moved to `request_status.failed.json` with `failed=true`.

Re-running `send` with existing `request_status.json` throws an error. Delete or move the status file to retry.

## Examples

```bash
# Summarization with different temperatures
python -m data_gen.simple_response.send \
    --model anthropic/claude-sonnet-4-5 \
    --data news_articles \
    --generation_config data_gen/simple_response/config/temp0_summary.yaml

python -m data_gen.simple_response.send \
    --model anthropic/claude-sonnet-4-5 \
    --data news_articles \
    --generation_config data_gen/simple_response/config/temp1_summary.yaml

# Question answering
python -m data_gen.simple_response.send \
    --model together/meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo \
    --data trivia_questions \
    --generation_config data_gen/simple_response/config/qa_config.yaml

# Check results
python -m data_gen.simple_response.receive \
    --model anthropic/claude-sonnet-4-5 \
    --data news_articles \
    --generation_config data_gen/simple_response/config/temp0_summary.yaml
```

## API Keys

Set environment variables:
- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `FIREWORKS_API_KEY`
- `TOGETHER_API_KEY`