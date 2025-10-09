#!/bin/bash
# Simple runner for CSV-based 2T experiments
#
# Usage:
#   ./run_csv_experiment.sh data/experiments/pairwise/two-turn/typos/S1/claude-3-5-haiku-20241022.csv
#
# Or with article text:
#   ./run_csv_experiment.sh path/to/data.csv "Article text here..."

CSV_PATH=$1
ARTICLE_TEXT=${2:-""}
MODEL="anthropic/claude-3-5-haiku-20241022"
TASK="two_turn_summary_recognition_csv"

if [ -z "$CSV_PATH" ]; then
    echo "Error: Please provide CSV path as first argument"
    echo "Usage: ./run_csv_experiment.sh <csv_path> [article_text]"
    exit 1
fi

if [ ! -f "$CSV_PATH" ]; then
    echo "Error: CSV file not found: $CSV_PATH"
    exit 1
fi

echo "=========================================="
echo "Running 2T Experiment from CSV"
echo "=========================================="
echo "CSV: $CSV_PATH"
echo "Model: $MODEL"
echo "Task: $TASK"
echo "=========================================="
echo

# Build inspect eval command
CMD="inspect eval protocols/pairwise/tasks.py@$TASK --model $MODEL -T csv_path=$CSV_PATH"

if [ -n "$ARTICLE_TEXT" ]; then
    CMD="$CMD -T article_text=\"$ARTICLE_TEXT\""
fi

echo "Command: $CMD"
echo
eval $CMD
