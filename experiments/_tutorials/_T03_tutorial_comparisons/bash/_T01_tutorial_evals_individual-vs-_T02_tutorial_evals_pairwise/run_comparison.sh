#!/bin/bash
# Entry point: run the comparison script(s) in this directory via the shared
# runner. scripts/utils/run.sh skips this file (run_*.sh) and config.sh; the
# comparison script reads model_names from config.yaml.
# Run from the self-rec-research repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash scripts/utils/run.sh "$SCRIPT_DIR"
