#!/bin/bash
# Entry point: run every sweep script in this directory via the shared runner.
# scripts/utils/run.sh skips this file (so it doesn't recurse); each sweep
# script reads model_names from ../../config.yaml. Run from the repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash scripts/utils/run.sh "$SCRIPT_DIR"
