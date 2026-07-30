#!/bin/bash
# Entry point: run every sweep script in this directory via the shared runner.
# scripts/utils/run.sh skips this file (so it doesn't recurse) and config.sh;
# each sweep script reads its parameters from ../../config.yaml.
# Run from the self-rec-research repo root.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
bash scripts/utils/run.sh "$SCRIPT_DIR"
