#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

scripts/run.sh $SCRIPT_DIR/bigcodebench $SCRIPT_DIR/wikisum $SCRIPT_DIR/sharegpt $SCRIPT_DIR/pku_saferlhf $SCRIPT_DIR/_inter-dataset
