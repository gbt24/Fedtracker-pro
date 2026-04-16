#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

python -m experiments.exp_baseline --config "${CONFIG_PATH}"

echo "Training baseline run finished."
