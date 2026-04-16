#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

python experiments/exp_baseline.py --config "${CONFIG_PATH}"

echo "Training baseline run finished."
