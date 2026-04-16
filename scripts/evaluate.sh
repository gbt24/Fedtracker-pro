#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

python -m experiments.exp_ablation --config "${CONFIG_PATH}"

echo "Evaluation (ablation) run finished."
