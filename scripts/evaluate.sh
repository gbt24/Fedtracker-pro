#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/default.yaml}"

python experiments/exp_ablation.py --config "${CONFIG_PATH}"

echo "Evaluation (ablation) run finished."
