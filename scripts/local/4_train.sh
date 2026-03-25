#!/bin/bash
set -euo pipefail

echo "===== V-JEPA Pipeline Training ====="
echo "Date: $(date)"
echo "==================================="

mkdir -p logs

CONFIG="${CONFIG:-configs/vjepa_training.yaml}"
OVERRIDE="${OVERRIDE:-}"

CMD=(uv run python scripts/4_train.py --config "$CONFIG")
if [ -n "$OVERRIDE" ]; then
    CMD+=(--override "$OVERRIDE")
fi

"${CMD[@]}"

echo "===== Done at $(date) ====="
