#!/bin/bash
set -euo pipefail

echo "===== Event Decoder Training ====="
echo "Date: $(date)"
echo "==================================="

mkdir -p logs

CONFIG="${CONFIG:-configs/training.yaml}"
OVERRIDE="${OVERRIDE:-}"

CMD=(uv run python scripts/5_train.py --config "$CONFIG")
if [ -n "$OVERRIDE" ]; then
    CMD+=(--override "$OVERRIDE")
fi

"${CMD[@]}"

echo "===== Done at $(date) ====="
