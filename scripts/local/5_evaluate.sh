#!/bin/bash
set -euo pipefail

echo "===== Event Decoder Evaluation ====="
echo "Date: $(date)"
echo "====================================="

mkdir -p logs

uv run python scripts/5_evaluate.py \
    --config "${CONFIG:-configs/training.yaml}" \
    --checkpoint "${CHECKPOINT:-data/checkpoints/best.pt}"

echo "===== Done at $(date) ====="
