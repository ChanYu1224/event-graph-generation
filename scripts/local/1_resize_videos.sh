#!/bin/bash
set -euo pipefail

echo "===== Resize Videos ====="
echo "Date: $(date)"
echo "========================="

mkdir -p logs

uv run python scripts/1_resize_videos.py \
    --input-dir "${INPUT_DIR:-data/mp4}" \
    --output-dir "${OUTPUT_DIR:-data/resized}" \
    --max-side "${MAX_SIDE:-1008}" \
    --resume

echo "===== Done at $(date) ====="
