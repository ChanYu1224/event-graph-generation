#!/bin/bash
set -euo pipefail

VIDEO_PATH="${VIDEO_PATH:?VIDEO_PATH must be set}"
CHECKPOINT="${CHECKPOINT:-data/checkpoints/best.pt}"
OUTPUT="${OUTPUT:-output/event_graph.json}"
CONFIG="${CONFIG:-configs/inference.yaml}"

echo "===== Event Graph Inference ====="
echo "Video:      ${VIDEO_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT}"
echo "Config:     ${CONFIG}"
echo "=================================="

mkdir -p output

uv run python scripts/7_run_inference.py \
    --video "$VIDEO_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --config "$CONFIG"

echo "===== Done at $(date) ====="
