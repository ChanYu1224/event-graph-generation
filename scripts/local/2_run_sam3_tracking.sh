#!/bin/bash
set -euo pipefail

echo "===== SAM 3 Tracking ====="
echo "Date: $(date)"
echo "=========================="

mkdir -p logs

uv run python scripts/2_run_sam3_tracking.py \
    --config configs/sam3.yaml \
    --video-dir "${VIDEO_DIR:-data/mp4}" \
    --output-dir "${OUTPUT_DIR:-data/sam3_outputs}" \
    --resume

echo "===== Done at $(date) ====="
