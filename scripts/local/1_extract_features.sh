#!/bin/bash
set -euo pipefail

echo "===== V-JEPA Feature Extraction ====="
echo "Date: $(date)"
echo "======================================"

mkdir -p logs

VIDEO_DIR="${VIDEO_DIR:-data/resized/trial}"
OUTPUT_DIR="${OUTPUT_DIR:-data/vjepa_features}"

uv run python scripts/1_extract_features.py \
    --config "${CONFIG:-configs/vjepa.yaml}" \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "===== Done at $(date) ====="
