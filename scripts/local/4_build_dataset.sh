#!/bin/bash
set -euo pipefail

echo "===== Build Dataset ====="
echo "Date: $(date)"
echo "========================="

mkdir -p logs

uv run python scripts/4_build_dataset.py \
    --sam3-dir "${SAM3_DIR:-data/sam3_outputs}" \
    --annotations-dir "${ANNOTATIONS_DIR:-data/annotations}" \
    --output-dir "${OUTPUT_DIR:-data/aligned}"

echo "===== Done at $(date) ====="
