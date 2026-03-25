#!/bin/bash
set -euo pipefail

echo "===== Build V-JEPA Dataset ====="
echo "Date: $(date)"
echo "================================="

mkdir -p logs

uv run python scripts/3_build_dataset.py \
    --features-dir "${FEATURES_DIR:-data/vjepa_features}" \
    --output-dir "${OUTPUT_DIR:-data/vjepa_aligned}" \
    --annotations-dir "${ANNOTATIONS_DIR:-data/annotations}" \
    --vocab "${VOCAB:-configs/vocab.yaml}" \
    --actions "${ACTIONS:-configs/actions.yaml}"

echo "===== Done at $(date) ====="
