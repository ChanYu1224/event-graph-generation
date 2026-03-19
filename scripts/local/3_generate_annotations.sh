#!/bin/bash
set -euo pipefail

echo "=== VLM Annotation ==="
echo "Date: $(date)"
echo "======================"

mkdir -p logs

uv run python scripts/3_generate_annotations.py --resume

echo "=== Done at $(date) ==="
