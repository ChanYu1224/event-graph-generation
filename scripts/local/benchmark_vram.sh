#!/bin/bash
set -euo pipefail

echo "=== VRAM Benchmark ==="
echo "Date: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || true
echo ""

uv run python scripts/benchmark_vram.py

echo "=== Done at $(date) ==="
