#!/bin/bash
# Step 2 & 3: Build dataset and submit training job
# Run this after V-JEPA extraction is complete
set -euo pipefail

echo "===== Step 2: Build V-JEPA Dataset ====="

# Clean previous dataset to rebuild with all videos
rm -rf data/vjepa_aligned/samples data/vjepa_aligned/splits

uv run python scripts/4b_build_vjepa_dataset.py \
    --annotations-dir data/annotations \
    --features-dir data/vjepa_features \
    --output-dir data/vjepa_aligned \
    --vocab configs/vocab.yaml \
    --actions configs/actions.yaml

echo ""
echo "=== Dataset stats ==="
wc -l data/vjepa_aligned/splits/*.txt
echo ""

echo "===== Step 3: Submit Training Job ====="
sbatch scripts/slurm/5_train_vjepa_full.sh
