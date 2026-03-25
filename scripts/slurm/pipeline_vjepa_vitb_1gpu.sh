#!/bin/bash
#SBATCH --job-name=vjepa-vitb-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_vjepa_vitb_1gpu_%j.log

set -euo pipefail
mkdir -p logs

echo "===== Step 1: Build ViT-B Dataset ====="
rm -rf data/vjepa_aligned_v21_vitb/samples data/vjepa_aligned_v21_vitb/splits

uv run python scripts/3_build_dataset.py \
    --features-dir data/vjepa_features_v21_vitb \
    --output-dir data/vjepa_aligned_v21_vitb \
    --annotations-dir data/annotations \
    --vocab configs/vocab.yaml \
    --actions configs/actions.yaml

echo ""
echo "=== Dataset stats ==="
wc -l data/vjepa_aligned_v21_vitb/splits/*.txt
echo ""

echo "===== Step 2: Train (1-GPU) ====="
uv run python scripts/4_train.py \
    --config configs/vjepa_training.yaml \
    --override configs/experiment/vjepa_vitb.yaml
