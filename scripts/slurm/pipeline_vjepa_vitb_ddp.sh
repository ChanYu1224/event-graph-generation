#!/bin/bash
#SBATCH --job-name=vjepa-vitb-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_vjepa_vitb_ddp_%j.log

set -euo pipefail
mkdir -p logs

echo "===== Step 1: Build V-JEPA ViT-B Dataset ====="

uv run python scripts/4b_build_vjepa_dataset.py \
    --features-dir data/vjepa_features_v21_vitb \
    --output-dir data/vjepa_aligned_v21_vitb \
    --annotations-dir data/annotations \
    --vocab configs/vocab.yaml \
    --actions configs/actions.yaml

echo ""
echo "=== Dataset stats ==="
wc -l data/vjepa_aligned_v21_vitb/splits/*.txt
echo ""

echo "===== Step 2: DDP Training (4-GPU) ====="

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    scripts/5_train.py \
    --config configs/vjepa_training.yaml \
    --override configs/experiment/vjepa_vitb_ddp.yaml
