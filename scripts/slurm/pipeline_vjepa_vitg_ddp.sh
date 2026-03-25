#!/bin/bash
#SBATCH --job-name=vjepa-vitg-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/pipeline_vjepa_vitg_ddp_%j.log

set -euo pipefail
mkdir -p logs

echo "===== Step 1: Build V-JEPA ViT-g Dataset ====="

uv run python scripts/3_build_dataset.py \
    --features-dir data/vjepa_features_v21_vitg \
    --output-dir data/vjepa_aligned_v21_vitg \
    --annotations-dir data/annotations \
    --vocab configs/vocab.yaml \
    --actions configs/actions.yaml

echo ""
echo "=== Dataset stats ==="
wc -l data/vjepa_aligned_v21_vitg/splits/*.txt
echo ""

echo "===== Step 2: DDP Training (4-GPU) ====="

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=29503 \
    scripts/4_train.py \
    --config configs/vjepa_training.yaml \
    --override configs/experiment/vjepa_vitg_ddp.yaml
