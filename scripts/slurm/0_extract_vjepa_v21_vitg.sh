#!/bin/bash
#SBATCH --job-name=vjepa21-vitg
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/vjepa21_vitg_%j.log

set -euo pipefail
echo "===== V-JEPA 2.1 ViT-g Feature Extraction ====="
echo "Job ID: ${SLURM_JOB_ID}, Date: $(date), Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

VIDEO_DIR="${VIDEO_DIR:-data/resized/room}"

uv run python scripts/0_extract_vjepa_features.py \
    --config configs/vjepa_vitg.yaml \
    --video-dir "$VIDEO_DIR" \
    --output-dir data/vjepa_features_v21_vitg \
    --device cuda

echo "===== Done at $(date) ====="
