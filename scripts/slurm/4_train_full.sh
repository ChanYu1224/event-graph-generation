#!/bin/bash
#SBATCH --job-name=vjepa-train-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vjepa_train_full_%j.log

set -euo pipefail
echo "===== V-JEPA Pipeline Training (Full) ====="
echo "Job ID: ${SLURM_JOB_ID}, Date: $(date), Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

CONFIG="${CONFIG:-configs/vjepa_training.yaml}"
OVERRIDE="${OVERRIDE:-configs/experiment/vjepa_full.yaml}"

uv run python scripts/4_train.py \
    --config "$CONFIG" \
    --override "$OVERRIDE"

echo "===== Done at $(date) ====="
