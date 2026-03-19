#!/bin/bash
#SBATCH --job-name=event-decoder-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_event_decoder_%j.log

set -euo pipefail

echo "===== Event Decoder Training ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "==================================="

mkdir -p logs

CONFIG="${CONFIG:-configs/training.yaml}"
OVERRIDE="${OVERRIDE:-configs/experiment/event_decoder_v1.yaml}"

uv run python scripts/5_train.py \
    --config "$CONFIG" \
    --override "$OVERRIDE"

echo "===== Done at $(date) ====="
