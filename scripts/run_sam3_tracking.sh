#!/bin/bash
#SBATCH --job-name=sam3-tracking
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sam3_tracking_%A_%a.log
#SBATCH --array=0-3

set -euo pipefail

echo "===== SAM 3 Tracking Job ====="
echo "Job ID: ${SLURM_ARRAY_JOB_ID:-local}_${SLURM_ARRAY_TASK_ID:-0}"
echo "Shard: ${SLURM_ARRAY_TASK_ID:-0} / ${SLURM_ARRAY_TASK_COUNT:-1}"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "Date: $(date)"
echo "Working dir: $(pwd)"
echo "=============================="

# Ensure logs directory exists
mkdir -p logs

SHARD_ID=${SLURM_ARRAY_TASK_ID:-0}
NUM_SHARDS=${SLURM_ARRAY_TASK_COUNT:-1}

# Run tracking with sharding and resume support
uv run python scripts/run_sam3_tracking.py \
    --config configs/sam3.yaml \
    --video-dir data/mp4 \
    --output-dir data/sam3_outputs \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS" \
    --resume

echo "===== Shard $SHARD_ID Complete ====="
echo "Date: $(date)"
