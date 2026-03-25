#!/bin/bash
#SBATCH --job-name=vjepa-extract-full
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vjepa_extract_full_%j_%a.log
#SBATCH --array=0-3

set -euo pipefail

NUM_SHARDS=4
SHARD_ID=${SLURM_ARRAY_TASK_ID}

echo "===== V-JEPA Feature Extraction (Shard ${SHARD_ID}/${NUM_SHARDS}) ====="
echo "Job ID: ${SLURM_JOB_ID}, Array Task: ${SLURM_ARRAY_TASK_ID}, Date: $(date), Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

VIDEO_DIR="${VIDEO_DIR:-data/resized/room}"
OUTPUT_DIR="${OUTPUT_DIR:-data/vjepa_features}"

# Each shard uses a different GPU via CUDA_VISIBLE_DEVICES set by SLURM
uv run python scripts/1_extract_features.py \
    --config configs/vjepa.yaml \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda \
    --shard-id "$SHARD_ID" \
    --num-shards "$NUM_SHARDS"

echo "===== Shard ${SHARD_ID} Done at $(date) ====="
