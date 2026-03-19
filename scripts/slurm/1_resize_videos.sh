#!/bin/bash
#SBATCH --job-name=resize-videos
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/resize_videos_%j.log

set -euo pipefail

echo "===== Resize Videos ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Node: $(hostname)"
echo "GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)x A5000"
echo "Date: $(date)"
echo "========================="

mkdir -p logs

uv run python scripts/1_resize_videos.py \
    --input-dir "${INPUT_DIR:-data/mp4}" \
    --output-dir "${OUTPUT_DIR:-data/resized}" \
    --max-side "${MAX_SIDE:-1008}" \
    --workers "${WORKERS:-16}" \
    --gpu-ids "${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
    --resume

echo "===== Done at $(date) ====="
