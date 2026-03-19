#!/bin/bash
#SBATCH --job-name=event-inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/inference_%j.log

set -euo pipefail

VIDEO_PATH="${VIDEO_PATH:?VIDEO_PATH must be set}"
CHECKPOINT="${CHECKPOINT:-data/checkpoints/best.pt}"
OUTPUT="${OUTPUT:-output/event_graph.json}"
CONFIG="${CONFIG:-configs/inference.yaml}"

echo "===== Event Graph Inference ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Video:      ${VIDEO_PATH}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output:     ${OUTPUT}"
echo "Config:     ${CONFIG}"
echo "Date: $(date)"
echo "=================================="

mkdir -p logs output

uv run python scripts/7_run_inference.py \
    --video "$VIDEO_PATH" \
    --checkpoint "$CHECKPOINT" \
    --output "$OUTPUT" \
    --config "$CONFIG"

echo "===== Done at $(date) ====="
