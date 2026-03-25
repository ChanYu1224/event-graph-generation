#!/bin/bash
#SBATCH --job-name=event-decoder-eval
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/eval_event_decoder_%j.log

set -euo pipefail

echo "===== Event Decoder Evaluation ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "====================================="

mkdir -p logs

uv run python scripts/5_evaluate.py \
    --config "${CONFIG:-configs/training.yaml}" \
    --checkpoint "${CHECKPOINT:-data/checkpoints/best.pt}"

echo "===== Done at $(date) ====="
