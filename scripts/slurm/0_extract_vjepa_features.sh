#!/bin/bash
#SBATCH --job-name=vjepa-extract
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=3:00:00
#SBATCH --output=logs/vjepa_extract_%j.log

set -euo pipefail
echo "===== V-JEPA Feature Extraction ====="
echo "Job ID: ${SLURM_JOB_ID}, Date: $(date), Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

mkdir -p logs

VIDEO_DIR="${VIDEO_DIR:-data/resized/trial}"
OUTPUT_DIR="${OUTPUT_DIR:-data/vjepa_features}"

uv run python scripts/0_extract_vjepa_features.py \
    --config configs/vjepa.yaml \
    --video-dir "$VIDEO_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --device cuda

echo "===== Done at $(date) ====="
