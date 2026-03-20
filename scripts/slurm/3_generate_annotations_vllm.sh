#!/bin/bash
#SBATCH --job-name=vlm-annotate-vllm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vlm_annotate_vllm_%j.log

set -euo pipefail

echo "=== VLM Annotation Job (VLLM) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "=================================="

mkdir -p logs

uv run python scripts/3_generate_annotations.py \
  --config configs/vlm_vllm.yaml \
  --video-dir data/resized/room \
  --output-dir data/annotations \
  --resume

echo "=== Done at $(date) ==="
