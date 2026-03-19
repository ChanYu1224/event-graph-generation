#!/bin/bash
#SBATCH --job-name=benchmark_vram
#SBATCH --partition=gpu
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/benchmark_vram_%j.out
#SBATCH --error=logs/benchmark_vram_%j.err

set -euo pipefail

mkdir -p logs

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== VRAM Benchmark ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

uv run python scripts/benchmark_vram.py

echo "=== Done at $(date) ==="
