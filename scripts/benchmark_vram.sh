#!/bin/bash
#SBATCH --job-name=benchmark_vram
#SBATCH --partition=gpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/benchmark_vram_%j.out
#SBATCH --error=logs/benchmark_vram_%j.err

set -euo pipefail

cd /home/team-006/nishikawa/event-graph-generation
mkdir -p logs

# Slurmが CUDA_VISIBLE_DEVICES を設定しない場合、先頭4 GPUに制限
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

uv run python scripts/benchmark_vram.py
