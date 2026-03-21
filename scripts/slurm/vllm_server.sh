#!/bin/bash
#SBATCH --job-name=vllm-server
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=logs/vllm_server_%j.log

set -euo pipefail

echo "=== VLLM Server ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "==================="

mkdir -p logs

uv run vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 16}' \
  --dtype bfloat16 \
  --port 8000
