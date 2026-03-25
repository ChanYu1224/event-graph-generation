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

echo "=== VLM Annotation Job (VLLM Server) ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPUs: ${CUDA_VISIBLE_DEVICES:-not set}"
echo "========================================="

mkdir -p logs

# Start VLLM server in the background
uv run vllm serve Qwen/Qwen3.5-35B-A3B \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.90 \
  --max-model-len 32768 \
  --max-num-seqs 16 \
  --enable-prefix-caching \
  --limit-mm-per-prompt '{"image": 16}' \
  --dtype bfloat16 \
  --port 8002 &
SERVER_PID=$!

# Wait for server to become ready (check both health endpoint and process)
echo "Waiting for VLLM server to start (PID=${SERVER_PID})..."
SERVER_READY=false
for i in $(seq 1 120); do
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "ERROR: Server process (PID=${SERVER_PID}) died during startup"
    exit 1
  fi
  if curl -s http://localhost:8002/health > /dev/null 2>&1; then
    echo "Server ready after $((i * 5))s"
    SERVER_READY=true
    break
  fi
  sleep 5
done

if [ "$SERVER_READY" = false ]; then
  echo "ERROR: Server failed to start within 600s"
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

# Run annotation
uv run python scripts/2_generate_annotations.py \
  --config configs/vlm_vllm_server.yaml \
  --vocab-config configs/vocab.yaml \
  --video-dir data/resized/room \
  --output-dir data/annotations \
  --resume

# Stop server
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "=== Done at $(date) ==="
