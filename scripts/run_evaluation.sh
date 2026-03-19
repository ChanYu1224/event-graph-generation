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

mkdir -p logs

uv run python scripts/evaluate.py --config configs/training.yaml --checkpoint data/checkpoints/best.pt
