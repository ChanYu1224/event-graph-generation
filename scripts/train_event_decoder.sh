#!/bin/bash
#SBATCH --job-name=event-decoder-train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/train_event_decoder_%j.log

set -euo pipefail

mkdir -p logs

uv run python scripts/train.py --config configs/training.yaml --override configs/experiment/event_decoder_v1.yaml
