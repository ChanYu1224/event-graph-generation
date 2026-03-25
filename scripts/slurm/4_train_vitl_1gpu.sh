#!/bin/bash
#SBATCH --job-name=vjepa-vitl-1gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_vjepa_vitl_1gpu_%j.log

set -euo pipefail
mkdir -p logs

uv run python scripts/4_train.py \
    --config configs/vjepa_training.yaml
