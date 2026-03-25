#!/bin/bash
#SBATCH --job-name=vjepa-vitb-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/train_vjepa_vitb_ddp_%j.log

set -euo pipefail
mkdir -p logs

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=29501 \
    scripts/4_train.py \
    --config configs/vjepa_training.yaml \
    --override configs/experiment/vjepa_vitb_ddp.yaml
