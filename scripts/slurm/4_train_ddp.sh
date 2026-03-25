#!/bin/bash
#SBATCH --job-name=vjepa-ddp
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gpus=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#SBATCH --output=logs/vjepa_ddp_%j.log

mkdir -p logs

uv run torchrun \
    --nproc_per_node=4 \
    --master_port=29500 \
    scripts/4_train.py \
    --config configs/vjepa_training.yaml \
    --override configs/experiment/vjepa_ddp.yaml
