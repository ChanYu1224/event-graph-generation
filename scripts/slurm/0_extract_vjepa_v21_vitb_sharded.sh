#!/bin/bash
#SBATCH --job-name=vjepa21-vitb-ext
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --gpus=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/vjepa21_vitb_sharded_%j.log

set -euo pipefail
mkdir -p logs

echo "===== V-JEPA 2.1 ViT-B Feature Extraction (2-GPU sharded) ====="
echo "Job ID: ${SLURM_JOB_ID}, Date: $(date), Host: $(hostname)"

VIDEO_DIR="${VIDEO_DIR:-data/resized/room}"

for SHARD in 0 1; do
    CUDA_VISIBLE_DEVICES=$SHARD uv run python scripts/0_extract_vjepa_features.py \
        --config configs/vjepa_vitb.yaml \
        --video-dir "$VIDEO_DIR" \
        --output-dir data/vjepa_features_v21_vitb \
        --device cuda \
        --shard-id $SHARD \
        --num-shards 2 &
done

wait
echo "===== Done at $(date) ====="
