#!/bin/bash
#SBATCH --job-name=build-dataset
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/build_dataset_%j.log

set -euo pipefail

echo "===== Build Dataset ====="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "========================="

mkdir -p logs

uv run python scripts/4_build_dataset.py \
    --sam3-dir "${SAM3_DIR:-data/sam3_outputs}" \
    --annotations-dir "${ANNOTATIONS_DIR:-data/annotations}" \
    --output-dir "${OUTPUT_DIR:-data/aligned}"

echo "===== Done at $(date) ====="
