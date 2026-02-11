#!/bin/bash
#SBATCH --job-name=aev-train-quick
#SBATCH --cluster=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_quick_%j.out
#SBATCH --error=logs/train_quick_%j.err
# =============================================================================
# Quick test: train for 5 epochs only (vs 200 in production).
# Use to verify GPU allocation, data loading, and training loop.
# =============================================================================

source "$(dirname "$0")/../../config.sh"

echo "Quick test: training for 5 epochs..."

aev-plig-train \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --epochs 5

echo "Quick training test completed."
