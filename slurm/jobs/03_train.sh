#!/bin/bash
#SBATCH --job-name=aev-train
#SBATCH --cluster=htc
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
# =============================================================================
# Step 3: Train model ensemble.
# Trains all ensemble members (seeds defined in aev_plig/config.py).
# All hyperparameters (lr, epochs, batch_size, etc.) use Python defaults.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

echo "Training model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"

aev-plig-train \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME"

echo "Training completed. Models saved to output/trained_models/"
