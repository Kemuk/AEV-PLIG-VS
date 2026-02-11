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
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Train model ensemble.
#
# Single-seed mode (parallel): Set SEED and TIMESTAMP env vars
# Ensemble mode (sequential): Run without env vars (trains all seeds)
# =============================================================================

source slurm/config.sh

echo "Training model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"

# Check if running in single-seed mode (parallel job)
if [[ -n "${SEED:-}" ]]; then
    echo "Single-seed mode: Seed $SEED, Timestamp $TIMESTAMP"

    python scripts/train.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME" \
        --seed "$SEED" \
        --timestamp "$TIMESTAMP"
else
    echo "Ensemble mode: training all seeds sequentially"

    python scripts/train.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME"
fi

echo "Training completed. Models saved to models/"
