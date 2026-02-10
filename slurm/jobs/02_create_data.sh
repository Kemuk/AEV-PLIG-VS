#!/bin/bash
#SBATCH --job-name=aev-data
#SBATCH --cluster=htc
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/data_%j.out
#SBATCH --error=logs/data_%j.err
# =============================================================================
# Step 2: Create PyTorch datasets from generated graph pickles.
# Combines pdbbind, bindingnet, and bindingdb into train/valid/test splits.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

echo "Creating PyTorch datasets..."
python create_pytorch_data.py

echo "Data creation completed. Output in data/processed/"
