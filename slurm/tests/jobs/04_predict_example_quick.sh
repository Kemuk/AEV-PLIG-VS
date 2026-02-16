#!/bin/bash
#SBATCH --job-name=aev-predict-quick
#SBATCH --cluster=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/predict_quick_%j.out
#SBATCH --error=logs/predict_quick_%j.err
# =============================================================================
# Quick prediction smoke test on example dataset.
#
# Default command equivalent:
# python scripts/predict.py \
#   --dataset_csv data/example_dataset.csv \
#   --data_name example \
#   --trained_model_name model_GATv2Net_ligsim90_fep_benchmark
#
# Optional overrides at submission:
# sbatch --export=ALL,TRAINED_MODEL_NAME=<name>,PREDICT_CSV=<path>,PREDICT_NAME=<name> \
#   slurm/tests/jobs/04_predict_example_quick.sh
# =============================================================================
set -euo pipefail

source "$(dirname "$0")/../../config.sh"

TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:-model_GATv2Net_ligsim90_fep_benchmark}"
PREDICT_CSV="${PREDICT_CSV:-data/example_dataset.csv}"
PREDICT_NAME="${PREDICT_NAME:-example}"

echo "========================================"
echo "QUICK TEST: Prediction on example dataset"
echo "========================================"
echo "Model:   ${MODEL_NAME}"
echo "Weights: ${TRAINED_MODEL_NAME}"
echo "Input:   ${PREDICT_CSV}"
echo "Output:  ${PREDICT_NAME}"
echo "========================================"

aev-plig-predict \
    --model "${MODEL_NAME}" \
    --trained_model_name "${TRAINED_MODEL_NAME}" \
    --dataset_csv "${PREDICT_CSV}" \
    --data_name "${PREDICT_NAME}" \
    --num_workers "${CPUS_STANDARD}" \
    --device auto

echo "Prediction quick test complete"
echo "Generated: output/predictions/${PREDICT_NAME}_predictions.parquet"
