#!/bin/bash
#SBATCH --job-name=aev-predict
#SBATCH --cluster=htc
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/predict_%j.out
#SBATCH --error=logs/predict_%j.err
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Step 4: Run predictions with a trained model ensemble.
#
# Before running, set TRAINED_MODEL_NAME to the model produced by training.
# Check output/trained_models/ for the name (includes a timestamp), e.g.:
#   export TRAINED_MODEL_NAME="20260208-143000_model_GATv2NetBayesian_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"
# =============================================================================

source slurm/config.sh

# --- Required: set these before submission ---
TRAINED_MODEL_NAME="${TRAINED_MODEL_NAME:?Error: set TRAINED_MODEL_NAME before running this job}"
PREDICT_CSV="${PREDICT_CSV:-data/example_dataset.csv}"
PREDICT_NAME="${PREDICT_NAME:-predictions}"

echo "Model:   $TRAINED_MODEL_NAME"
echo "Input:   $PREDICT_CSV"
echo "Output:  $PREDICT_NAME"

python scripts/predict.py \
    --model "$MODEL_NAME" \
    --trained_model_name "$TRAINED_MODEL_NAME" \
    --dataset_csv "$PREDICT_CSV" \
    --data_name "$PREDICT_NAME" \
    --num_workers "$CPUS_STANDARD" \
    --device auto

echo "Predictions saved to output/predictions/${PREDICT_NAME}_predictions.parquet"
