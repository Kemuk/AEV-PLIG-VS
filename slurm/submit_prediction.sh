#!/bin/bash
# =============================================================================
# Submit prediction job only.
# Usage: TRAINED_MODEL_NAME="<name>" ./slurm/submit_prediction.sh [dataset.csv] [output_name]
#
# Example:
#   TRAINED_MODEL_NAME="20260208_model_GATv2NetBayesian_pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark" \
#     ./slurm/submit_prediction.sh data/my_dataset.csv my_results
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${TRAINED_MODEL_NAME:?Error: set TRAINED_MODEL_NAME before running this script}"

PREDICT_CSV="${1:-data/example_dataset.csv}"
PREDICT_NAME="${2:-predictions}"

echo "Submitting prediction job to htc cluster..."
echo "  Model:   $TRAINED_MODEL_NAME"
echo "  Input:   $PREDICT_CSV"
echo "  Output:  $PREDICT_NAME"

J1=$(sbatch --cluster=htc --parsable \
    --export=ALL,TRAINED_MODEL_NAME="$TRAINED_MODEL_NAME",PREDICT_CSV="$PREDICT_CSV",PREDICT_NAME="$PREDICT_NAME" \
    "$SCRIPT_DIR/jobs/04_predict.sh" | cut -d';' -f1)

echo "  04_predict: $J1"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --cluster=htc"
echo "  tail -f logs/predict_${J1}.out"
