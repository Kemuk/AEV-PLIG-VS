#!/bin/bash
# =============================================================================
# Submit full pipeline: graphs → data → train → predict
# Usage: TRAINED_MODEL_NAME will be unknown until training completes.
#        Prediction uses example dataset as a smoke test.
#
#   ./slurm/submit_slurm.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting full pipeline to htc cluster..."

J1=$(sbatch --cluster=htc --parsable "$SCRIPT_DIR/jobs/01_generate_graphs.sh")
echo "  01_generate_graphs: $J1"

J2=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J1" "$SCRIPT_DIR/jobs/02_create_data.sh")
echo "  02_create_data:     $J2 (after $J1)"

J3=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J2" "$SCRIPT_DIR/jobs/03_train.sh")
echo "  03_train:           $J3 (after $J2)"

echo ""
echo "Training pipeline submitted (jobs 01-03)."
echo ""
echo "NOTE: Prediction (job 04) must be submitted separately after training"
echo "completes, because the model name includes a timestamp."
echo "Once training finishes, check the model name:"
echo "  ls output/trained_models/"
echo "Then submit prediction:"
echo "  TRAINED_MODEL_NAME=\"<name>\" ./slurm/submit_prediction.sh"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --cluster=htc"
