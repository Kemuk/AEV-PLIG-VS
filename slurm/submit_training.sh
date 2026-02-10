#!/bin/bash
# =============================================================================
# Submit training pipeline: graphs → data → train
# Usage: ./slurm/submit_training.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting training pipeline to htc cluster..."

J1=$(sbatch --cluster=htc --parsable "$SCRIPT_DIR/jobs/01_generate_graphs.sh" | cut -d';' -f1)
echo "  01_generate_graphs: $J1"

J2=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J1" "$SCRIPT_DIR/jobs/02_create_data.sh" | cut -d';' -f1)
echo "  02_create_data:     $J2 (after $J1)"

J3=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J2" "$SCRIPT_DIR/jobs/03_train.sh" | cut -d';' -f1)
echo "  03_train:           $J3 (after $J2)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER --cluster=htc"
echo "  tail -f logs/train_${J3}.out"
