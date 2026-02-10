#!/bin/bash
# =============================================================================
# Submit training pipeline: graphs → data → train
# Usage: ./slurm/submit_training.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting training pipeline (arc/htc clusters)..."

J1=$(sbatch --cluster=arc --parsable "$SCRIPT_DIR/jobs/01_generate_graphs.sh" | cut -d';' -f1)
echo "  01_generate_graphs: $J1 (arc)"

J2=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J1" "$SCRIPT_DIR/jobs/02_create_data.sh" | cut -d';' -f1)
echo "  02_create_data:     $J2 (htc, after $J1)"

J3=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J2" "$SCRIPT_DIR/jobs/03_train.sh" | cut -d';' -f1)
echo "  03_train:           $J3 (htc, after $J2)"

echo ""
echo "Pipeline submitted. Monitor with:"
echo "  squeue -u \$USER --cluster=arc  # for graph generation"
echo "  squeue -u \$USER --cluster=htc  # for data creation and training"
echo "  tail -f logs/train_${J3}.out"
