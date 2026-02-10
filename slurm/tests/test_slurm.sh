#!/bin/bash
# =============================================================================
# Test SLURM job submission on devel partition (10 min limit).
# Verifies: job scripts parse correctly, dependency chain works, cluster config.
# Jobs will likely timeout — this is expected.
# Usage: ./slurm/tests/test_slurm.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOBS_DIR="$SCRIPT_DIR/../jobs"

# Override partition and time for all jobs
DEVEL_OVERRIDES="--partition=devel --time=00:10:00"

echo "Submitting test jobs to devel partition (10 min limit)..."
echo "NOTE: Jobs are expected to timeout. This tests submission only."
echo ""

J1=$(sbatch --cluster=htc --parsable $DEVEL_OVERRIDES "$JOBS_DIR/01_generate_graphs.sh" | cut -d';' -f1)
echo "  01_generate_graphs: $J1"

J2=$(sbatch --cluster=htc --parsable $DEVEL_OVERRIDES --dependency=afterok:"$J1" "$JOBS_DIR/02_create_data.sh" | cut -d';' -f1)
echo "  02_create_data:     $J2 (after $J1)"

J3=$(sbatch --cluster=htc --parsable $DEVEL_OVERRIDES --gres=gpu:1 --dependency=afterok:"$J2" "$JOBS_DIR/03_train.sh" | cut -d';' -f1)
echo "  03_train:           $J3 (after $J2)"

# For predict, provide a dummy model name — it will fail but tests submission
J4=$(sbatch --cluster=htc --parsable $DEVEL_OVERRIDES --gres=gpu:1 \
    --export=ALL,TRAINED_MODEL_NAME="test_dummy",PREDICT_CSV="data/example_dataset.csv",PREDICT_NAME="test_devel" \
    --dependency=afterok:"$J3" "$JOBS_DIR/04_predict.sh" | cut -d';' -f1)
echo "  04_predict:         $J4 (after $J3)"

echo ""
echo "All 4 jobs submitted. Check status:"
echo "  squeue -u \$USER --cluster=htc"
echo ""
echo "Cancel all test jobs:"
echo "  scancel --cluster=htc $J1 $J2 $J3 $J4"
