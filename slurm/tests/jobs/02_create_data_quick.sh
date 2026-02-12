#!/bin/bash
#SBATCH --job-name=aev-data-quick
#SBATCH --cluster=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/data_quick_%j.out
#SBATCH --error=logs/data_quick_%j.err
# =============================================================================
# Quick test: Dry run using pre-existing TEST split subset.
#
# Runs create_pytorch_data.py in QUICK_TEST mode (test split only).
# Tests all 3 optimization phases with ~5-10K graphs:
#   Phase 1: Parallel pickle loading (all 3 files)
#   Phase 2: Polars CSV processing (test split only)
#   Phase 3: Parallel graph processing (test subset only)
#
# Uses FULL resources (no constraints) but minimal data for quick validation.
# Expected runtime: ~2-5 minutes
# =============================================================================

source "$(dirname "$0")/../../config.sh"

echo "========================================================================"
echo "DRY RUN: Data Creation Pipeline (TEST SPLIT ONLY)"
echo "Testing all 3 phases with pre-existing test subset"
echo "========================================================================"
echo ""
echo "Job Info:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Node: $SLURM_NODELIST"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo ""

mkdir -p logs

# Enable quick test mode via environment variable
export QUICK_TEST=1

python create_pytorch_data.py

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ DRY RUN PASSED"
    echo "  All 3 phases working correctly"
    echo "  Ready to run full pipeline: sbatch slurm/jobs/02_create_data.sh"
else
    echo "✗ DRY RUN FAILED (exit code: $EXIT_CODE)"
    echo "  Check logs/data_quick_${SLURM_JOB_ID}.err for details"
fi
echo "========================================================================"

exit $EXIT_CODE
