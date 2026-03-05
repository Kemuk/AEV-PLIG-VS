#!/bin/bash
# =============================================================================
# Quick training test for the ranking model (10 minutes, devel partition).
#
# Tests the full train_retrieval.py pipeline with:
# - 2 epochs (vs 100 in production)
# - devel partition (10 min max)
#
# Usage:
#   RUN_NAME=retrieval_test ./slurm/tests/jobs/06_train_retrieval_quick.sh [--dataset DATASET]
#
# Required environment variables:
#   RUN_NAME    Name for this test run
#
# Optional environment variables:
#   DATASET     Dataset name (default: from slurm/config.sh DATASET_NAME)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# --- Required ---
RUN_NAME="${RUN_NAME:?Error: set RUN_NAME before running this job}"

# --- Overridable ---
DATASET="${DATASET:-${DATASET_NAME}}"

# SLURM settings (quick test)
PARTITION="${PARTITION_DEVEL}"
TIME_LIMIT="00:10:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1
EPOCHS=2

echo "========================================"
echo "QUICK TEST: Ranking Model Training"
echo "========================================"
echo "Run name:  ${RUN_NAME}"
echo "Dataset:   ${DATASET}"
echo "Epochs:    ${EPOCHS} (TEST)"
echo "Partition: ${PARTITION} (10 min max)"
echo "Output:    output/trained_models/${RUN_NAME}/"
echo "========================================"
echo ""

mkdir -p "$PROJECT_ROOT/slurm/logs"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}_quick
#SBATCH --cluster=${CLUSTER_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${PROJECT_ROOT}/slurm/logs/%x_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm/logs/%x_%j.err
#SBATCH --chdir=${PROJECT_ROOT}

source ${PROJECT_ROOT}/slurm/config.sh

echo "========================================="
echo "QUICK TEST: Training Ranking Model"
echo "========================================="
echo "Node:    \$(hostname)"
echo "Job ID:  \${SLURM_JOB_ID}"
echo "Dataset: ${DATASET}"
echo "Epochs:  ${EPOCHS}"
echo "========================================="
echo ""

train_cmd=(
    python scripts/train_retrieval.py
    --dataset "${DATASET}"
    --run-name "${RUN_NAME}"
    --epochs "${EPOCHS}"
    --device auto
)
printf 'CMD: %q ' "\${train_cmd[@]}"; echo
"\${train_cmd[@]}"

echo ""
echo "Quick test complete."
echo "Model saved to: output/trained_models/${RUN_NAME}/"
EOF
)

echo "========================================"
echo "Submitted job: ${JOB_ID}"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Expected runtime: ~5-10 minutes"
echo "========================================"
