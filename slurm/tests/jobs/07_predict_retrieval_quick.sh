#!/bin/bash
# =============================================================================
# Quick retrieval evaluation test (10 minutes, devel partition).
#
# Tests the full predict_retrieval.py pipeline on the test split.
#
# Usage:
#   MODEL_DIR=output/trained_models/retrieval_test \
#     ./slurm/tests/jobs/07_predict_retrieval_quick.sh [--dataset DATASET]
#
# Required environment variables:
#   MODEL_DIR   Path to trained model directory (must contain config.json + model.pt)
#
# Optional environment variables:
#   DATASET     Dataset name (default: from slurm/config.sh DATASET_NAME)
#   SPLIT       Data split: train/valid/test (default: test)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# --- Required ---
MODEL_DIR="${MODEL_DIR:?Error: set MODEL_DIR before running this job}"

# --- Overridable ---
DATASET="${DATASET:-${DATASET_NAME}}"
SPLIT="${SPLIT:-test}"
MODEL_DIR_NAME="$(basename "$MODEL_DIR")"

# SLURM settings (quick test)
PARTITION="${PARTITION_DEVEL}"
TIME_LIMIT="00:10:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1

echo "========================================"
echo "QUICK TEST: Retrieval Evaluation"
echo "========================================"
echo "Model dir: ${MODEL_DIR}"
echo "Dataset:   ${DATASET} / ${SPLIT}"
echo "Partition: ${PARTITION} (10 min max)"
echo "Output:    output/retrieval_results/${MODEL_DIR_NAME}/"
echo "========================================"
echo ""

mkdir -p "$PROJECT_ROOT/slurm/logs"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=retrieval_quick
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
echo "QUICK TEST: Retrieval Evaluation"
echo "========================================="
echo "Node:      \$(hostname)"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Model dir: ${MODEL_DIR}"
echo "Dataset:   ${DATASET} / ${SPLIT}"
echo "========================================="
echo ""

predict_cmd=(
    python scripts/predict_retrieval.py
    --model-dir "${MODEL_DIR}"
    --dataset "${DATASET}"
    --split "${SPLIT}"
    --device auto
)
printf 'CMD: %q ' "\${predict_cmd[@]}"; echo
"\${predict_cmd[@]}"

echo ""
echo "Quick test complete."
echo "Results saved to: output/retrieval_results/${MODEL_DIR_NAME}/"
EOF
)

echo "========================================"
echo "Submitted job: ${JOB_ID}"
echo "========================================"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "Expected runtime: ~5 minutes"
echo "========================================"
