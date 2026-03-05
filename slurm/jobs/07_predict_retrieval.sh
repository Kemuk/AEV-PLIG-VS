#!/bin/bash
# =============================================================================
# Submit a retrieval evaluation job.
#
# Scores all complexes in a data split, ranks ligands within each protein
# target, and computes EF, BEDROC, and RIE metrics. Works with any trained
# GATv2Net — affinity (MSE-trained) or ranking (margin-ranking-trained).
#
# Usage:
#   MODEL_DIR=output/trained_models/my_ranking_run \
#     ./slurm/jobs/07_predict_retrieval.sh [--dataset DATASET] [--split SPLIT]
#
# Required environment variables:
#   MODEL_DIR   Path to trained model directory (must contain config.json + model.pt)
#
# Optional environment variables:
#   DATASET          Dataset name (default: from slurm/config.sh DATASET_NAME)
#   SPLIT            Data split to evaluate: train/valid/test (default: test)
#   USE_DIAGNOSTICS  Set to 1 to run false-positive diagnostic analysis
#
# Output:
#   output/retrieval_results/<model-dir-name>/
#     {dataset}_{split}_predictions.parquet
#     {dataset}_{split}_retrieval_metrics.parquet
#     {dataset}_{split}_retrieval_summary.json
#     {dataset}_{split}_false_positives.parquet  (if USE_DIAGNOSTICS=1)
#     {dataset}_{split}_diagnostics_summary.json (if USE_DIAGNOSTICS=1)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# --- Required ---
MODEL_DIR="${MODEL_DIR:?Error: set MODEL_DIR before running this job (e.g. MODEL_DIR=output/trained_models/my_ranking_run)}"

# --- Overridable ---
DATASET="${DATASET:-${DATASET_NAME}}"
SPLIT="${SPLIT:-test}"
MODEL_DIR_NAME="$(basename "$MODEL_DIR")"

# SLURM settings
PARTITION="${PARTITION_SHORT}"
TIME_LIMIT="02:00:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1

echo "========================================"
echo "Retrieval Evaluation Job Submission"
echo "========================================"
echo "Model dir: ${MODEL_DIR}"
echo "Dataset:   ${DATASET}"
echo "Split:     ${SPLIT}"
echo "Partition: ${PARTITION}"
echo "Output:    output/retrieval_results/${MODEL_DIR_NAME}/"
echo "========================================"
echo ""

mkdir -p "$PROJECT_ROOT/slurm/logs"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=retrieval_${MODEL_DIR_NAME}
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
echo "Retrieval Evaluation"
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
[[ "\${USE_DIAGNOSTICS:-0}" == "1" ]] && predict_cmd+=(--diagnostics)
printf 'CMD: %q ' "\${predict_cmd[@]}"; echo
"\${predict_cmd[@]}"

echo ""
echo "Evaluation complete."
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
echo "View logs:"
echo "  ${PROJECT_ROOT}/slurm/logs/retrieval_${MODEL_DIR_NAME}_${JOB_ID}.out"
echo "========================================"
