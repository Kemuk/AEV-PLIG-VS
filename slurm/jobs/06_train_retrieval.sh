#!/bin/bash
# =============================================================================
# Submit a single ranking-model training job.
#
# Trains GATv2Net with pairwise margin ranking loss for virtual screening.
# Unlike the affinity ensemble (03_train.sh), this is a single job — ranking
# models do not require multi-seed ensembling.
#
# Usage:
#   RUN_NAME=my_ranking_run ./slurm/jobs/06_train_retrieval.sh [--dataset DATASET]
#
# Required environment variables:
#   RUN_NAME    Name for this training run (used as output directory name)
#
# Optional environment variables:
#   DATASET     Dataset name (default: from slurm/config.sh DATASET_NAME)
#   USE_WANDB   Set to 1 to enable Weights & Biases logging
#
# Output:
#   output/trained_models/$RUN_NAME/model.pt
#   output/trained_models/$RUN_NAME/config.json
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# --- Required ---
RUN_NAME="${RUN_NAME:?Error: set RUN_NAME before running this job (e.g. RUN_NAME=my_ranking_run)}"

# --- Overridable ---
DATASET="${DATASET:-${DATASET_NAME}}"

# SLURM settings
PARTITION="${PARTITION_SHORT}"
TIME_LIMIT="04:00:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1

echo "========================================"
echo "Ranking Model Training Job Submission"
echo "========================================"
echo "Run name:  ${RUN_NAME}"
echo "Dataset:   ${DATASET}"
echo "Partition: ${PARTITION}"
echo "Output:    output/trained_models/${RUN_NAME}/"
echo "========================================"
echo ""

mkdir -p "$PROJECT_ROOT/slurm/logs"

JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=${RUN_NAME}
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
echo "Training Ranking Model: ${RUN_NAME}"
echo "========================================="
echo "Node:    \$(hostname)"
echo "Job ID:  \${SLURM_JOB_ID}"
echo "Dataset: ${DATASET}"
echo "========================================="
echo ""

train_cmd=(
    python scripts/train_retrieval.py
    --dataset "${DATASET}"
    --run-name "${RUN_NAME}"
    --device auto
)
[[ "\${USE_WANDB:-0}" == "1" ]] && train_cmd+=(--wandb --wandb-project "aev-plig-vs")
printf 'CMD: %q ' "\${train_cmd[@]}"; echo
"\${train_cmd[@]}"

echo ""
echo "Training complete."
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
echo "View logs:"
echo "  ${PROJECT_ROOT}/slurm/logs/${RUN_NAME}_${JOB_ID}.out"
echo "========================================"
