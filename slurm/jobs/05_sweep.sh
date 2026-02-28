#!/bin/bash
# =============================================================================
# Submit W&B Bayesian sweep agent array job.
#
# Each SLURM task runs one W&B agent (--count 1) which requests a single HP
# configuration from the Bayesian controller, trains, logs val_pearson_r, and
# exits.  W&B coordinates across tasks automatically.
#
# Pre-requisite:
#   wandb sweep sweeps/gatv2net_sweep.yaml --project aev-plig-vs
#   → prints SWEEP_ID
#
# Usage:
#   ./slurm/jobs/05_sweep.sh <SWEEP_ID> [--agents N] [--dataset DATASET]
#
# Options:
#   --agents N        Number of parallel sweep agents (default: 50)
#   --dataset NAME    Training dataset name (default: from config.sh)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# Required positional: sweep ID from `wandb sweep` output
SWEEP_ID="${1:?Usage: $0 <SWEEP_ID> [--agents N] [--dataset DATASET]}"
shift

# Defaults
NUM_AGENTS=50
DATASET="${DATASET_NAME}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --agents)  NUM_AGENTS="$2"; shift 2 ;;
        --dataset) DATASET="$2";    shift 2 ;;
        *) shift ;;
    esac
done

# SLURM settings (matching 03_train.sh conventions)
PARTITION="${PARTITION_SHORT}"
TIME_LIMIT="04:00:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS="v100:1"

# W&B entity — falls back to wandb default (logged-in user) if unset
ENTITY="${WANDB_ENTITY:-}"
if [[ -n "$ENTITY" ]]; then
    AGENT_TARGET="${ENTITY}/aev-plig-vs/${SWEEP_ID}"
else
    AGENT_TARGET="aev-plig-vs/${SWEEP_ID}"
fi

mkdir -p "$PROJECT_ROOT/slurm/logs"

echo "========================================"
echo "W&B Sweep Agent Submission"
echo "========================================"
echo "Sweep ID:   ${SWEEP_ID}"
echo "Target:     ${AGENT_TARGET}"
echo "Dataset:    ${DATASET}"
echo "Agents:     ${NUM_AGENTS}"
echo "Partition:  ${PARTITION} (${TIME_LIMIT})"
echo "GPU:        ${GPUS}"
echo "========================================"
echo ""

JOB_ID=$(sbatch --parsable --array=1-${NUM_AGENTS} <<EOF
#!/bin/bash
#SBATCH --job-name=sweep_agent
#SBATCH --cluster=${CLUSTER_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${PROJECT_ROOT}/slurm/logs/%x_%A_%a.out
#SBATCH --error=${PROJECT_ROOT}/slurm/logs/%x_%A_%a.err
#SBATCH --chdir=${PROJECT_ROOT}

source ${PROJECT_ROOT}/slurm/config.sh

echo "========================================="
echo "W&B Sweep Agent"
echo "========================================="
echo "Node:     \$(hostname)"
echo "Job ID:   \${SLURM_JOB_ID} (task \${SLURM_ARRAY_TASK_ID})"
echo "Sweep:    ${SWEEP_ID}"
echo "========================================="
echo ""

wandb agent ${AGENT_TARGET} --count 1

echo ""
echo "✓ Sweep agent complete (task \${SLURM_ARRAY_TASK_ID})"
EOF
)

echo "========================================"
echo "✓ Submitted array job (${NUM_AGENTS} agents)"
echo "========================================"
echo "Job ID: ${JOB_ID}  (tasks ${JOB_ID}_1 to ${JOB_ID}_${NUM_AGENTS})"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER --cluster=${CLUSTER_NAME}"
echo ""
echo "View logs in:"
echo "  ${PROJECT_ROOT}/slurm/logs/"
echo ""
echo "After sweep completes, select top runs from W&B dashboard, then:"
echo "  for run in <run1> <run2> ...; do"
echo "    python scripts/predict.py --trained_model_name \$run --dataset_csv <CSV> --data_name <NAME>"
echo "  done"
echo "  python scripts/merge_sweep.py --trained_model_names <run1> <run2> ... \\"
echo "    --data_name <NAME> --output_name GATv2Net_sweep_${SWEEP_ID}"
echo "========================================"
