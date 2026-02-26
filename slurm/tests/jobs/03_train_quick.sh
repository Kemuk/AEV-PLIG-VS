#!/bin/bash
# =============================================================================
# Quick parallel training test (10 minutes)
#
# Tests parallel training with:
# - 3 seeds only (vs 10 in production)
# - 2 epochs (vs 200 in production)
# - Devel partition (10 min max)
#
# Usage:
#   ./slurm/tests/jobs/03_train_quick.sh [--model MODEL_NAME] [--dataset DATASET_NAME]
#
# Options:
#   --model MODELNAME       Override model from config (e.g., GATv2NetMixedPrecision)
#   --dataset DATASETNAME   Override dataset from config
# =============================================================================
set -euo pipefail

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration
source "$PROJECT_ROOT/slurm/config.sh"

# Parse command-line arguments
MODEL="${MODEL_NAME}"
DATASET="${DATASET_NAME}"
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --dataset)
            DATASET="$2"
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

# Hyperparameters (matching original training command)
ACTIVATION="leaky_relu"
BATCH_SIZE=128
EPOCHS=2  # Quick test: only 2 epochs
HEAD=3
HIDDEN_DIM=256
LR=0.00012291937615434127

# SLURM settings (quick test)
PARTITION="${PARTITION_DEVEL}"
TIME_LIMIT="00:10:00"  # 10 min max for devel
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1

# Quick test: only 3 seeds
SEEDS=(100 123 15)

# Create shared timestamp for ensemble
TIMESTAMP=$(date +%Y-%m-%d_%H-00)

echo "========================================"
echo "QUICK TEST: Parallel Training"
echo "========================================"
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Seeds:        ${#SEEDS[@]} models (TEST)"
echo "Epochs:       ${EPOCHS} (TEST)"
echo "Timestamp:    ${TIMESTAMP}"
echo "Partition:    ${PARTITION} (10 min max)"
echo "Output dir:   output/trained_models/${MODEL}_TEST_${TIMESTAMP}/"
echo "========================================"
echo ""

# Create log directory if needed
mkdir -p "$PROJECT_ROOT/slurm/logs"

# Submit all seeds as a single array job
JOB_ID=$(sbatch --parsable --array=0-$((${#SEEDS[@]}-1)) <<EOF
#!/bin/bash
#SBATCH --job-name=${MODEL}_quick
#SBATCH --cluster=${CLUSTER_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${PROJECT_ROOT}/slurm/logs/%x_%A_%a.out
#SBATCH --error=${PROJECT_ROOT}/slurm/logs/%x_%A_%a.err
#SBATCH --chdir=${PROJECT_ROOT}

# Load environment
source ${PROJECT_ROOT}/slurm/config.sh

# Resolve seed from array task index
SEEDS=(${SEEDS[*]})
SEED="\${SEEDS[\$SLURM_ARRAY_TASK_ID]}"

echo "========================================="
echo "QUICK TEST: Training Seed \${SEED}"
echo "========================================="
echo "Node:      \$(hostname)"
echo "Job ID:    \${SLURM_JOB_ID} (array task \${SLURM_ARRAY_TASK_ID})"
echo "Timestamp: ${TIMESTAMP}"
echo "Epochs:    ${EPOCHS}"
echo "========================================="
echo ""

# Train single model (array form avoids whitespace/line-continuation argument bugs)
train_cmd=(
    aev-plig-train
    --model "${MODEL}"
    --dataset "${DATASET}"
    --seed "\${SEED}"
    --timestamp "TEST_${TIMESTAMP}"
    --activation_function "${ACTIVATION}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --head "${HEAD}"
    --hidden_dim "${HIDDEN_DIM}"
    --lr "${LR}"
)
[[ "\${USE_WANDB:-0}" == "1" ]] && train_cmd+=(--wandb --wandb_project "aev-plig-dev")
printf 'CMD: %q ' "\${train_cmd[@]}"; echo
"\${train_cmd[@]}"

echo ""
echo "✓ Seed \${SEED} quick test complete"
echo "✓ Model saved to: output/trained_models/${MODEL}_TEST_${TIMESTAMP}/model_seed_\${SEED}.model"
EOF
)

echo "========================================"
echo "✓ Submitted array job (${#SEEDS[@]} tasks)"
echo "========================================"
echo "Job ID: ${JOB_ID}  (tasks ${JOB_ID}_0 to ${JOB_ID}_$((${#SEEDS[@]}-1)))"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  ${PROJECT_ROOT}/slurm/logs/"
echo ""
echo "Expected runtime: ~5-10 minutes per task"
echo ""
echo "Output will be saved to:"
echo "  output/trained_models/${MODEL}_TEST_${TIMESTAMP}/"
echo "========================================"
