#!/bin/bash
# =============================================================================
# Quick mixed precision training test (10 minutes)
#
# Tests mixed precision training with:
# - Both GATv2NetMixedPrecision and GATv2NetBayesianMixedPrecision
# - 1 seed each (functionality test, not ensemble)
# - 2 epochs (vs 200 in production)
# - Devel partition (10 min max)
#
# Usage:
#   ./slurm/tests/jobs/03_train_mixed_precision_quick.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Load configuration
source "$PROJECT_ROOT/slurm/config.sh"

# Training configuration
DATASET="${DATASET_NAME}"

# Hyperparameters (matching production training command)
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

# Test both mixed precision models with a single seed
MODELS=("GATv2NetMixedPrecision" "GATv2NetBayesianMixedPrecision")
SEED=100

# Create shared timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "QUICK TEST: Mixed Precision Training"
echo "========================================"
echo "Models:       ${MODELS[*]}"
echo "Dataset:      ${DATASET}"
echo "Seed:         ${SEED}"
echo "Epochs:       ${EPOCHS} (TEST)"
echo "Timestamp:    ${TIMESTAMP}"
echo "Partition:    ${PARTITION} (10 min max)"
echo "========================================"
echo ""

# Create log directory if needed
mkdir -p "$PROJECT_ROOT/slurm/logs"

# Submit parallel jobs
JOB_IDS=()
for MODEL in "${MODELS[@]}"; do
    echo "Submitting ${MODEL}..."

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=train_mp_${MODEL}
#SBATCH --cluster=${CLUSTER_NAME}
#SBATCH --partition=${PARTITION}
#SBATCH --time=${TIME_LIMIT}
#SBATCH --mem=${MEM}
#SBATCH --cpus-per-task=${CPUS}
#SBATCH --gres=gpu:${GPUS}
#SBATCH --output=${PROJECT_ROOT}/slurm/logs/%x_%j.out
#SBATCH --error=${PROJECT_ROOT}/slurm/logs/%x_%j.err
#SBATCH --chdir=${PROJECT_ROOT}

# Load environment
source ${PROJECT_ROOT}/slurm/config.sh

echo "========================================="
echo "QUICK TEST: Mixed Precision ${MODEL}"
echo "========================================="
echo "Node:      \$(hostname)"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Timestamp: ${TIMESTAMP}"
echo "Epochs:    ${EPOCHS}"
echo "========================================="
echo ""

# Train single model (array form avoids whitespace/line-continuation argument bugs)
train_cmd=(
    aev-plig-run
    --model "${MODEL}"
    --dataset "${DATASET}"
    --seed "${SEED}"
    --timestamp "${TIMESTAMP}"
    --activation_function "${ACTIVATION}"
    --batch_size "${BATCH_SIZE}"
    --epochs "${EPOCHS}"
    --head "${HEAD}"
    --hidden_dim "${HIDDEN_DIM}"
    --lr "${LR}"
)
printf 'CMD: %q ' "\${train_cmd[@]}"; echo
"\${train_cmd[@]}"

echo ""
echo "Mixed precision test complete for ${MODEL}"
echo "Model saved to: output/trained_models/${MODEL}_${TIMESTAMP}/model_seed_${SEED}.model"
EOF
)

    JOB_IDS+=("${JOB_ID}")
    echo "  -> Job ID: ${JOB_ID}"
done

echo ""
echo "========================================"
echo "Submitted ${#JOB_IDS[@]} mixed precision test jobs"
echo "========================================"
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in:"
echo "  ${PROJECT_ROOT}/slurm/logs/"
echo ""
echo "Output will be saved to:"
echo "  output/trained_models/<model>_${TIMESTAMP}/"
echo "========================================"
