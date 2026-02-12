#!/bin/bash
# =============================================================================
# Submit parallel training jobs for ensemble
#
# This script submits multiple SLURM jobs in parallel, each training a single
# model with a different random seed. All models share the same timestamp and
# are saved to: output/trained_models/{model}_{timestamp}/
#
# Usage:
#   ./slurm/jobs/03_train.sh
#
# Configuration is loaded from slurm/config.sh
# =============================================================================

# Get directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Load configuration
source "$PROJECT_ROOT/slurm/config.sh"

# Training configuration
MODEL="${MODEL_NAME}"
DATASET="${DATASET_NAME}"

# Hyperparameters (matching original training command)
ACTIVATION="leaky_relu"
BATCH_SIZE=128
EPOCHS=200
HEAD=3
HIDDEN_DIM=256
LR=0.00012291937615434127

# SLURM settings
PARTITION="${PARTITION_LONG}"
TIME_LIMIT="24:00:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1

# Ensemble seeds from config.py
SEEDS=(100 123 15 257 2 2012 3752 350 843 621)

# Create shared timestamp for ensemble
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "========================================"
echo "Parallel Training Job Submission"
echo "========================================"
echo "Model:        ${MODEL}"
echo "Dataset:      ${DATASET}"
echo "Seeds:        ${#SEEDS[@]} models"
echo "Timestamp:    ${TIMESTAMP}"
echo "Partition:    ${PARTITION}"
echo "Output dir:   output/trained_models/${MODEL}_${TIMESTAMP}/"
echo "========================================"
echo ""

# Create log directory if needed
mkdir -p "$PROJECT_ROOT/slurm/logs"

# Submit parallel jobs
JOB_IDS=()
for SEED in "${SEEDS[@]}"; do
    echo "Submitting seed ${SEED}..."

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=train_${MODEL}_s${SEED}
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
echo "Training Model: Seed ${SEED}"
echo "========================================="
echo "Node:      \$(hostname)"
echo "Job ID:    \${SLURM_JOB_ID}"
echo "Timestamp: ${TIMESTAMP}"
echo "========================================="
echo ""

# Train single model
aev-plig-train \\
    --model ${MODEL} \\
    --dataset ${DATASET} \\
    --seed ${SEED} \\
    --timestamp ${TIMESTAMP} \\
    --activation_function ${ACTIVATION} \\
    --batch_size ${BATCH_SIZE} \\
    --epochs ${EPOCHS} \\
    --head ${HEAD} \\
    --hidden_dim ${HIDDEN_DIM} \\
    --lr ${LR}

echo ""
echo "✓ Seed ${SEED} training complete"
echo "✓ Model saved to: output/trained_models/${MODEL}_${TIMESTAMP}/model_seed_${SEED}.model"
EOF
)

    JOB_IDS+=("${JOB_ID}")
    echo "  → Job ID: ${JOB_ID}"
done

echo ""
echo "========================================"
echo "✓ Submitted ${#SEEDS[@]} parallel jobs"
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
echo "  output/trained_models/${MODEL}_${TIMESTAMP}/"
echo "========================================"
