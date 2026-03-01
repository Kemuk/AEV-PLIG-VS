#!/bin/bash
# =============================================================================
# Quick sweep test (devel partition, 10 min limit).
#
# Differences from production (slurm/jobs/05_sweep.sh):
# - 3 agents only (vs 50)
# - 2 epochs per run (set in sweeps/gatv2net_sweep_quick.yaml command section)
# - devel partition (10 min max)
# - W&B project: aev-plig-dev (not aev-plig-vs)
# - Creates its own sweep automatically (no pre-existing SWEEP_ID required)
#
# Tests the full W&B agent ←→ controller handshake and train_model() code path.
# Jobs may timeout — this is expected on the devel partition.
#
# Usage:
#   ./slurm/tests/jobs/05_sweep_quick.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# Quick test settings
NUM_AGENTS=3
PARTITION="${PARTITION_DEVEL}"
TIME_LIMIT="00:10:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1
WANDB_PROJECT="aev-plig-dev"

mkdir -p "$PROJECT_ROOT/slurm/logs"

echo "========================================"
echo "QUICK TEST: W&B Sweep Agent"
echo "========================================"
echo "Agents:     ${NUM_AGENTS} (TEST)"
echo "Epochs:     2 (TEST — set in sweep_quick.yaml)"
echo "Partition:  ${PARTITION} (10 min max)"
echo "W&B:        ${WANDB_PROJECT}"
echo "========================================"
echo ""

# Create a fresh test sweep (self-contained — no pre-existing sweep required)
echo "Creating test sweep from sweeps/gatv2net_sweep_quick.yaml ..."
SWEEP_OUTPUT=$(cd "$PROJECT_ROOT" && wandb sweep sweeps/gatv2net_sweep_quick.yaml \
    --project "${WANDB_PROJECT}" --name "quick_test_$(date +%Y%m%d_%H%M%S)" 2>&1)
echo "$SWEEP_OUTPUT"

AGENT_TARGET=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=wandb agent )[^\s]+' | tail -1)
[[ -z "$AGENT_TARGET" ]] && { echo "ERROR: Could not extract agent target. Check W&B login."; exit 1; }

echo ""
echo "Agent target: ${AGENT_TARGET}"
echo ""

JOB_ID=$(sbatch --parsable --array=1-${NUM_AGENTS} <<EOF
#!/bin/bash
#SBATCH --job-name=sweep_agent_quick
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
echo "QUICK TEST: W&B Sweep Agent"
echo "========================================="
echo "Node:     \$(hostname)"
echo "Job ID:   \${SLURM_JOB_ID} (task \${SLURM_ARRAY_TASK_ID})"
echo "Sweep:    ${AGENT_TARGET}"
echo "Epochs:   2 (quick test)"
echo "========================================="
echo ""

wandb agent ${AGENT_TARGET} --count 1

echo ""
echo "✓ Quick test agent complete (task \${SLURM_ARRAY_TASK_ID})"
EOF
)

echo "========================================"
echo "✓ Submitted array job (${NUM_AGENTS} tasks)"
echo "========================================"
echo "Job ID: ${JOB_ID}  (tasks ${JOB_ID}_1 to ${JOB_ID}_${NUM_AGENTS})"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER --cluster=${CLUSTER_NAME}"
echo ""
echo "View logs in:"
echo "  ${PROJECT_ROOT}/slurm/logs/"
echo ""
echo "NOTE: devel partition has a 10 min limit."
echo "      Jobs may timeout — this is expected."
echo "      Goal is to verify W&B handshake and train_model() code path."
echo "========================================"
