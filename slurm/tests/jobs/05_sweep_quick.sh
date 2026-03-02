#!/bin/bash
# =============================================================================
# Quick sweep test (devel partition, 10 min limit).
#
# Differences from production (slurm/jobs/05_sweep.sh):
# - 1 agent per sweep (vs 50)
# - 2 epochs per run (set in the *_quick.yaml command section)
# - devel partition (10 min max)
# - W&B project: aev-plig-dev (not aev-plig-vs)
# - Creates its own sweep automatically (no pre-existing SWEEP_ID required)
#
# Tests the full W&B agent ←→ controller handshake and train_model() code path.
# Jobs may timeout — this is expected on the devel partition.
#
# Usage:
#   ./slurm/tests/jobs/05_sweep_quick.sh                                      # all 3 archetypes
#   ./slurm/tests/jobs/05_sweep_quick.sh sweeps/ablation_preserver_quick.yaml  # one archetype
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

source "$PROJECT_ROOT/slurm/config.sh"

# Quick test settings
PARTITION="${PARTITION_DEVEL}"
TIME_LIMIT="00:10:00"
MEM="${MEM_STANDARD}"
CPUS="${CPUS_STANDARD}"
GPUS=1
WANDB_PROJECT="aev-plig-dev"

# Default: all 3 archetypes.  Pass a single YAML to test one.
if [[ $# -gt 0 ]]; then
    SWEEP_YAMLS=("$1")
else
    SWEEP_YAMLS=(
        sweeps/ablation_preserver_quick.yaml
        sweeps/ablation_explorer_quick.yaml
        sweeps/ablation_occam_quick.yaml
    )
fi

mkdir -p "$PROJECT_ROOT/slurm/logs"

echo "========================================"
echo "QUICK TEST: W&B Sweep Agent"
echo "========================================"
echo "Sweeps:     ${#SWEEP_YAMLS[@]}"
echo "Partition:  ${PARTITION} (10 min max)"
echo "W&B:        ${WANDB_PROJECT}"
echo "========================================"
echo ""

for SWEEP_YAML in "${SWEEP_YAMLS[@]}"; do
    echo "── ${SWEEP_YAML} ──"

    SWEEP_BASE=$(basename "${SWEEP_YAML}" .yaml | sed 's/^ablation_//' | sed 's/_quick$//')
    SWEEP_NAME="quick_${SWEEP_BASE}_$(date +%Y%m%d_%H%M%S)"

    SWEEP_OUTPUT=$(cd "$PROJECT_ROOT" && wandb sweep "${SWEEP_YAML}" \
        --project "${WANDB_PROJECT}" --name "${SWEEP_NAME}" 2>&1)
    echo "$SWEEP_OUTPUT"

    AGENT_TARGET=$(echo "$SWEEP_OUTPUT" | grep -oP '(?<=wandb agent )[^\s]+' | tail -1)
    [[ -z "$AGENT_TARGET" ]] && { echo "ERROR: Could not extract agent target. Check W&B login."; exit 1; }

    echo "Agent target: ${AGENT_TARGET}"

    JOB_ID=$(sbatch --parsable <<EOF
#!/bin/bash
#SBATCH --job-name=sweep_quick_${SWEEP_BASE}
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
echo "QUICK TEST: ${SWEEP_YAML}"
echo "========================================="
echo "Node:     \$(hostname)"
echo "Job ID:   \${SLURM_JOB_ID}"
echo "Sweep:    ${AGENT_TARGET}"
echo "========================================="
echo ""

wandb agent ${AGENT_TARGET} --count 1

echo ""
echo "✓ Quick test agent complete"
EOF
    )

    echo "Submitted job ${JOB_ID}"
    echo ""
done

echo "========================================"
echo "✓ All sweeps submitted"
echo "========================================"
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
