#!/bin/bash
# =============================================================================
# Submit parallel ensemble training jobs
# Usage: ./slurm/submit_training.sh
#
# Submits one SLURM job per seed for parallel training.
# All models share the same timestamp and are saved to:
#   models/{model}_{timestamp}/model_seed_{seed}.model
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting parallel ensemble training..."

# Read seeds from Python config (single source of truth)
SEEDS=($(python3 -c "from aev_plig.config import Config; print(' '.join(map(str, Config.ENSEMBLE_SEEDS)))"))

# Validate seeds (check for duplicates)
if ! python3 -c "from aev_plig.config import Config; Config.validate_ensemble_seeds()"; then
    echo "ERROR: Seed validation failed!"
    exit 1
fi

# Generate shared timestamp for all models
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

echo "Seeds: ${SEEDS[*]}"
echo "Count: ${#SEEDS[@]} models"
echo "Timestamp: $TIMESTAMP"
echo ""

# Submit one job per seed
JOB_IDS=()
for seed in "${SEEDS[@]}"; do
    JOB_ID=$(sbatch \
        --cluster=htc \
        --parsable \
        --export=ALL,SEED=$seed,TIMESTAMP=$TIMESTAMP \
        "$SCRIPT_DIR/jobs/03_train.sh" | cut -d';' -f1)

    JOB_IDS+=($JOB_ID)
    echo "  Seed $seed → Job $JOB_ID"
done

echo ""
echo "Submitted ${#SEEDS[@]} parallel jobs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER --cluster=htc"
echo "Logs:    tail -f logs/train_*.out"
