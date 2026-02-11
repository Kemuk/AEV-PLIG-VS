# Minimal Parallel Training Plan

## 🎯 Goal
Simple parallel ensemble training. No fancy flags, no renaming files, just make it work.

---

## 📝 Changes Required (3 files)

### **1. Modify `scripts/train.py`**

Add two optional arguments for single-seed mode:

```python
# In parse_args() function, add:
parser.add_argument('--seed', type=int, default=None,
                   help='Train single model with this seed (default: train all seeds)')
parser.add_argument('--ensemble_index', type=int, default=0,
                   help='Model index for filename (0-9)')
```

**Modify train_ensemble() logic:**

```python
def train_ensemble(args):
    # ... existing setup ...

    # NEW: Check if single-seed mode
    if args.seed is not None:
        # Single seed mode (for parallel jobs)
        ensemble_seeds = [args.seed]
        start_index = args.ensemble_index
        print(f"Single-seed mode: training seed {args.seed} as model #{start_index}")
    else:
        # Original ensemble mode (all seeds sequentially)
        ensemble_seeds = Config.ENSEMBLE_SEEDS
        start_index = 0
        print(f"Ensemble mode: training {len(ensemble_seeds)} models")

    for i, seed in enumerate(ensemble_seeds):
        model_index = start_index + i  # Use provided index in single-seed mode

        # ... rest of training loop ...

        # Save with correct index
        model_file_name = f"{timestr}_model_{args.model}_{args.dataset}_{model_index}.model"
        # ... continue ...
```

**That's it!** No breaking changes, fully backwards compatible.

---

### **2. Modify `slurm/submit_training.sh`**

Replace with simple parallel submission:

```bash
#!/bin/bash
# =============================================================================
# Submit parallel ensemble training jobs
# Usage: ./slurm/submit_training.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Submitting parallel ensemble training..."

# Read seeds from Python config (single source of truth)
SEEDS=($(python3 -c "from aev_plig.config import Config; print(' '.join(map(str, Config.ENSEMBLE_SEEDS)))"))

# Generate shared timestamp for all models
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo "Seeds: ${SEEDS[*]}"
echo "Count: ${#SEEDS[@]} models"
echo "Timestamp: $TIMESTAMP"
echo ""

# Submit one job per seed
JOB_IDS=()
for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"

    JOB_ID=$(sbatch \
        --cluster=htc \
        --parsable \
        --export=ALL,SEED=$seed,ENSEMBLE_INDEX=$i,TIMESTAMP=$TIMESTAMP \
        "$SCRIPT_DIR/jobs/03_train.sh" | cut -d';' -f1)

    JOB_IDS+=($JOB_ID)
    echo "  [${i}] Seed $seed → Job $JOB_ID"
done

echo ""
echo "Submitted ${#SEEDS[@]} parallel jobs: ${JOB_IDS[*]}"
echo ""
echo "Monitor: squeue -u \$USER --cluster=htc"
echo "Logs:    tail -f logs/train_*.out"
```

**No dependencies, no pipelines, just pure parallel training.**

---

### **3. Modify `slurm/jobs/03_train.sh`**

Make it handle both modes (sequential vs single-seed):

```bash
#!/bin/bash
#SBATCH --job-name=aev-train
#SBATCH --cluster=htc
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Train model ensemble.
#
# Single-seed mode (parallel): Set SEED and ENSEMBLE_INDEX env vars
# Ensemble mode (sequential): Run without env vars (trains all seeds)
# =============================================================================

source slurm/config.sh

echo "Training model: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"

# Check if running in single-seed mode (parallel job)
if [[ -n "${SEED:-}" ]]; then
    echo "Single-seed mode: Seed $SEED, Index ${ENSEMBLE_INDEX:-0}"

    python scripts/train.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME" \
        --seed "$SEED" \
        --ensemble_index "${ENSEMBLE_INDEX:-0}"
else
    echo "Ensemble mode: training all seeds sequentially"

    python scripts/train.py \
        --model "$MODEL_NAME" \
        --dataset "$DATASET_NAME"
fi

echo "Training completed. Models saved to output/trained_models/"
```

**Backwards compatible:** No SEED env var = original behavior.

---

### **4. Replace `slurm/tests/jobs/03_train_quick.sh`**

Make it ACTUALLY quick (runs in <10 minutes on devel):

```bash
#!/bin/bash
#SBATCH --job-name=aev-train-quick
#SBATCH --cluster=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_quick_%j.out
#SBATCH --error=logs/train_quick_%j.err
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Quick test: 1 seed, 2 epochs, devel partition (10 min max)
# Validates SLURM setup, GPU access, and training pipeline.
# =============================================================================

source slurm/config.sh

echo "===== Quick Training Test ====="
echo "Partition: devel (10 min max)"
echo "Epochs: 2"
echo "Seed: 100 (first seed only)"
echo "================================"

python scripts/train.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --seed 100 \
    --ensemble_index 0 \
    --epochs 2

echo "Quick test completed. Check output/trained_models/ for model file."
```

**Key changes:**
- ✅ Devel partition (10 min limit)
- ✅ Only 1 seed (seed 100)
- ✅ Only 2 epochs
- ✅ Uses new --seed argument
- ✅ Completes in ~5-8 minutes

---

## 🧪 Testing Workflow

### **Step 1: Test quick script (validates single-seed mode)**
```bash
cd $DATA/AEV-PLIG-VS
sbatch --cluster=htc slurm/tests/jobs/03_train_quick.sh
```

**Expected:**
- Runs on devel partition
- Trains seed 100, 2 epochs
- Completes in ~5-8 minutes
- Creates: `{timestamp}_model_{model}_{dataset}_0.model`

---

### **Step 2: Test parallel submission (validates parallel mode)**
```bash
cd $DATA/AEV-PLIG-VS

# Edit submit_training.sh temporarily to use only 3 seeds
# Or just run it and cancel after verifying submission works
./slurm/submit_training.sh

# Check jobs submitted
squeue -u $USER --cluster=htc

# Cancel if testing
scancel <job_ids>
```

**Expected:**
- Submits 10 jobs (one per seed)
- Each job shows in queue
- Job names all show "aev-train"
- Logs show correct seed/index

---

### **Step 3: Production run (full ensemble)**
```bash
./slurm/submit_training.sh
```

**Expected:**
- 10 parallel jobs
- Each trains 1 seed, 200 epochs
- Completes in ~4 hours (if 10 GPUs available)
- Creates 10 model files with same timestamp prefix

---

## 📊 File Changes Summary

| File | Change | Lines Changed |
|------|--------|---------------|
| `scripts/train.py` | Add `--seed`, `--ensemble_index` args | ~15 lines |
| `slurm/submit_training.sh` | Replace with parallel loop | ~35 lines |
| `slurm/jobs/03_train.sh` | Add single-seed mode check | ~10 lines |
| `slurm/tests/jobs/03_train_quick.sh` | Replace with tiny test | ~25 lines |

**Total:** ~85 lines of changes across 4 files

---

## ✅ Advantages of This Approach

1. **No renaming** - All existing files keep their names
2. **Backwards compatible** - Original sequential mode still works
3. **Simple** - No argument parsing, no fancy flags
4. **Testable** - Quick test actually runs in <10 minutes
5. **Parsimonious** - Minimal bash, straightforward logic

---

## 🔄 Backwards Compatibility

### **Old way (still works):**
```bash
sbatch slurm/jobs/03_train.sh  # Trains all 10 seeds sequentially
```

### **New way (parallel):**
```bash
./slurm/submit_training.sh     # Trains all 10 seeds in parallel
```

Both work! No breaking changes.

---

## 💡 Key Design Decisions

### **1. Why modify train.py instead of wrapper script?**
- ✅ Cleaner - logic in one place
- ✅ Testable - can test with `python scripts/train.py --seed 100`
- ✅ Flexible - works from anywhere (SLURM, local, notebook)

### **2. Why use env vars instead of CLI args?**
- ✅ SLURM standard - `sbatch --export=VAR=value`
- ✅ Simpler job script - no argument parsing in bash
- ✅ Clear separation - submission script sets vars, job script reads them

### **3. Why shared timestamp?**
- ✅ Ensemble coherence - all models belong together
- ✅ Prediction compatibility - scripts expect matching timestamps
- ✅ Organization - easy to identify ensemble batches

---

## 🚀 Ready to Implement?

This is much simpler than the full plan. Just 4 file modifications, no new files, no complex argument parsing.

**Estimated time:** 15 minutes to implement, 10 minutes to test.

**Should I proceed?**
