# Parallel Ensemble Training Plan

## 🎯 Goal
Replace sequential ensemble training with parallel jobs to utilize multiple GPUs efficiently.

**Current:** 1 job trains 10 models sequentially (~40 hours on 1 GPU)
**Target:** 10 jobs train 10 models in parallel (~4 hours on 10 GPUs)

---

## 📊 Current Architecture Analysis

### **Training Script Behavior** (`scripts/train.py`)

**Line 63-111:** Loops through all seeds from `Config.ENSEMBLE_SEEDS`:
```python
ensemble_seeds = Config.ENSEMBLE_SEEDS  # [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]

for i, seed in enumerate(ensemble_seeds):
    # Train one model with this seed
    # Save as: {timestamp}_model_{model}_{dataset}_{i}.model
```

**Problem:** No way to train a single model with one seed!
**Missing:** `--seed` argument to override ensemble mode

---

## 🔧 Proposed Architecture

### **Three-Layer System**

```
┌─────────────────────────────────────────────────────────┐
│  1. Submission Script (slurm/submit_training.sh)      │
│     - Reads seeds from Config or env var                │
│     - Submits N parallel jobs (one per seed)           │
│     - Handles test vs production mode                   │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  2. Single-Seed Job (slurm/jobs/03_train_single.sh)   │
│     - SLURM job script for ONE model                    │
│     - Receives SEED, ENSEMBLE_INDEX via env vars       │
│     - Calls train.py with --seed argument               │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│  3. Training Script (scripts/train.py)                 │
│     - NEW: --seed argument for single-model mode        │
│     - OLD: no --seed = ensemble mode (all seeds)        │
│     - Backwards compatible                              │
└─────────────────────────────────────────────────────────┘
```

---

## 📝 Detailed Component Specs

### **1. Modified Training Script** (`scripts/train.py`)

#### **New Arguments:**
```python
parser.add_argument('--seed', type=int, default=None,
                    help='Train single model with this seed (overrides ensemble mode)')
parser.add_argument('--ensemble_index', type=int, default=None,
                    help='Model index in ensemble (for filename, e.g., 0-9)')
```

#### **Modified Logic:**
```python
def train_ensemble(args):
    # ...existing setup...

    # NEW: Single-seed mode
    if args.seed is not None:
        ensemble_seeds = [args.seed]
        ensemble_index = args.ensemble_index if args.ensemble_index is not None else 0
        print(f"Single-seed mode: training model with seed {args.seed}")
    else:
        # Original ensemble mode
        ensemble_seeds = Config.ENSEMBLE_SEEDS
        print(f"Ensemble mode: training {len(ensemble_seeds)} models")

    for i, seed in enumerate(ensemble_seeds):
        # Use provided ensemble_index in single-seed mode
        model_index = ensemble_index if args.seed is not None else i

        # Save as: {timestamp}_model_{model}_{dataset}_{model_index}.model
        model_file_name = f"{timestr}_model_{args.model}_{args.dataset}_{model_index}.model"
        # ...rest of training loop...
```

**Key Design Decisions:**
- ✅ **Backwards compatible:** No `--seed` = ensemble mode (original behavior)
- ✅ **Single-seed mode:** `--seed 100` trains only that seed
- ✅ **Consistent naming:** `ensemble_index` ensures filenames match (e.g., `_0.model`, `_1.model`)
- ✅ **Shared timestamp:** All parallel jobs use same timestamp prefix (passed via env var)

---

### **2. Single-Seed Job Script** (`slurm/jobs/03_train_single.sh`)

```bash
#!/bin/bash
#SBATCH --job-name=aev-train-s%a
#SBATCH --cluster=htc
#SBATCH --partition=long
#SBATCH --time=24:00:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --output=logs/train_seed_%a_%j.out
#SBATCH --error=logs/train_seed_%a_%j.err
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Single-seed ensemble training job.
# Called by submit_training.sh - do not submit directly!
#
# Required environment variables:
#   SEED            - Random seed for this model (e.g., 100)
#   ENSEMBLE_INDEX  - Model index in ensemble (e.g., 0-9)
#   TIMESTAMP       - Shared timestamp for all ensemble members
#   EPOCHS          - Number of epochs (default: 200)
# =============================================================================

source slurm/config.sh

# Validate required env vars
: "${SEED:?Error: SEED not set}"
: "${ENSEMBLE_INDEX:?Error: ENSEMBLE_INDEX not set}"
: "${TIMESTAMP:?Error: TIMESTAMP not set}"

# Optional overrides
EPOCHS="${EPOCHS:-200}"

echo "===== Single-Seed Training ====="
echo "Seed:           $SEED"
echo "Ensemble index: $ENSEMBLE_INDEX"
echo "Timestamp:      $TIMESTAMP"
echo "Epochs:         $EPOCHS"
echo "Model:          $MODEL_NAME"
echo "Dataset:        $DATASET_NAME"
echo "================================"

python scripts/train.py \
    --model "$MODEL_NAME" \
    --dataset "$DATASET_NAME" \
    --seed "$SEED" \
    --ensemble_index "$ENSEMBLE_INDEX" \
    --epochs "$EPOCHS"

echo "Training completed for seed $SEED (index $ENSEMBLE_INDEX)"
```

**Key Features:**
- `%a` in job name = array task ID (if using job arrays)
- Validates required env vars (fail fast if missing)
- Passes seed, index, and epochs to train.py
- Logs include seed in filename for easy debugging

---

### **3. Submission Script** (`slurm/submit_training.sh` - REPLACE CURRENT)

```bash
#!/bin/bash
# =============================================================================
# Submit parallel ensemble training jobs
#
# Usage:
#   ./slurm/submit_training.sh [OPTIONS]
#
# Options:
#   --test          Use devel partition, 2 epochs, first 3 seeds only
#   --seeds "S1 S2" Override seeds (space-separated)
#   --epochs N      Override number of epochs (default: 200)
#
# Environment Variables:
#   ENSEMBLE_SEEDS  Override seeds (space-separated, e.g., "100 123 15")
#   EPOCHS          Override epochs
#
# Examples:
#   # Production: train all 10 models in parallel
#   ./slurm/submit_training.sh
#
#   # Test: quick devel run with 3 seeds
#   ./slurm/submit_training.sh --test
#
#   # Custom seeds
#   ./slurm/submit_training.sh --seeds "100 200 300"
#
#   # Custom epochs
#   ./slurm/submit_training.sh --epochs 50
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
# Parse Arguments
# =============================================================================
TEST_MODE=false
CUSTOM_SEEDS=""
CUSTOM_EPOCHS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --test)
            TEST_MODE=true
            shift
            ;;
        --seeds)
            CUSTOM_SEEDS="$2"
            shift 2
            ;;
        --epochs)
            CUSTOM_EPOCHS="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--test] [--seeds \"S1 S2\"] [--epochs N]"
            exit 1
            ;;
    esac
done

# =============================================================================
# Determine Seeds
# =============================================================================
if [[ -n "$CUSTOM_SEEDS" ]]; then
    # Command-line override
    SEEDS=($CUSTOM_SEEDS)
    echo "Using custom seeds from --seeds: ${SEEDS[*]}"
elif [[ -n "${ENSEMBLE_SEEDS:-}" ]]; then
    # Environment variable override
    SEEDS=($ENSEMBLE_SEEDS)
    echo "Using custom seeds from \$ENSEMBLE_SEEDS: ${SEEDS[*]}"
else
    # Default: read from Python config
    SEEDS=($(python3 -c "from aev_plig.config import Config; print(' '.join(map(str, Config.ENSEMBLE_SEEDS)))"))
    echo "Using default seeds from aev_plig/config.py: ${SEEDS[*]}"
fi

# =============================================================================
# Test Mode Overrides
# =============================================================================
if [[ "$TEST_MODE" == true ]]; then
    echo ""
    echo "========================================="
    echo "  TEST MODE ENABLED"
    echo "========================================="
    echo "- Partition: devel (10 min max)"
    echo "- Epochs: 2 (quick validation)"
    echo "- Seeds: first 3 only"
    echo "========================================="
    echo ""

    # Use only first 3 seeds
    SEEDS=("${SEEDS[@]:0:3}")
    PARTITION="devel"
    TIME="00:10:00"
    EPOCHS=2
else
    # Production mode
    PARTITION="long"
    TIME="24:00:00"
    EPOCHS="${CUSTOM_EPOCHS:-${EPOCHS:-200}}"
fi

# =============================================================================
# Generate Shared Timestamp
# =============================================================================
# All ensemble members must use the same timestamp prefix for model files
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

echo ""
echo "========================================="
echo "  Parallel Ensemble Training"
echo "========================================="
echo "Seeds:     ${SEEDS[*]}"
echo "Count:     ${#SEEDS[@]} models"
echo "Epochs:    $EPOCHS"
echo "Partition: $PARTITION"
echo "Time:      $TIME"
echo "Timestamp: $TIMESTAMP"
echo "========================================="
echo ""

# =============================================================================
# Submit Jobs
# =============================================================================
JOB_IDS=()

for i in "${!SEEDS[@]}"; do
    seed="${SEEDS[$i]}"

    # Submit job with environment variables
    JOB_ID=$(sbatch \
        --cluster=htc \
        --partition="$PARTITION" \
        --time="$TIME" \
        --parsable \
        --export=SEED=$seed,ENSEMBLE_INDEX=$i,TIMESTAMP=$TIMESTAMP,EPOCHS=$EPOCHS \
        "$SCRIPT_DIR/jobs/03_train_single.sh" | cut -d';' -f1)

    JOB_IDS+=($JOB_ID)
    echo "  [${i}] Seed $seed → Job $JOB_ID"
done

echo ""
echo "========================================="
echo "  Submitted ${#SEEDS[@]} parallel jobs"
echo "========================================="
echo ""
echo "Monitor progress:"
echo "  squeue -u \$USER --cluster=htc"
echo ""
echo "View logs:"
echo "  tail -f logs/train_seed_*_${JOB_IDS[0]}.out"
echo ""
echo "Wait for completion:"
echo "  squeue -u \$USER --cluster=htc | grep aev-train"
echo ""
echo "Job IDs: ${JOB_IDS[*]}"
echo ""
```

**Key Features:**
- ✅ **Default seeds from Config** - Single source of truth
- ✅ **Override options** - Command-line or env var
- ✅ **Test mode** - `--test` for quick devel validation
- ✅ **Shared timestamp** - All models use same prefix
- ✅ **Clear logging** - Shows exactly what's running
- ✅ **Easy monitoring** - Provides helpful commands

---

## 🧪 Testing Strategy

### **Phase 1: Devel Partition (10 min limit)**

**Test with 3 seeds, 2 epochs:**
```bash
cd $DATA/AEV-PLIG-VS
./slurm/submit_training.sh --test
```

**What this tests:**
- ✅ Job submission works
- ✅ Environment variables pass correctly
- ✅ Seed argument works
- ✅ Training loop runs
- ✅ Model files saved correctly
- ✅ Naming convention matches (indices 0, 1, 2)

**Expected behavior:**
- 3 jobs submitted to `devel` partition
- Each runs ~5-10 minutes (2 epochs)
- Outputs: `{timestamp}_model_{model}_{dataset}_0.model`, `_1.model`, `_2.model`
- All logs in `logs/train_seed_*_*.out`

**Validation checklist:**
```bash
# Check jobs submitted
squeue -u $USER --cluster=htc

# Watch first job's progress
tail -f logs/train_seed_0_*.out

# After completion, verify outputs
ls -lh output/trained_models/*_0.model
ls -lh output/trained_models/*_1.model
ls -lh output/trained_models/*_2.model

# Check all 3 models have same timestamp prefix
ls -1 output/trained_models/ | grep model | head -3
```

---

### **Phase 2: Short Partition (Full Validation)**

**Test with 3 seeds, 10 epochs:**
```bash
./slurm/submit_training.sh --seeds "100 123 15" --epochs 10
```

**What this tests:**
- ✅ Longer training runs
- ✅ Model convergence
- ✅ GPU utilization
- ✅ Memory usage
- ✅ No errors after extended runtime

**Expected duration:** ~30-60 minutes per model

---

### **Phase 3: Production Run (Full Ensemble)**

**All 10 seeds, 200 epochs:**
```bash
./slurm/submit_training.sh
```

**What this tests:**
- ✅ Full pipeline
- ✅ All 10 models train successfully
- ✅ Ensemble predictions work
- ✅ Reproducibility (same seeds = same results)

**Expected duration:** ~4 hours (if 10 GPUs available)

---

## 🔄 Backwards Compatibility

### **Keep Legacy Scripts (Optional)**

**Option A:** Replace `03_train.sh` with parallel version
- Rename current: `03_train.sh` → `03_train_sequential.sh` (backup)
- Create new: `03_train_single.sh` (parallel version)
- Update: `submit_training.sh` (use parallel submission)

**Option B:** Keep both modes
- Keep `03_train.sh` for sequential (backwards compat)
- Add `03_train_single.sh` for parallel
- Add `submit_training_parallel.sh` (new submission script)
- Update documentation to recommend parallel mode

**Recommendation:** Option A (replace) because:
- Parallel is strictly better (faster, more efficient)
- Reduces confusion (one way to do things)
- Old script still works (just slower)

---

## 📁 File Changes Summary

### **New Files:**
1. `slurm/jobs/03_train_single.sh` - Single-seed job script
2. (This plan doc - can be kept as reference)

### **Modified Files:**
1. `scripts/train.py` - Add `--seed` and `--ensemble_index` arguments
2. `slurm/submit_training.sh` - Replace with parallel submission logic
3. `slurm/tests/jobs/03_train_quick.sh` - Update to use new arguments (optional)

### **Deprecated (Optional):**
1. `slurm/jobs/03_train.sh` - Keep as `03_train_sequential.sh` or remove

---

## 📊 Performance Comparison

### **Sequential (Current)**
- **GPUs used:** 1
- **Time per model:** ~4 hours
- **Total time:** 10 models × 4 hours = **40 hours**
- **GPU efficiency:** 100% (1 GPU fully utilized)
- **Wall time:** 40 hours

### **Parallel (New)**
- **GPUs used:** 10 (ideal) or N available
- **Time per model:** ~4 hours
- **Total time:** max(4 hours) = **4 hours** (10× speedup!)
- **GPU efficiency:** 100% (all GPUs fully utilized)
- **Wall time:** 4 hours

### **Partial Parallelization (Realistic)**
- **GPUs used:** 3-5
- **Time:** ~8-13 hours
- **Speedup:** 3-5×

**Example with 3 GPUs:**
```
Batch 1: Seeds 0, 1, 2 (parallel) → 4 hours
Batch 2: Seeds 3, 4, 5 (parallel) → 4 hours
Batch 3: Seeds 6, 7, 8 (parallel) → 4 hours
Batch 4: Seed 9 (alone)          → 4 hours
Total: 16 hours (2.5× speedup)
```

---

## 🎯 Implementation Checklist

### **Phase 1: Code Changes**
- [ ] Modify `scripts/train.py` to add `--seed` and `--ensemble_index` args
- [ ] Add single-seed mode logic
- [ ] Test locally with `--seed 100 --ensemble_index 0 --epochs 2`

### **Phase 2: Job Scripts**
- [ ] Create `slurm/jobs/03_train_single.sh`
- [ ] Update `slurm/submit_training.sh` for parallel submission
- [ ] Add `--test` mode for devel partition

### **Phase 3: Testing**
- [ ] Devel test: `./slurm/submit_training.sh --test`
- [ ] Verify 3 models train successfully
- [ ] Check output files have correct naming
- [ ] Verify logs show correct seeds

### **Phase 4: Production**
- [ ] Full ensemble run: `./slurm/submit_training.sh`
- [ ] Monitor all 10 jobs
- [ ] Verify ensemble predictions work
- [ ] Document in README

### **Phase 5: Cleanup**
- [ ] Update main README with new submission method
- [ ] Add examples to `slurm/README.md`
- [ ] Archive old sequential script (optional)
- [ ] Update this plan status

---

## 💡 Future Enhancements

### **SLURM Job Arrays** (Advanced)
Instead of submitting N separate jobs, use a single job array:

```bash
# Submit one job with 10 array tasks
sbatch --array=0-9 slurm/jobs/03_train_array.sh
```

**Pros:**
- Cleaner queue (1 job instead of 10)
- Better SLURM accounting
- Automatic array task ID

**Cons:**
- Less flexible (all tasks same config)
- Harder to customize per-seed

**When to use:** For very large ensembles (50+ seeds)

### **Adaptive Seed Selection**
Read seeds from external file:
```bash
# seeds.txt
100
123
15
```

```bash
SEEDS=($(cat seeds.txt))
```

**Benefit:** Easy to customize per experiment without editing config

---

## 📚 Usage Examples

### **Example 1: Quick Test (Devel)**
```bash
# Test with 3 seeds, 2 epochs, devel partition
./slurm/submit_training.sh --test
```

### **Example 2: Custom Seeds**
```bash
# Train only specific seeds
./slurm/submit_training.sh --seeds "100 200 300 400 500"
```

### **Example 3: Reduced Epochs**
```bash
# Full ensemble but only 50 epochs
./slurm/submit_training.sh --epochs 50
```

### **Example 4: Production**
```bash
# Full ensemble, 200 epochs (default)
./slurm/submit_training.sh
```

### **Example 5: Environment Variable Override**
```bash
# Use env var for seeds
export ENSEMBLE_SEEDS="42 1337 9000"
./slurm/submit_training.sh
```

---

## 🔍 Troubleshooting

### **Problem: "SEED not set" error**
**Cause:** Job script called directly instead of via submit script
**Fix:** Always use `./slurm/submit_training.sh`, never `sbatch 03_train_single.sh`

### **Problem: Different timestamps on models**
**Cause:** Each job generated its own timestamp
**Fix:** Ensure TIMESTAMP env var is passed correctly

### **Problem: Models overwrite each other**
**Cause:** Multiple jobs have same ensemble_index
**Fix:** Check that indices are unique (0, 1, 2, ...)

### **Problem: Devel partition times out**
**Cause:** Devel has 10-minute limit
**Fix:** Use `--test` mode (2 epochs) or switch to `short` partition

---

**Status:** Planning complete - Ready for implementation
**Next Step:** Implement Phase 1 (Code Changes)
