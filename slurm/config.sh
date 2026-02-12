#!/bin/bash
# =============================================================================
# AEV-PLIG SLURM Configuration
# Common environment setup and cluster settings for all SLURM jobs.
# Source this file at the top of every job script:
#   source "$(dirname "$0")/../config.sh"
# =============================================================================

# ======================== Cluster Settings ========================
CLUSTER_NAME="htc"
PARTITION_SHORT="short"              # 12h max
PARTITION_LONG="long"                # unlimited
PARTITION_DEVEL="devel"              # 10 min max (testing only)
PARTITION_INTERACTIVE="interactive"  # 4h max

# ======================== Resource Presets ========================
MEM_STANDARD="20GB"
MEM_LARGE="32GB"
CPUS_STANDARD=8
CPUS_GRAPH_GEN=4

# ======================== Runtime Choices ========================
# These are the only "what to run" settings.
# All hyperparameters (lr, epochs, batch_size, hidden_dim, etc.) use
# Python defaults from aev_plig/config.py — single source of truth.
# Override in individual job scripts with CLI flags if needed.
MODEL_NAME="GATv2NetBayesian"
DATASET_NAME="pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"

# ======================== Paths ========================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LOG_DIR="$PROJECT_ROOT/logs"

# ======================== Environment Setup ========================
module load Anaconda3
module load CUDA

# Isolate conda env from user-site packages (~/.local)
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1

# Activate conda environment and ensure its binaries are on PATH
CONDA_ENV="$DATA/envs/aev-plig"
export PATH="$CONDA_ENV/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_ENV/lib:${LD_LIBRARY_PATH:-}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"

# Re-assert env path after activation/module hooks and clear shell command cache
export PATH="$CONDA_ENV/bin:$PATH"
hash -r

# Create directories
mkdir -p "$LOG_DIR"

# Move to project root
cd "$PROJECT_ROOT"

# ======================== Diagnostics ========================
echo "===== AEV-PLIG Environment ====="
echo "Date:         $(date)"
echo "Node:         $(hostname)"
echo "Project root: $PROJECT_ROOT"
echo "Conda env:    $CONDA_ENV"
echo "Python:       $(which python)"
echo "Python ver:   $(python -V 2>&1)"
echo "Train entry:  $(command -v aev-plig-train || echo 'not found')"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU:          $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
fi
echo "================================"
