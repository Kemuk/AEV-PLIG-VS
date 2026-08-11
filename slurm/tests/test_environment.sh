#!/bin/bash
# =============================================================================
# Validate that the HPC environment is correctly set up.
# Run this before submitting any SLURM jobs.
# Usage: ./slurm/tests/test_environment.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

PASS=0
FAIL=0

check() {
    local description="$1"
    shift
    if "$@" > /dev/null 2>&1; then
        echo "  PASS: $description"
        ((PASS++))
    else
        echo "  FAIL: $description"
        ((FAIL++))
    fi
}

echo ""
echo "===== Module & Environment ====="
check "Conda environment active"      test -n "$CONDA_DEFAULT_ENV"
check "Python available"              python --version
check "\$DATA is set"                 test -n "$DATA"
check "Conda env directory exists"    test -d "$CONDA_ENV"

echo ""
echo "===== Python Packages ====="
check "torch"              python -c "import torch"
check "torch_geometric"    python -c "import torch_geometric"
check "torch_scatter"      python -c "import torch_scatter"
check "rdkit"              python -c "from rdkit import Chem"
check "aev_plig"           python -c "import aev_plig"
check "torchani_mod"       python -c "import torchani_mod"
check "biopandas"          python -c "import biopandas"
check "qcelemental"        python -c "import qcelemental"

echo ""
echo "===== CUDA ====="
check "CUDA available to PyTorch"  python -c "import torch; assert torch.cuda.is_available()"
check "nvidia-smi"                 nvidia-smi

echo ""
echo "===== Data Files ====="
check "PDB_Atom_Keys.csv"         test -f "$PROJECT_ROOT/data/PDB_Atom_Keys.csv"
check "pdbbind_processed.csv"     test -f "$PROJECT_ROOT/data/pdbbind_processed.csv"
check "bindingnet_processed.csv"  test -f "$PROJECT_ROOT/data/bindingnet_processed.csv"
check "bindingdb_processed.csv"   test -f "$PROJECT_ROOT/data/bindingdb_processed.csv"
check "example_dataset.csv"       test -f "$PROJECT_ROOT/data/example_dataset.csv"

echo ""
echo "===== Scripts ====="
check "generate_pdbbind_graphs.py"    test -f "$PROJECT_ROOT/scripts/generate_pdbbind_graphs.py"
check "generate_bindingnet_graphs.py" test -f "$PROJECT_ROOT/scripts/generate_bindingnet_graphs.py"
check "generate_bindingdb_graphs.py"  test -f "$PROJECT_ROOT/scripts/generate_bindingdb_graphs.py"
check "create_pytorch_data.py"        test -f "$PROJECT_ROOT/create_pytorch_data.py"
check "aev-plig-train command"        command -v aev-plig-train
check "aev-plig-predict command"      command -v aev-plig-predict

echo ""
echo "================================"
echo "Results: $PASS passed, $FAIL failed"
echo "================================"

if [ "$FAIL" -gt 0 ]; then
    echo "Fix failures before submitting jobs."
    exit 1
fi

echo "Environment is ready."
