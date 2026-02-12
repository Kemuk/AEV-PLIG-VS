#!/bin/bash
#SBATCH --job-name=aev-data-verify
#SBATCH --cluster=htc
#SBATCH --partition=devel
#SBATCH --time=00:10:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/data_verify_%j.out
#SBATCH --error=logs/data_verify_%j.err
# =============================================================================
# Smoke test: create datasets (quick test mode), verify .pt roundtrip,
# DataLoader compatibility, and core integration tests.
# =============================================================================
set -e

source "$(dirname "$0")/../../config.sh"

mkdir -p logs

echo "=== 1) Create quick-test dataset artifacts ==="
QUICK_TEST=1 python create_pytorch_data.py

echo "=== 2) Verify .pt artifact can be loaded ==="
python - <<'PY'
import torch

data = torch.load('data/processed/quick_test_test.pt', weights_only=False)
assert len(data) > 0, 'quick_test_test.pt contains no graphs'
first = data[0]
print(f'graphs={len(data)} node_dim={first.x.shape[1]} edge_dim={first.edge_attr.shape[1]}')
PY

echo "=== 3) Verify DataLoader works with plain list[Data] ==="
python - <<'PY'
import torch
from torch_geometric.loader import DataLoader

data = torch.load('data/processed/quick_test_test.pt', weights_only=False)
loader = DataLoader(data, batch_size=min(16, len(data)), shuffle=False)
batch = next(iter(loader))
print(f'batch_nodes={batch.x.shape[0]} batch_edges={batch.edge_index.shape[1]}')
PY

echo "=== 4) Run focused integration tests ==="
pytest tests/integration/test_dataset_creation.py tests/integration/test_prediction_e2e.py

EXIT_CODE=$?

echo ""
echo "========================================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ VERIFY PASSED"
else
    echo "✗ VERIFY FAILED (exit code: $EXIT_CODE)"
fi
echo "========================================================================"

exit $EXIT_CODE
