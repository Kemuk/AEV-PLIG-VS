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

echo "=== 2) Verify manifest-based shard artifacts can be loaded ==="
python - <<'PY'
import json
import torch
from pathlib import Path

dataset_root = Path('data/processed/quick_test')
split_dir = dataset_root / 'test'
manifest_path = split_dir / 'manifest.json'
assert manifest_path.exists(), f'missing manifest: {manifest_path}'

manifest = json.loads(manifest_path.read_text())
assert manifest['parts'], 'manifest has no parts'

first_part = torch.load(split_dir / manifest['parts'][0], weights_only=False)
assert len(first_part) > 0, 'first test shard contains no graphs'
first = first_part[0]
print(f"parts={len(manifest['parts'])} graphs_first_part={len(first_part)} node_dim={first.x.shape[1]} edge_dim={first.edge_attr.shape[1]}")
PY

echo "=== 3) Verify DataLoader works across all quick_test shards ==="
python - <<'PY'
import json
import torch
from pathlib import Path
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader

dataset_root = Path('data/processed/quick_test')
split_dir = dataset_root / 'test'
manifest = json.loads((split_dir / 'manifest.json').read_text())
parts = [torch.load(split_dir / name, weights_only=False) for name in manifest['parts']]
dataset = parts[0] if len(parts) == 1 else ConcatDataset(parts)
loader = DataLoader(dataset, batch_size=min(16, manifest['num_graphs_written']), shuffle=False)
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
