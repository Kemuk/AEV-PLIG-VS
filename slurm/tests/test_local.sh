#!/bin/bash
# =============================================================================
# Test AEV-PLIG in an interactive session.
# Run inside an interactive allocation:
#   srun --cluster=htc --partition=interactive --mem=20GB --cpus-per-task=4 --gres=gpu:1 --time=04:00:00 --pty bash
#   ./slurm/tests/test_local.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../config.sh"

echo ""
echo "===== Test 1: Imports ====="
python -c "
from aev_plig import GATv2Net, get_model, Config
from aev_plig.models import GATv2NetBayesian
import torch, torch_geometric, torch_scatter
print('All imports OK')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
"

echo ""
echo "===== Test 2: Data Loading ====="
python -c "
import pandas as pd
from aev_plig.config import Config

df = pd.read_csv('data/example_dataset.csv')
print(f'Example dataset: {len(df)} rows')
print(f'Columns: {list(df.columns)}')

atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=',')
print(f'Atom keys: {len(atom_keys)} entries')
"

echo ""
echo "===== Test 3: Model Creation ====="
python -c "
import torch
from aev_plig.models import get_model

# Test standard model
model = get_model('GATv2Net', node_feature_dim=25, edge_feature_dim=4)
print(f'GATv2Net params: {sum(p.numel() for p in model.parameters()):,}')

# Test Bayesian model
model_b = get_model('GATv2NetBayesian', node_feature_dim=25, edge_feature_dim=4)
print(f'GATv2NetBayesian params: {sum(p.numel() for p in model_b.parameters()):,}')

print('Model creation OK')
"

echo ""
echo "===== Test 4: Example Prediction (CPU) ====="
python -c "
import subprocess, sys
result = subprocess.run(
    [sys.executable, 'scripts/predict.py',
     '--dataset_csv', 'data/example_dataset.csv',
     '--data_name', 'test_local',
     '--device', 'cpu',
     '--num_workers', '1'],
    capture_output=True, text=True, timeout=300
)
print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
if result.returncode != 0:
    print('STDERR:', result.stderr[-500:])
    sys.exit(1)
print('Prediction test OK')
" 2>&1 || echo "Test 4 failed (may need trained model — this is OK for initial setup)"

echo ""
echo "All local tests complete."
