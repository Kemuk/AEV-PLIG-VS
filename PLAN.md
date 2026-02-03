# AEV-PLIG Development Plan

## Status
Last updated: 2026-02-02
Current phase: Core features complete, expanding test coverage

## Completed
- [x] Refactor codebase into modular package (v2.0)
- [x] Centralize configuration in `aev_plig/config.py`
- [x] Remove code duplication (~500 lines eliminated)
- [x] Create model registry for extensibility
- [x] Integration test suite (7 test files)
- [x] GitHub Actions CI workflow
- [x] Bayesian last layer (GATv2NetBayesian) + minimal tests
- [x] Bayesian training support (auto-detect in Trainer)

## Up Next
1. Download data script — HIGH
2. Unit tests (Phase 2) — MEDIUM
3. Regression tests (Phase 3) — LOW

## Backburner
- Dependency consolidation (setup.py extras)
- Migrate setup.py → pyproject.toml (see notes below)

---

## Planned: Download Data Script

Priority: HIGH
File: `scripts/download_data.sh`

### Purpose
Bash script to download training datasets with optional parallel downloads for Linux.

### Data Sources

| Dataset | URL | Archive | Extract To |
|---------|-----|---------|------------|
| PDBbind (refined) | `http://pdbbind.org.cn/download/PDBbind_v2020_refined.tar.gz` | tar.gz | `data/pdbbind/refined-set/` |
| PDBbind (general) | `http://pdbbind.org.cn/download/PDBbind_v2020_other_PL.tar.gz` | tar.gz | `data/pdbbind/general-set/` |
| BindingNet | `http://bindingnet.huanglab.org.cn/api/api/download/binding_database` | tar.gz | `data/bindingnet/from_chembl_client/` |
| BindingDB-DCS | `https://www.bindingdb.org/bind/chemsearch/marvin/SDFdownload.jsp?download_file=/rwd/data/surflex/surflex.tar` | tar | `data/bindingdb/surflex/` |

### Usage
```bash
# Download all datasets (sequential)
./scripts/download_data.sh

# Download with 4 parallel downloads (Linux)
./scripts/download_data.sh --threads 4

# Download only PDBbind
./scripts/download_data.sh --dataset pdbbind

# Download and extract
./scripts/download_data.sh --extract

# Full example
./scripts/download_data.sh --dataset all --threads 4 --extract
```

### Options
```
--dataset DATASET   Which dataset: pdbbind, bindingnet, bindingdb, all (default: all)
--threads N         Parallel downloads (default: 1)
--extract           Extract archives after download
--skip-existing     Skip if file already exists
--output-dir DIR    Base directory (default: data/)
-h, --help          Show help
```

### Directory Structure Created
```
data/
├── pdbbind/
│   ├── refined-set/      # PDBbind refined
│   └── general-set/      # PDBbind general
├── bindingnet/
│   └── from_chembl_client/
└── bindingdb/
    └── surflex/
```

### Implementation Notes
- Use `wget -c` for resume support
- Use `wget --progress=bar:force` for progress
- Parallel: background jobs with `&` and `wait`
- Auto-detect archive type (tar.gz vs tar)

---

## Planned: Testing

### Phase 1: Integration Tests (HIGH PRIORITY)
Target: Cover critical data flow paths

| Test | File | Modules Covered |
|------|------|-----------------|
| Single molecule → graph | test_graph_construction.py | loaders, features, graphs |
| Graphs → dataset | test_dataset_creation.py | graphs, datasets |
| Model forward pass | test_model_forward.py | datasets, models |
| Training step | test_training_loop.py | models, training |
| Validation metrics | test_training_loop.py | training |
| Prediction E2E | test_prediction_e2e.py | all prediction modules |
| Scaler round-trip | test_scaler.py | datasets |
| Ensemble aggregation | test_ensemble.py | prediction |

### Phase 2: Unit Tests (MEDIUM PRIORITY)
Target: 70% coverage per module

| Module | Test File | Key Functions |
|--------|-----------|---------------|
| config.py | test_config.py | Config defaults, path validation |
| features.py | test_features.py | one_of_k_encoding, atom_features, bond_features |
| loaders.py | test_loaders.py | load_protein_atoms, load_ligand_atoms, compute_aevs |
| graphs.py | test_graphs.py | create_graph edge cases |
| datasets.py | test_datasets.py | GraphDataset, GraphDatasetPredict |
| models.py | test_models.py | GATv2Net layers, get_model registry |
| training.py | test_training.py | rmse, pearson, spearman, concordance_index |

### Phase 3: Regression Tests (LOW PRIORITY)
- Golden outputs for 3ao4 molecule (graph structure, AEV vectors)
- Stored predictions for example_dataset.csv
- Model determinism with fixed seeds

---

## Planned: Bayesian Last Layer

Priority: HIGH
Depends on: GitHub Actions CI

### Motivation
Add uncertainty quantification to predictions. Useful for:
- Identifying low-confidence predictions
- Active learning / experimental prioritization
- Out-of-distribution detection

### Architecture Change
```
Current:
  GATv2 → GlobalPool → MLP(1024→512→256→1) → scalar

Proposed:
  GATv2 → GlobalPool → MLP(1024→512→256) → VBLL(256→1) → (mean, variance)
```

### Implementation Steps
1. Add `GATv2NetBayesian` class to `aev_plig/models.py`
2. Register in model registry: `MODEL_REGISTRY['GATv2NetBayesian'] = GATv2NetBayesian`
3. Modify `Trainer` to handle `(mean, var)` tuple outputs
4. Add Gaussian NLL loss: `0.5 * (log(var) + (y - mean)² / var)`
5. Update `Predictor` to return uncertainty estimates
6. Add uncertainty calibration metrics

### Libraries to Evaluate
- **VBLL (recommended)**: https://github.com/VectorInstitute/vbll
  - Drop-in replacement for final linear layer
  - Single forward pass (no sampling)
  - Paper: https://arxiv.org/html/2404.11599v1
- **Alternative**: Manual implementation with `nn.Linear` outputting (mean, log_var)

### Code Sketch
```python
# aev_plig/models.py
class GATv2NetBayesian(nn.Module):
    def __init__(self, ...):
        # Same backbone as GATv2Net
        self.conv_layers = ...
        self.pool = ...
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
        )
        # Bayesian head outputs mean and log_variance
        self.mean_head = nn.Linear(256, 1)
        self.logvar_head = nn.Linear(256, 1)

    def forward(self, data):
        # ... backbone forward ...
        features = self.mlp(pooled)
        mean = self.mean_head(features)
        var = F.softplus(self.logvar_head(features)) + 1e-6
        return mean, var
```

### Minimal Tests for Bayesian
| Test | Purpose |
|------|---------|
| test_bayesian_output_shape | Returns (mean, var) tuple |
| test_variance_positivity | var > 0 always |

Additional tests (later):
- test_uncertainty_calibration — ~68% within ±1σ on held-out data
- test_ood_uncertainty — Higher variance for dissimilar molecules

---

## File Structure Target

```
tests/
├── __init__.py
├── conftest.py              # Shared fixtures
├── pytest.ini               # Pytest configuration
│
├── integration/             # Phase 1 (current)
│   ├── __init__.py
│   ├── test_graph_construction.py
│   ├── test_dataset_creation.py
│   ├── test_model_forward.py
│   ├── test_training_loop.py
│   ├── test_prediction_e2e.py
│   ├── test_scaler.py
│   └── test_ensemble.py
│
├── unit/                    # Phase 2
│   └── ...
│
├── regression/              # Phase 3
│   ├── test_golden_outputs.py
│   └── golden_data/
│
└── fixtures/
    └── molecules/           # Small test molecules
```

---

## Configuration

### Coverage Target
- Overall: 70%
- Critical modules (features, graphs, models): 80%

### Test Dependencies
```
pytest>=7.0
pytest-cov>=4.0
```

### Running Tests
```bash
# All tests
pytest

# With coverage
pytest --cov=aev_plig --cov-report=html

# Integration only
pytest tests/integration/

# Skip slow tests
pytest -m "not slow"
```

---

## Planned: Dependency Consolidation

Priority: MEDIUM
Status: Deferred

### Current State
Two conda environment files exist:
- `aev-plig-linux.yml` — Linux + CUDA
- `aev-plig-mac.yml` — macOS CPU

Keep these for now as reference.

### Future Proposal
Use `setup.py` with `extras_require` for optional dependencies:

```python
# setup.py
extras_require={
    'cuda': [
        'torch-scatter',
        'torch-sparse',
    ],
    'dev': [
        'pytest>=7.0',
        'pytest-cov>=4.0',
    ],
},
```

### User Install Commands (future)
```bash
# CPU only (default)
pip install .

# With CUDA support
pip install .[cuda]

# For development (includes tests)
pip install -e .[dev]

# Everything
pip install -e .[cuda,dev]
```

### CUDA PyTorch Note
For CUDA-enabled PyTorch, users install PyTorch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install .[cuda]
```

---

## Planned: GitHub Actions CI

Priority: URGENT
Depends on: Nothing (do first)

### Workflow: `.github/workflows/ci.yml`

**Trigger:** Push to `main` only

**Strategy:** Matrix for Linux + macOS

```yaml
name: CI

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ${{ matrix.os }}
    strategy:
      fail-fast: false
      matrix:
        os: [ubuntu-latest, macos-latest]
        python-version: ["3.10"]

    steps:
      - Checkout code
      - Setup Python
      - Cache pip dependencies
      - Install PyTorch CPU
      - Install package: pip install -e .[dev]
      - Run tests: pytest --cov=aev_plig
```

### Key Decisions
| Decision | Choice | Rationale |
|----------|--------|-----------|
| Conda vs pip | pip | Faster in CI (~2-3 min saved) |
| CUDA in CI | No (CPU only) | Simpler, GitHub runners don't have GPU |
| Python version | 3.10 | Latest in setup.py classifiers |
| Coverage upload | Optional | Add Codecov later if needed |

### Estimated Run Time
~3-5 minutes per OS

### File Structure
```
.github/
└── workflows/
    └── ci.yml    # Single workflow with matrix
```

---

## Backburner: Migrate to pyproject.toml

Priority: LOW
Status: Deferred

### Why Migrate
- `setup.py` is legacy (PEP 517/518 deprecated it)
- `pyproject.toml` is declarative and cleaner
- Better build isolation

### Proposed Structure
```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "aev-plig"
version = "2.0.0"
requires-python = ">=3.9"

dependencies = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    # NOTE: torch-scatter/sparse removed - require manual install
    "rdkit>=2023.0.0",
    "torchani>=2.2.0",
    # ... other deps
]

[project.optional-dependencies]
dev = ["pytest>=7.0", "pytest-cov>=4.0"]
```

### torch-scatter Problem
torch-scatter requires PyTorch at build time (imports torch in setup.py).
pip doesn't guarantee install order, so `pip install -e .` fails on fresh env.

**Workaround:** Don't include torch-scatter in dependencies. Document manual install:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install -e .
```

---

## Notes
- Keep test fixtures minimal (≤3 molecules) for speed
- Use `scope="session"` for expensive fixtures (atom_keys, models)
- CPU-only for CI; GPU optional for local
- Integration tests should complete in <60 seconds total
