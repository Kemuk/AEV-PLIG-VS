# AEV-PLIG Development Plan

## Status
Last updated: 2026-02-05
Current phase: Modernizing package configuration

## Completed
- [x] Refactor codebase into modular package (v2.0)
- [x] Centralize configuration in `aev_plig/config.py`
- [x] Remove code duplication (~500 lines eliminated)
- [x] Create model registry for extensibility
- [x] Integration test suite (7 test files)
- [x] GitHub Actions CI workflow
- [x] Bayesian last layer (GATv2NetBayesian) + minimal tests
- [x] Bayesian training support (auto-detect in Trainer)
- [x] Migrate setup.py → pyproject.toml (torch-scatter build fix)

## Up Next
1. Download data script — HIGH
2. Unit tests (Phase 2) — MEDIUM
3. Regression tests (Phase 3) — LOW

## Backburner
- Dependency consolidation (optional extras for dev/test)

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

## Planned: Migrate to pyproject.toml

Priority: HIGH
Status: COMPLETED
Last Updated: 2026-02-05

### Why Migrate
- `setup.py` is deprecated (PEP 517/518)
- `pyproject.toml` is declarative, modern, and cleaner
- Better build isolation
- Consolidates configuration (pytest can move here too)

### Migration Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Build backend | setuptools | Most compatible, currently in use |
| License | BSD-3-Clause | Match LICENSE.txt (setup.py incorrectly said MIT) |
| Dependencies | All in pyproject.toml | No requirements.txt needed |
| pytest.ini | Migrate to pyproject.toml | Consolidate configuration |
| setup.py fate | Remove completely | Clean break, no shims |

### Complete pyproject.toml Structure

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "torch>=2.0.0"]
build-backend = "setuptools.build_meta"

[project]
name = "aev-plig"
version = "2.0.0"
description = "Graph Neural Network-based Scoring Function for Protein-Ligand Binding Affinity Prediction"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "BSD-3-Clause"}
authors = [
    {name = "AEV-PLIG Development Team"}
]
keywords = ["bioinformatics", "chemistry", "machine-learning", "drug-discovery", "protein-ligand"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Bio-Informatics",
    "Topic :: Scientific/Engineering :: Chemistry",
    "License :: OSI Approved :: BSD License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
]

dependencies = [
    "torch>=2.0.0",
    "torch-geometric>=2.3.0",
    "torch-scatter>=2.1.0",
    "rdkit>=2023.0.0",
    "torchani>=2.2.0",
    "biopandas>=0.4.0",
    "qcelemental>=0.25.0",
    "scikit-learn>=1.0.0",
    "pandas>=1.5.0",
    "numpy>=1.23.0",
    "scipy>=1.9.0",
    "tqdm>=4.65.0",
]

[project.urls]
Homepage = "https://github.com/Jnelen/AEV-PLIG"
Repository = "https://github.com/Jnelen/AEV-PLIG"
Documentation = "https://github.com/Jnelen/AEV-PLIG"

[project.scripts]
aev-plig-train = "scripts.train:main"
aev-plig-predict = "scripts.predict:main"
aev-plig-generate-graphs = "scripts.generate_pdbbind_graphs:main"

[tool.setuptools]
zip-safe = false

[tool.setuptools.packages.find]
include = ["aev_plig*", "torchani_mod*"]
exclude = ["scripts", "tests*", "data", "output"]

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_functions = ["test_*"]
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
    "integration: marks integration tests",
    "regression: marks regression tests",
]
filterwarnings = [
    "ignore::FutureWarning",
    "ignore::DeprecationWarning",
]
```

### Implementation Steps

#### Step 1: Create pyproject.toml
- Create new file with complete configuration above
- Verify all 12 dependencies from setup.py fallback list are included
- Include both `aev_plig*` and `torchani_mod*` in package discovery
- Migrate complete pytest.ini configuration

#### Step 2: Testing Phase
Before removing old files, verify:
1. **Installation tests:**
   ```bash
   pip install -e .          # Editable install
   pip install .             # Normal install
   python -m build           # Build wheel and sdist
   ```

2. **Package import tests:**
   ```bash
   python -c "import aev_plig"
   python -c "import torchani_mod"
   ```

3. **Console scripts tests:**
   ```bash
   aev-plig-train --help
   aev-plig-predict --help
   aev-plig-generate-graphs --help
   ```

4. **Pytest integration:**
   ```bash
   pytest                    # Should find tests
   pytest -v                 # Verbose mode
   pytest -m "not slow"      # Marker filtering works
   ```

#### Step 3: Remove Legacy Files
Once testing passes:
- Delete `setup.py`
- Delete `pytest.ini`

#### Step 4: Documentation Updates
Update README.md:
- Note modern pyproject.toml-based installation
- Remove any setup.py references
- Update installation section if needed

### Key Changes from setup.py

| Item | setup.py | pyproject.toml |
|------|----------|----------------|
| License classifier | MIT License (WRONG) | BSD License (CORRECT) |
| Config location | setup.py + pytest.ini | Single pyproject.toml |
| Dependencies source | requirements.txt fallback | Declarative in [project] |
| Package discovery | find_packages() | Explicit include/exclude |
| Version | Hardcoded | Hardcoded (same) |

### Open Questions for Future

1. **Version management:** Consider dynamic versioning from `__version__`?
2. **Optional extras:** Add `[project.optional-dependencies]` for dev/test tools?
3. **torch-scatter compatibility:** May need manual install in some environments (known issue)

### torch-scatter Installation Solution

**Problem:** torch-scatter requires PyTorch available at build time, but pip doesn't guarantee dependency installation order.

**Solution:** Add torch to `[build-system].requires` to ensure it's installed before torch-scatter builds.

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel", "torch>=2.0.0"]
```

**How it works:**
1. PEP 517/518 mandates build-system requirements are installed first
2. torch is available in the isolated build environment
3. torch-scatter can build successfully
4. Single command installation: `pip install -e .`

**Trade-off:** torch is installed in both the build environment and user environment, but pip caches wheels so the second install is fast.

**Installation:**
```bash
# Single command - handles everything automatically
pip install -e .
```

For CUDA-specific PyTorch installations, install torch first:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -e .
```

---

## Notes
- Keep test fixtures minimal (≤3 molecules) for speed
- Use `scope="session"` for expensive fixtures (atom_keys, models)
- CPU-only for CI; GPU optional for local
- Integration tests should complete in <60 seconds total
