# AEV-PLIG Development Plan

## Status
Last updated: 2026-02-09
Current phase: HPC workflow implementation

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
- [x] Add missing torchani_mod dependencies (lark, requests)
- [x] SLURM HPC workflow (13 scripts, see details below)
- [x] Add `Config.MODEL_NAME` — single source of truth for default model
- [x] Add `.predict()` method to models — uniform inference interface
- [x] Add `--model` flag to `predict.py` — was hardcoded to GATv2Net
- [x] Fix `predict.py` to pass model class (not instance) to Predictor

## Up Next
1. Download data script — HIGH
2. Unit tests (Phase 2) — MEDIUM
3. Regression tests (Phase 3) — LOW

## Backburner
- Dependency consolidation (optional extras for dev/test)

---

## SLURM HPC Workflow

Priority: URGENT
Status: Implemented
Last Updated: 2026-02-09

### Purpose
Create a complete SLURM workflow for running the AEV-PLIG pipeline on the HTC cluster at the user's HPC facility. This workflow automates:
1. Graph generation from protein-ligand complexes (3 datasets in parallel)
2. PyTorch data creation (train/valid/test splits)
3. Bayesian model training (ensemble of 5 models)
4. Prediction with uncertainty quantification

### Target Cluster: HTC
- Cluster name: `htc`
- All jobs must use `--cluster=htc`
- Available partitions:
  - **short**: 12h max (for most jobs)
  - **medium**: 2 days max
  - **long**: unlimited (for training)
  - **devel**: 10 min max (TESTS ONLY)
  - **interactive**: 4h max (for interactive testing)

### Folder Structure

```
slurm/
├── config.sh                       # Environment setup + cluster settings (merged)
│
├── jobs/                           # Production SLURM batch jobs
│   ├── 01_generate_graphs.sh      # Generate graphs (8h, 32GB, 4 CPUs)
│   ├── 02_create_data.sh          # Create PyTorch data (2h, 20GB, 8 CPUs)
│   ├── 03_train.sh                # Train model (24h, 20GB, 8 CPUs, GPU)
│   └── 04_predict.sh              # Predict (4h, 20GB, 8 CPUs, GPU)
│
├── tests/                          # Testing infrastructure
│   ├── jobs/                       # Quick test SLURM jobs
│   │   ├── 01_generate_graphs_quick.sh   # Test graph gen (30min)
│   │   └── 03_train_quick.sh             # Test training (2h, 5 epochs)
│   ├── test_slurm.sh               # Test SLURM submission (devel partition)
│   ├── test_local.sh               # Test in interactive session
│   └── test_environment.sh         # Validate setup
│
├── submit_training.sh              # Submit: graphs → data → train
├── submit_prediction.sh            # Submit: predict only
└── submit_slurm.sh                 # Submit: full pipeline
```

**Design decisions:**
- `env.sh` and `config.sh` merged into single `config.sh` — every job sources one file
- No hyperparameter duplication — Python defaults in `aev_plig/config.py` are the single source of truth. `config.sh` only has SLURM settings and runtime choices (model name, dataset name)
- Prediction (job 04) submitted separately because trained model name includes a timestamp

### Resource Specifications

#### Production Jobs

| Job | Partition | Time | Memory | CPUs | GPU | Notes |
|-----|-----------|------|--------|------|-----|-------|
| 01_generate_graphs.sh | short | 08:00:00 | 32GB | 4 | - | 3 parallel processes |
| 02_create_data.sh | short | 02:00:00 | 20GB | 8 | - | Single process |
| 03_train.sh | long | 24:00:00 | 20GB | 8 | gpu:1 | Ensemble of 5 models |
| 04_predict.sh | short | 04:00:00 | 20GB | 8 | gpu:1 | Predictions + uncertainty |

#### Test Jobs (devel partition only)

All test submissions use `--partition=devel` with 10-minute time limit and reduced resources.

### Pipeline Flow

```
01_generate_graphs.sh (afterok)
    ↓
02_create_data.sh (afterok)
    ↓
03_train.sh (afterok)
    ↓
04_predict.sh
```

**Job Dependencies:**
- Use `--dependency=afterok:$JOB_ID` for sequential execution
- Automatic failure propagation (jobs cancelled if dependency fails)

### Configuration (slurm/config.sh)

**Merged environment setup + cluster settings. All jobs source this file:**
```bash
source "$(dirname "$0")/../config.sh"
```

**Modules loaded:** `Anaconda3`, `Boost/1.77.0-GCC-11.2.0`, `CUDA`

**Conda environment:** `$DATA/envs/aev-plig` (bin added to PATH)

**SLURM settings:**
- Cluster name: `CLUSTER_NAME="htc"`
- Partition names: `PARTITION_SHORT`, `PARTITION_LONG`, etc.
- Memory presets: `MEM_STANDARD=20GB`, `MEM_LARGE=32GB`

**Runtime choices (not hyperparameters):**
- `MODEL_NAME="GATv2NetBayesian"`
- `DATASET_NAME="pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"`

Hyperparameters (lr, epochs, batch_size, etc.) are NOT duplicated here — they use
Python defaults from `aev_plig/config.py` as the single source of truth.

### Memory Management

**Graph Generation (01):**
- Runs 3 Python processes in parallel (`&` and `wait`)
- Each process has isolated memory (automatic cleanup between processes)
- No explicit memory clearing needed

**Data Creation (02):**
- Single Python process
- Loads all pickle files at once (pdbbind, bindingnet, bindingdb)
- Memory clears automatically when process exits

**Training/Prediction (03, 04):**
- Single Python process per job
- GPU memory: 20GB (sufficient for batch_size=128)

### Testing Strategy

**Three-tier testing:**

1. **Environment Validation** (`test_environment.sh`)
   - Check conda environment exists
   - Verify Python packages (torch, torch_geometric, torch_scatter, aev_plig)
   - Check CUDA availability
   - Verify data files and scripts exist
   - Run before submitting any jobs

2. **Local Interactive Testing** (`test_local.sh`)
   - Run in interactive session: `srun --cluster=htc --partition=interactive --mem=20GB --cpus-per-task=4 --gres=gpu:1 --time=04:00:00 --pty bash`
   - Test imports, data loading, model creation
   - Fast iteration without SLURM queue

3. **SLURM Submission Testing** (`test_slurm.sh`)
   - Submit all 4 jobs to devel partition (10 min limit)
   - Tests job submission and dependency chain
   - Jobs will likely timeout - this is expected
   - Verifies SLURM syntax and cluster configuration

### Submission Scripts

**submit_training.sh** (graphs → data → train):
```bash
J1=$(sbatch --cluster=htc --parsable slurm/jobs/01_generate_graphs.sh)
J2=$(sbatch --cluster=htc --parsable --dependency=afterok:$J1 slurm/jobs/02_create_data.sh)
J3=$(sbatch --cluster=htc --parsable --dependency=afterok:$J2 slurm/jobs/03_train.sh)
```

**submit_prediction.sh** (predict only):
```bash
sbatch --cluster=htc slurm/jobs/04_predict.sh
```

**submit_slurm.sh** (full pipeline):
```bash
# Chains all 4 jobs with dependencies
```

### Quick Test Jobs

For rapid testing without full pipeline:

- **01_generate_graphs_quick.sh**: Generate only PDBbind graphs (smallest dataset), 30 min
- **03_train_quick.sh**: Train for 5 epochs only (vs 200), 2h

**Use case:** Verify environment setup, test code changes, check GPU allocation

### Implementation Files (13 total)

**Configuration (1):**
- `slurm/config.sh` (merged environment + cluster settings)

**Production Jobs (4):**
- `slurm/jobs/01_generate_graphs.sh`
- `slurm/jobs/02_create_data.sh`
- `slurm/jobs/03_train.sh`
- `slurm/jobs/04_predict.sh`

**Submission Scripts (3):**
- `slurm/submit_training.sh`
- `slurm/submit_prediction.sh`
- `slurm/submit_slurm.sh`

**Test Scripts (3):**
- `slurm/tests/test_environment.sh`
- `slurm/tests/test_local.sh`
- `slurm/tests/test_slurm.sh`

**Quick Test Jobs (2):**
- `slurm/tests/jobs/01_generate_graphs_quick.sh`
- `slurm/tests/jobs/03_train_quick.sh`

### Key Design Decisions

1. **Single config.sh** (NOT separate env.sh + config.sh, NOT YAML):
   - Merged environment setup and cluster settings into one file
   - No hyperparameter duplication — Python config.py is the single source of truth
   - No external dependencies (PyYAML, etc.)
   - YAML config documented for future enhancement

2. **Structured folders** (NOT flat):
   - More organized than most scientific repos
   - Clear separation: jobs/ vs tests/
   - Easy to navigate and maintain
   - Balances organization with HPC conventions

3. **Memory consistency**:
   - All jobs use 20GB except graph generation (32GB for 3 parallel processes)
   - Simplifies resource allocation
   - Matches GPU memory requirements

4. **Partition selection**:
   - short: Most jobs fit in 12h limit
   - long: Only training (24h required for 200 epochs)
   - devel: Only for test_slurm.sh submissions

5. **Test philosophy**:
   - Validate environment first
   - Test locally in interactive session
   - Test SLURM submission on devel
   - Quick test jobs for rapid iteration
   - Never waste production resources on testing

### Future Enhancements (Documented for Later)

**Config-driven workflow (YAML):**
- `slurm/config.yaml` for all parameters
- `slurm/load_config.py` to parse and export env vars
- Better for experiment tracking and reproducibility
- Standard in scientific ML workflows
- Deferred until after initial implementation

**Workflow managers:**
- Snakemake integration for complex dependencies
- Nextflow for pipeline orchestration
- Too heavy for current needs (4-job linear pipeline)

### Usage Examples

**Standard workflow:**
```bash
# 1. Validate environment
./slurm/tests/validate_environment.sh

# 2. Submit training pipeline
./slurm/submit_training.sh

# 3. Monitor
squeue -u $USER --cluster=htc

# 4. View logs
tail -f logs/train_*.out
```

**Testing workflow:**
```bash
# Interactive testing
srun --cluster=htc --partition=interactive --mem=20GB --cpus-per-task=4 --gres=gpu:1 --time=04:00:00 --pty bash
./slurm/tests/test_local.sh

# SLURM submission testing
./slurm/tests/test_slurm.sh
```

**Quick test:**
```bash
# Test graph generation only
sbatch --cluster=htc slurm/tests/jobs/01_generate_graphs_quick.sh

# Test training with 5 epochs
sbatch --cluster=htc slurm/tests/jobs/03_train_quick.sh
```

### References

**Best Practices:**
- [SLURM Workflows Best Practices - ARCC](https://arccwiki.atlassian.net/wiki/spaces/DOCUMENTAT/pages/2231795764/Slurm+Workflows+and+Best+Practices)
- [NASA NCCS SLURM Best Practices](https://www.nccs.nasa.gov/nccs-users/instructional/using-slurm/best-practices)
- [BIH HPC Project Structure](https://hpc-docs.cubi.bihealth.org/best-practice/project-structure/)
- [HBC Training: Data Organization](https://hbctraining.github.io/Intro-to-rnaseq-hpc-salmon/lessons/01_data_organization.html)

**Example Repositories:**
- [CSCfi/machine-learning-scripts](https://github.com/CSCfi/machine-learning-scripts/tree/master/slurm) - ML on SLURM
- [y0ast/slurm-for-ml](https://github.com/y0ast/slurm-for-ml) - Hyperparameter search workflow
- [PyTorch SLURM Examples](https://github.com/pytorch/examples/tree/main/distributed/ddp-tutorial-series/slurm) - Distributed training

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
    "lark>=1.1.0",
    "requests>=2.28.0",
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

### Missing Dependencies Fix (torchani_mod)

**Problem:** The vendored `torchani_mod` package has dependencies not declared in original setup.py.

**Symptoms:**
```
ModuleNotFoundError: No module named 'lark'
```

**Root Cause:**
- `torchani_mod/__init__.py` imports `neurochem` module unconditionally
- `torchani_mod/neurochem/__init__.py` requires `lark` and `requests`
- These were not in the original dependency list

**Solution:** Added missing dependencies:
- `lark>=1.1.0` - for neurochem resource parser
- `requests>=2.28.0` - for neurochem resource downloading

**Dependencies Added:**
```toml
dependencies = [
    # ... existing dependencies ...
    "lark>=1.1.0",       # NEW - torchani_mod.neurochem parser
    "requests>=2.28.0",  # NEW - torchani_mod.neurochem resources
]
```

### HPC Installation Workflow

**Recommended approach for HPC clusters:**

1. **Load required modules:**
   ```bash
   module load cuda/12.1    # Adjust version for your cluster
   module load gcc/11       # May be needed for compilation
   ```

2. **Activate conda environment:**
   ```bash
   conda activate aev-plig
   ```

3. **Install with --no-build-isolation** (faster on HPC):
   ```bash
   pip install -e . --no-build-isolation
   ```

   **Why --no-build-isolation?**
   - Reuses environment's torch installation
   - Avoids downloading/rebuilding torch in isolated environment
   - Faster installation on HPC with pre-installed dependencies
   - Works well when torch is already installed from conda/pip

4. **Verify installation:**
   ```bash
   python -c "import aev_plig; import torchani_mod"
   python scripts/generate_pdbbind_graphs.py --help
   ```

**Alternative (standard method):**
```bash
pip install -e .   # Uses build isolation, slower but more reproducible
```

**Important:** Keep CUDA module loaded for all sessions where you run GPU-enabled code.

---

## Notes
- Keep test fixtures minimal (≤3 molecules) for speed
- Use `scope="session"` for expensive fixtures (atom_keys, models)
- CPU-only for CI; GPU optional for local
- Integration tests should complete in <60 seconds total
