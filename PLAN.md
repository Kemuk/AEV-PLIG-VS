# AEV-PLIG Development Plan

## Status
Last updated: 2026-02-02
Current phase: Integration testing

## Completed
- [x] Refactor codebase into modular package (v2.0)
- [x] Centralize configuration in `aev_plig/config.py`
- [x] Remove code duplication (~500 lines eliminated)
- [x] Create model registry for extensibility

## In Progress
- [ ] Integration test suite (8 tests)

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

Priority: MEDIUM
Depends on: Integration tests complete

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

### Tests Required for Bayesian
| Test | Purpose |
|------|---------|
| test_bayesian_output_shape | Returns (mean, var) tuple |
| test_variance_positivity | var > 0 always |
| test_uncertainty_calibration | ~68% within ±1σ on held-out data |
| test_ood_uncertainty | Higher variance for dissimilar molecules |

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

## Notes
- Keep test fixtures minimal (≤3 molecules) for speed
- Use `scope="session"` for expensive fixtures (atom_keys, models)
- CPU-only for CI; GPU optional for local
- Integration tests should complete in <60 seconds total
