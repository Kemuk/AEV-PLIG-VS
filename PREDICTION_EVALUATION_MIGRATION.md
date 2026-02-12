# Prediction & Evaluation Code Migration Guide

## 🎯 Overview

This document outlines the required changes to prediction and evaluation scripts due to the new seed-based model file organization.

---

## 📁 New Model Organization

### **Old Structure:**
```
output/trained_models/
  20250211-120000_model_GATv2Net_dataset_0.model
  20250211-120000_model_GATv2Net_dataset_1.model
  20250211-120000_model_GATv2Net_dataset_2.model
  ...
  20250211-120000_model_GATv2Net_dataset_9.model
  20250211-120000_model_GATv2Net_dataset.pickle  (scaler)
```

### **New Structure:**
```
models/
  GATv2Net_20250211_120000/
    model_seed_100.model
    model_seed_123.model
    model_seed_15.model
    model_seed_257.model
    model_seed_2.model
    model_seed_2012.model
    model_seed_3752.model
    model_seed_350.model
    model_seed_843.model
    model_seed_621.model
    scaler.pickle
```

---

## 🔄 Required Changes

### **1. Model Loading Function**

#### **Old Approach (Index-Based):**
```python
def load_ensemble_models(model_dir, model_name, dataset, timestamp, num_models=10):
    """Load ensemble models by index."""
    models = []
    for i in range(num_models):
        model_path = os.path.join(
            model_dir,
            f"{timestamp}_model_{model_name}_{dataset}_{i}.model"
        )
        checkpoint = torch.load(model_path)
        model = initialize_model(...)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)
    return models
```

#### **New Approach (Seed-Based):**
```python
import glob

def load_ensemble_models(model_run_dir):
    """
    Load all ensemble models from a training run directory.

    Args:
        model_run_dir: Path to model directory (e.g., "models/GATv2Net_20250211_120000")

    Returns:
        list: Loaded model objects, sorted by seed number
    """
    # Find all model files
    model_files = glob.glob(os.path.join(model_run_dir, "model_seed_*.model"))

    if not model_files:
        raise FileNotFoundError(f"No models found in {model_run_dir}")

    # Sort by seed number for consistency
    model_files.sort(key=lambda x: int(x.split('seed_')[1].split('.')[0]))

    models = []
    for model_path in model_files:
        checkpoint = torch.load(model_path)
        seed = checkpoint['seed']

        # Initialize model (get architecture from checkpoint metadata)
        model = initialize_model(
            model_name=checkpoint['model_name'],
            node_feature_dim=...,
            edge_feature_dim=...,
            config=...
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)

        print(f"Loaded: {os.path.basename(model_path)} (seed {seed})")

    return models
```

---

### **2. Scaler Loading Function**

#### **Old Approach:**
```python
def load_scaler(model_dir, model_name, dataset, timestamp):
    """Load scaler by constructing filename."""
    scaler_path = os.path.join(
        model_dir,
        f"{timestamp}_model_{model_name}_{dataset}.pickle"
    )
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)
```

#### **New Approach:**
```python
def load_scaler(model_run_dir):
    """
    Load scaler from model run directory.

    Args:
        model_run_dir: Path to model directory (e.g., "models/GATv2Net_20250211_120000")

    Returns:
        Scaler object
    """
    scaler_path = os.path.join(model_run_dir, "scaler.pickle")

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    with open(scaler_path, 'rb') as f:
        return pickle.load(f)
```

---

### **3. Ensemble Prediction Function**

#### **Old Approach:**
```python
def predict_ensemble(data_loader, model_dir, model_name, dataset, timestamp):
    """Predict using ensemble of models."""
    models = load_ensemble_models(model_dir, model_name, dataset, timestamp)
    scaler = load_scaler(model_dir, model_name, dataset, timestamp)

    # ... rest of prediction logic
```

#### **New Approach:**
```python
def predict_ensemble(data_loader, model_run_dir):
    """
    Predict using ensemble of models from a training run.

    Args:
        data_loader: PyTorch DataLoader
        model_run_dir: Path to model directory (e.g., "models/GATv2Net_20250211_120000")

    Returns:
        Ensemble predictions
    """
    models = load_ensemble_models(model_run_dir)
    scaler = load_scaler(model_run_dir)

    # Aggregate predictions from all models
    all_predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            preds = predict_single_model(model, data_loader, scaler)
            all_predictions.append(preds)

    # Average predictions
    ensemble_preds = np.mean(all_predictions, axis=0)
    return ensemble_preds
```

---

### **4. Command-Line Interface Updates**

#### **Old CLI:**
```python
parser.add_argument('--model_dir', default='output/trained_models')
parser.add_argument('--model_name', required=True)
parser.add_argument('--dataset', required=True)
parser.add_argument('--timestamp', required=True)
```

#### **New CLI:**
```python
parser.add_argument('--model_run_dir', required=True,
                   help='Path to model run directory (e.g., models/GATv2Net_20250211_120000)')

# OR: Auto-discover latest run
parser.add_argument('--model_family', default='GATv2Net',
                   help='Model family (finds latest run automatically)')
```

---

### **5. Helper Function: Find Latest Model Run**

```python
def find_latest_model_run(model_family, models_dir='models'):
    """
    Find the most recent training run for a model family.

    Args:
        model_family: Model name (e.g., "GATv2Net")
        models_dir: Base models directory

    Returns:
        Path to latest model run directory
    """
    pattern = os.path.join(models_dir, f"{model_family}_*")
    run_dirs = glob.glob(pattern)

    if not run_dirs:
        raise FileNotFoundError(f"No runs found for model family: {model_family}")

    # Sort by timestamp (embedded in directory name)
    run_dirs.sort(reverse=True)  # Latest first
    return run_dirs[0]

# Usage
latest_run = find_latest_model_run("GATv2Net")
models = load_ensemble_models(latest_run)
```

---

### **6. Validation Helper: Check Model Directory**

```python
def validate_model_run_dir(model_run_dir):
    """
    Validate that a model run directory contains all required files.

    Args:
        model_run_dir: Path to model directory

    Raises:
        FileNotFoundError: If directory or required files are missing
        ValueError: If no model files found
    """
    if not os.path.exists(model_run_dir):
        raise FileNotFoundError(f"Model directory not found: {model_run_dir}")

    if not os.path.isdir(model_run_dir):
        raise ValueError(f"Not a directory: {model_run_dir}")

    # Check for scaler
    scaler_path = os.path.join(model_run_dir, "scaler.pickle")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found in {model_run_dir}")

    # Check for model files
    model_files = glob.glob(os.path.join(model_run_dir, "model_seed_*.model"))
    if not model_files:
        raise ValueError(f"No model files found in {model_run_dir}")

    print(f"✓ Valid model run directory: {model_run_dir}")
    print(f"  - Found {len(model_files)} models")
    print(f"  - Found scaler")
    return True
```

---

## 📝 Files That Need Updating

### **Prediction Scripts:**
1. `scripts/predict.py` - Main prediction script
2. Any ensemble prediction utilities

### **Evaluation Scripts:**
3. Any evaluation/benchmarking scripts that load models
4. Cross-validation scripts
5. Hyperparameter tuning scripts

### **Notebook/Analysis:**
6. Jupyter notebooks that load models
7. Result analysis scripts
8. Figure generation scripts

---

## 🧪 Testing Strategy

### **1. Create Test Script**

```python
# tests/test_model_loading.py

def test_load_models_old_format():
    """Test loading models from old format (backwards compatibility)."""
    # ... test old format

def test_load_models_new_format():
    """Test loading models from new seed-based format."""
    model_dir = "models/GATv2Net_20250211_120000"
    models = load_ensemble_models(model_dir)
    assert len(models) == 10

def test_load_scaler_new_format():
    """Test loading scaler from new format."""
    model_dir = "models/GATv2Net_20250211_120000"
    scaler = load_scaler(model_dir)
    assert scaler is not None

def test_validate_model_directory():
    """Test model directory validation."""
    model_dir = "models/GATv2Net_20250211_120000"
    assert validate_model_run_dir(model_dir) is True
```

### **2. Migration Test**

```bash
# Test that predictions work with new format
python scripts/predict.py --model_run_dir models/GATv2Net_20250211_120000 \
                          --input_file data/test_ligands.csv \
                          --output_file predictions.csv
```

---

## ⚠️ Breaking Changes

### **What Breaks:**
1. **Hardcoded index-based model loading** - Must switch to seed-based
2. **Filename assumptions** - Scripts expecting `_0.model`, `_1.model`, etc.
3. **Scaler filenames** - Scripts looking for `{timestamp}_model_*.pickle`
4. **Directory structure** - Scripts assuming flat `output/trained_models/`

### **What Still Works:**
1. **Model checkpoint format** - Still uses `torch.save()` with same structure
2. **Scaler format** - Still uses `pickle.dump()`
3. **Model architecture** - No changes to model code
4. **Training logic** - Core training loop unchanged

---

## 🔄 Backwards Compatibility Strategy

### **Option 1: Support Both Formats**

```python
def load_ensemble_models_auto(path):
    """
    Auto-detect and load models from either old or new format.

    Args:
        path: Can be either:
              - New: "models/GATv2Net_20250211_120000" (directory)
              - Old: Specify base_dir and construct paths

    Returns:
        list: Loaded models
    """
    if os.path.isdir(path):
        # New format: path is the model run directory
        return load_ensemble_models_new_format(path)
    else:
        # Old format: construct from parameters
        raise ValueError("Old format no longer supported. Please use new directory structure.")
```

### **Option 2: Migration Script**

```python
# scripts/migrate_models.py

def migrate_old_models_to_new_format(old_dir, timestamp, model_name, dataset):
    """
    Migrate old model files to new directory structure.

    Args:
        old_dir: Old models directory (e.g., "output/trained_models")
        timestamp: Timestamp of the run
        model_name: Model name
        dataset: Dataset name
    """
    # Create new directory
    new_dir = f"models/{model_name}_{timestamp}"
    os.makedirs(new_dir, exist_ok=True)

    # Read Config.ENSEMBLE_SEEDS to map indices to seeds
    from aev_plig.config import Config
    seeds = Config.ENSEMBLE_SEEDS

    # Move model files
    for i, seed in enumerate(seeds):
        old_path = os.path.join(old_dir, f"{timestamp}_model_{model_name}_{dataset}_{i}.model")
        new_path = os.path.join(new_dir, f"model_seed_{seed}.model")

        if os.path.exists(old_path):
            shutil.copy2(old_path, new_path)
            print(f"Migrated: {old_path} → {new_path}")

    # Move scaler
    old_scaler = os.path.join(old_dir, f"{timestamp}_model_{model_name}_{dataset}.pickle")
    new_scaler = os.path.join(new_dir, "scaler.pickle")

    if os.path.exists(old_scaler):
        shutil.copy2(old_scaler, new_scaler)
        print(f"Migrated scaler: {old_scaler} → {new_scaler}")

    print(f"\n✓ Migration complete: {new_dir}")
```

---

## 📚 Example Usage

### **Old Code:**
```python
# Predict using ensemble
models = load_ensemble_models(
    model_dir="output/trained_models",
    model_name="GATv2Net",
    dataset="pdbbind_U_bindingnet_ligsim90",
    timestamp="20250211-120000"
)
scaler = load_scaler(
    model_dir="output/trained_models",
    model_name="GATv2Net",
    dataset="pdbbind_U_bindingnet_ligsim90",
    timestamp="20250211-120000"
)
```

### **New Code:**
```python
# Predict using ensemble (much simpler!)
model_run_dir = "models/GATv2Net_20250211_120000"

models = load_ensemble_models(model_run_dir)
scaler = load_scaler(model_run_dir)

# Or: Auto-find latest run
model_run_dir = find_latest_model_run("GATv2Net")
models = load_ensemble_models(model_run_dir)
```

---

## ✅ Summary Checklist

- [ ] Update `scripts/predict.py` to use new loading functions
- [ ] Update evaluation scripts
- [ ] Test model loading with new directory structure
- [ ] Update any Jupyter notebooks
- [ ] Document new CLI arguments
- [ ] Create migration script (optional, for old models)
- [ ] Update README with new usage examples
- [ ] Update any documentation referencing model paths

---

## 📞 Questions?

If you encounter issues during migration:
1. Check that model directory exists and contains `model_seed_*.model` files
2. Verify scaler.pickle exists in the directory
3. Use `validate_model_run_dir()` to check directory structure
4. Ensure models were trained with new training script (seed metadata in checkpoints)

---

**Last Updated:** 2026-02-11
**Status:** Ready for implementation
