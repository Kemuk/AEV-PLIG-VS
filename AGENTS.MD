# AGENTS.MD - Future Development Tasks for AI Agents

This document contains planned features and improvements for the AEV-PLIG project. It is specifically formatted to help AI agents understand the codebase and implement future enhancements.

---

## Priority Features

### 🔥 HIGH PRIORITY: Store Metadata in PyTorch Data Objects

**Status**: Not yet implemented
**Difficulty**: Medium
**Estimated Impact**: High - enables consistent output format across all prediction modes

#### Problem Statement

Currently, when using processed data mode (loading from `.pt` files), the prediction output lacks original metadata like `unique_id`, `pK`, `sdf_file`, and `pdb_file`. This makes it difficult to trace predictions back to source structures.

**Current behavior:**
- CSV mode output: `unique_id | pK | sdf_file | pdb_file | pred_ensemble | pred_0 | ... | pred_9`
- Processed mode output: `graph_id | pred_ensemble | pred_0 | ... | pred_9` ⚠️ Missing metadata!

#### Proposed Solution

Store metadata as custom attributes directly in PyTorch Geometric `Data` objects. This makes data files self-contained and enables consistent output format regardless of prediction mode.

#### Implementation Plan

**Files to modify:**

1. **`aev_plig/datasets.py`** - Modify `create_dataset()` function
   - Add optional metadata parameters: `unique_ids=None, original_pK=None, sdf_files=None, pdb_files=None`
   - Store metadata as custom attributes on each Data object:
     ```python
     def create_dataset(graph_ids, mol_graphs, scaler=None, unique_ids=None,
                       original_pK=None, sdf_files=None, pdb_files=None):
         data_list = []
         for i, graph_id in enumerate(graph_ids):
             data = Data(...)  # existing code

             # Add metadata attributes
             if unique_ids is not None:
                 data.unique_id = unique_ids[i]
             if original_pK is not None:
                 data.original_pK = original_pK[i]
             if sdf_files is not None:
                 data.sdf_file = sdf_files[i]
             if pdb_files is not None:
                 data.pdb_file = pdb_files[i]

             data_list.append(data)
         return data_list
     ```

2. **`create_pytorch_data.py`** - Pass metadata to `create_dataset()`
   - Extract metadata from CSV: `unique_ids`, `pK` (unscaled), `sdf_file`, `pdb_file`
   - Pass to `create_dataset()` when creating train/valid/test datasets
   - Example:
     ```python
     train_data, scaler = create_dataset(
         train_graph_ids,
         mol_graphs,
         scaler=scaler,
         unique_ids=train_df['unique_id'].tolist(),
         original_pK=train_df['pK'].tolist(),  # unscaled
         sdf_files=train_df['sdf_file'].tolist(),
         pdb_files=train_df['pdb_file'].tolist()
     )
     ```

3. **`scripts/predict.py`** - Extract metadata from Data objects
   - In `load_processed_data()`: Extract metadata if available
   - In output preparation: Include metadata columns if present
   - Example:
     ```python
     # After loading processed data
     if hasattr(all_data[0], 'unique_id'):
         df['unique_id'] = [d.unique_id for d in all_data]
         df['pK'] = [d.original_pK for d in all_data]
         df['sdf_file'] = [d.sdf_file for d in all_data]
         df['pdb_file'] = [d.pdb_file for d in all_data]
     ```

#### Benefits

✅ **Consistent output format** - Both CSV and processed modes produce identical output columns
✅ **Self-contained data files** - No need for separate metadata files
✅ **Better traceability** - Easy to map predictions back to source structures
✅ **Simplifies workflow** - No manual joining required
✅ **Backward compatible** - Old .pt files without metadata still work (just missing columns)

#### Testing Plan

1. Regenerate processed datasets with new metadata
2. Run predictions in processed mode
3. Verify output contains all metadata columns
4. Test with old .pt files to ensure backward compatibility

#### Related Files

- `aev_plig/datasets.py:149` - `create_dataset()` function
- `create_pytorch_data.py:170` - Dataset creation calls
- `scripts/predict.py:212` - `load_processed_data()` function
- `scripts/predict.py:127` - `validate_and_process_csv()` function

---

## Future Enhancements

### UMAP Embedding Visualisation in WandB

**Status**: Planned, not yet implemented
**Difficulty**: Medium
**Priority**: Medium — shelved for a dedicated session

#### Goal

After `aev-plig-predict --wandb`, log a UMAP projection of the learned graph
embeddings (the pooled GNN representation before the FC head) to the WandB
evaluation run.  Three colourings are needed:

| Plot | Colour axis | Scientific question |
|------|-------------|---------------------|
| `eval_umap_pK` | True pK value | Does embedding space encode binding affinity? |
| `eval_umap_error` | `\|ε\|` = `\|true − pred\|` | Which regions of chemical space fail? |
| `eval_umap_uncertainty` | Predicted σ (Bayesian only) | Where is the model uncertain? |

Subsample to ≤ 2 000 random points so UMAP stays fast regardless of dataset
size.  Degrade silently if `umap-learn` is not installed (same pattern as
`uncertainty_toolbox`).

#### Open Decision

`run_predictions()` in `aev_plig/prediction.py` currently returns the
predictions array and doesn't expose the loaded model.  Two options:

- **Option A (preferred)** — add a `load_model(args, config)` helper to
  `aev_plig/prediction.py`.  `predict.py` calls it, keeps a reference to the
  first model, and passes it to both `run_predictions()` and the new
  `extract_embeddings()`.  Clean composable interface.

- **Option B** — add `return_model=True` kwarg to `run_predictions()`.
  Simpler but muddies the return type.

#### Implementation Steps

1. **`aev_plig/prediction.py`** — add two helpers:

   ```python
   def load_model(args, model_cfg: dict, device):
       """Load a single trained model from model_cfg and args."""
       from aev_plig.models import MODEL_REGISTRY
       model_class = MODEL_REGISTRY[model_cfg["model"]]
       model = model_class(node_feature_dim=..., edge_feature_dim=...,
                           config=args)
       ckpt_path = Path(Config.TRAINED_MODELS_DIR) / args.trained_model_name \
                   / f"model_seed_{first_seed}.model"
       model.load_state_dict(torch.load(ckpt_path, map_location=device))
       model.eval()
       return model

   def extract_embeddings(model, loader, device, max_points: int = 2000) -> np.ndarray:
       """Return (N, 2*hidden_dim) graph-level embeddings via a forward hook on lin1."""
       import torch
       captured = []
       def _hook(module, inp, out):
           captured.append(inp[0].detach().cpu())
       handle = model.lin1.register_forward_hook(_hook)
       with torch.no_grad():
           for batch in loader:
               batch = batch.to(device)
               model(batch)
       handle.remove()
       emb = torch.cat(captured, dim=0).numpy()
       if len(emb) > max_points:
           idx = np.random.choice(len(emb), max_points, replace=False)
           emb = emb[idx]
       return emb
   ```

   Note: the attribute `model.lin1` is the first FC layer in `BaseGATv2Net`.
   Verify the exact attribute name in `aev_plig/models.py` before implementing.

2. **`scripts/predict.py`** — after `save_results()`, inside `if args.wandb:`:

   ```python
   from aev_plig.prediction import load_model, extract_embeddings
   first_model = load_model(args, model_cfg, device=args.device)
   emb = extract_embeddings(first_model, test_loader, device=args.device)
   log_evaluation_to_wandb(run, df, model_cfg, embeddings=emb)
   ```

   `test_loader` needs to be retained from `run_predictions()` — either
   return it alongside the df, or re-construct it before this block.

3. **`aev_plig/results.py`** — extend `log_evaluation_to_wandb` signature:

   ```python
   def log_evaluation_to_wandb(run, df, model_cfg, truth_col="pK",
                               embeddings: np.ndarray | None = None) -> None:
   ```

   Add UMAP section at the end (inside the existing `try/except` plot block):

   ```python
   if embeddings is not None:
       try:
           import umap
           import wandb as _wandb
           reducer = umap.UMAP(n_components=2, random_state=42)
           coords = reducer.fit_transform(embeddings)
           n = len(coords)
           # align embeddings with df rows (may be subsampled)
           y_t = y_true[:n]; y_p = y_pred[:n]; err = abs_res[:n]
           for label, values, colorscale in [
               ("pK",    y_t,  "Viridis"),
               ("error", err,  "Reds"),
           ]:
               fig = go.Figure(go.Scatter(
                   x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                   mode="markers",
                   marker=dict(color=values.tolist(), colorscale=colorscale,
                               showscale=True, size=5, opacity=0.7),
                   text=[f"{label}={v:.2f}" for v in values],
               ))
               fig.update_layout(
                   xaxis_title="UMAP-1", yaxis_title="UMAP-2",
                   title=f"Embedding UMAP coloured by {label}",
               )
               run.log({f"eval_umap_{label}": _wandb.Plotly(fig)})
           # Uncertainty plot (Bayesian only)
           var_col = "preds_var"
           if var_col in pl_df.columns:
               sigma = np.sqrt(pl_df[var_col].to_numpy()[:n])
               fig_unc = go.Figure(go.Scatter(
                   x=coords[:, 0].tolist(), y=coords[:, 1].tolist(),
                   mode="markers",
                   marker=dict(color=sigma.tolist(), colorscale="Plasma",
                               showscale=True, size=5, opacity=0.7),
               ))
               fig_unc.update_layout(title="Embedding UMAP coloured by σ")
               run.log({"eval_umap_uncertainty": _wandb.Plotly(fig_unc)})
       except Exception:
           pass
   ```

4. **`pyproject.toml`** — add `"umap-learn>=0.5.0"` to the `[wandb]` optional
   dep group:

   ```toml
   wandb = ["wandb>=0.16.0", "uncertainty-toolbox>=0.1.0", "umap-learn>=0.5.0"]
   ```

#### New Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| `umap-learn` | `>=0.5.0` | UMAP dimensionality reduction |

#### Testing

```bash
# After training a quick model:
aev-plig-predict \
    --trained_model_name GATv2Net_TEST_... \
    --dataset pdbbind \
    --wandb --wandb_project aev-plig-dev

# Check WandB run for:
#   eval_umap_pK, eval_umap_error  (always)
#   eval_umap_uncertainty           (Bayesian models only)
```

#### Related Files

- `aev_plig/models.py` — check exact name of `lin1` (first FC layer)
- `aev_plig/prediction.py` — add `load_model()` and `extract_embeddings()`
- `aev_plig/results.py:208` — `log_evaluation_to_wandb()` signature + UMAP block
- `scripts/predict.py` — call `extract_embeddings()` before `log_evaluation_to_wandb()`
- `pyproject.toml` — add `umap-learn` to `[wandb]` optional dep



### Virtual Screening Support

**Status**: Not yet planned
**Difficulty**: High
**Priority**: Medium

Add capability to screen large compound libraries against protein targets.

**Key features needed:**
- Batch processing of large SDF files
- Progress tracking and checkpointing
- Multi-GPU support for parallel screening
- Output ranking by predicted affinity

**Files to create:**
- `aev_plig/screening.py` - Screening pipeline module
- `scripts/screen.py` - CLI tool for virtual screening

---

### Decoy Training Support

**Status**: Not yet planned
**Difficulty**: High
**Priority**: Medium

Enable training with ranking losses using protein-ligand decoys.

**Key features needed:**
- Pairwise ranking loss implementation
- Decoy dataset loader
- Modified training loop for ranking objectives
- Evaluation metrics for ranking quality

**Files to modify:**
- `aev_plig/training.py` - Add ranking loss functions
- `aev_plig/datasets.py` - Add decoy dataset class

---

### Model Registry Expansion

**Status**: Ongoing
**Difficulty**: Low-Medium
**Priority**: Low

Add more GNN architectures to the model registry.

**Potential models to add:**
- TransformerConv
- SuperGAT
- PNA (Principal Neighbourhood Aggregation)
- DimeNet++

**Files to modify:**
- `aev_plig/models.py` - Add new model classes and register them

---

### CLI Tool Enhancement

**Status**: Not yet planned
**Difficulty**: Low
**Priority**: Low

Add user-friendly command-line interface with subcommands.

**Proposed structure:**
```bash
aev-plig train --model GATv2Net --dataset pdbbind
aev-plig predict --input data.csv --model my_model
aev-plig screen --library compounds.sdf --target protein.pdb
```

**Files to create:**
- `aev_plig/cli.py` - CLI entry point
- Update `setup.py` - Add console_scripts entry points

---

## Development Guidelines for AI Agents

### Code Style

- Follow existing code conventions in the codebase
- Use type hints for function parameters and returns
- Add comprehensive docstrings (Google style)
- Keep functions focused and modular
- Avoid code duplication - use the package modules

### Testing Strategy

- Test both CSV and processed data modes
- Verify backward compatibility with existing files
- Check edge cases (empty data, missing files, etc.)
- Validate output format matches expectations

### Git Workflow

- Create feature branch: `claude/feature-name-{sessionId}`
- Write clear commit messages describing changes
- Push to remote branch for review
- Include session URL in commit messages

### Key Architecture Principles

1. **Centralized Configuration** - All parameters in `aev_plig/config.py`
2. **Modular Design** - Each module has single responsibility
3. **Zero Duplication** - Reuse package functions, never copy code
4. **Backward Compatibility** - Maintain support for existing workflows
5. **Clean Interfaces** - Functions have clear inputs/outputs

### Important File Locations

- **Configuration**: `aev_plig/config.py`
- **Models**: `aev_plig/models.py`
- **Training**: `aev_plig/training.py`
- **Prediction**: `aev_plig/prediction.py`
- **Datasets**: `aev_plig/datasets.py`
- **Scripts**: `scripts/` directory

---

## Notes

This document should be updated as features are implemented or new priorities emerge. When completing a task, move it to a "Completed Features" section with implementation date and session ID.

**Document Version**: 1.0
**Last Updated**: 2026-02-16
**Format**: Optimized for AI agent consumption
