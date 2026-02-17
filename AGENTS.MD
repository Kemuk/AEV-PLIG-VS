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
