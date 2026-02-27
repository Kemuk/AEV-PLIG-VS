# AEV-PLIG-VS — Project Context

## Package layout

```
aev_plig/         Core library
  config.py       Single source of truth for paths (PREDICTIONS_DIR, TRAINED_MODELS_DIR, MODEL_NAME)
  prediction.py   run_predictions() + save_results() — entry point for inference
  results.py      load_predictions(), load_all_predictions(), per_target_metrics(), …
  training.py     Training loop, loss functions, concordance_index
  models/         GATv2Net, GATv2NetBayesian, GATv2NetMixedPrecision, …

scripts/
  train.py        CLI: aev-plig-train
  predict.py      CLI: aev-plig-predict
  generate_pdbbind_graphs.py

notebooks/
  results.py      Marimo reactive notebook for post-prediction analysis
  AEV-PLIG_figures.ipynb  Manuscript figures (matplotlib, Jupyter)

tests/            pytest suite
```

## Key conventions

- **Model manifest**: every trained model directory contains `config.json` with at least
  a `"model"` key (e.g. `"GATv2Net"`). This is the canonical source for the architecture
  name; code should read it rather than parsing directory names.

- **Save path**: `output/predictions/{model_name}/{trained_model_name}/{data_name}_predictions.parquet`
  (`model_name` comes from `config.json["model"]`; `trained_model_name` is the directory name
  under `output/trained_models/`).

- **load_all_predictions** signature:
  ```python
  load_all_predictions(
      trained_model_names: list[str],
      data_name: str,
      predictions_dir: Path | None = None,
      trained_models_dir: Path | None = None,
  ) -> pl.DataFrame
  ```
  Returns a Polars DataFrame with columns `model_name` and `trained_model_name` added.

- **Polars throughout**: all internal DataFrames are Polars. Pandas is only used where
  third-party code requires it (e.g. legacy graph-generation scripts).

- **Config.py is authoritative**: default paths and hyperparameters live in `Config`.
  Do not hard-code paths elsewhere.

## Git preferences

- Commit messages: imperative mood, ≤72 chars subject line.
- Do not push to `master` directly; work on feature branches.
- Do not include co-author attribution lines in commits.
