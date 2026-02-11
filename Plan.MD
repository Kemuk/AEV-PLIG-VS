# AEV-PLIG Development Plan

## 🚀 Current Priority: Parallel Ensemble Training with Seed-Based Filenames

**Status:** Implementation in progress

**Goal:** Replace sequential ensemble training with parallel GPU jobs for 10× speedup using robust seed-based filenames.

**See:** [MINIMAL_PARALLEL_PLAN.md](MINIMAL_PARALLEL_PLAN.md) for original implementation plan

**Quick Summary:**
- Current: 1 job trains 10 models sequentially (~40 hours)
- Target: 10 jobs train 10 models in parallel (~4 hours with 10 GPUs)
- **Key Improvement:** Use seed in filename (`model_seed_100.model`) instead of index
- **Organization:** Models grouped by family: `models/{model}_{timestamp}/model_seed_100.model`
- **Validation:** Enforce no duplicate seeds in config
- Changes: 4 files for training, updated tests, prediction/evaluation documentation
- Test: `sbatch slurm/tests/jobs/03_train_quick.sh` (1 seed, 2 epochs, <10 min)
- Production: `./slurm/submit_training.sh` (all seeds from config)

### Architecture Changes

**New Directory Structure:**
```
models/
  GATv2Net_20250211_120000/
    model_seed_100.model
    model_seed_123.model
    model_seed_15.model
    ...
```

**Why Seed-Based Filenames?**
- ✅ No index confusion or ordering bugs
- ✅ Clear which seed → which model
- ✅ Works with any custom seeds
- ✅ Parallel-safe (seeds are unique)
- ✅ Validates no duplicate seeds at startup

---

## 🎯 Next Priority: Results Processing & Visualization System

**Status:** Planning phase (to be implemented after pipeline runs successfully)

**Goal:** Create a comprehensive results analysis system that reproduces the paper's figures and metrics using modern data tools (Polars + Plotly).

---

## Architecture Overview

### **Data Flow**
```
Predictions (CSV/Parquet)
    ↓
Polars DataFrame (load, filter, transform)
    ↓
Metrics Calculation (Polars groupby → Numpy → Scipy)
    ↓
Results Aggregation (Polars DataFrame)
    ↓
Visualization (Plotly native support - NO conversion!)
    ↓
Publication-quality figures (PNG/PDF)
```

### **Key Design Decisions**

1. **Polars for Data Manipulation** ✅
   - 5-10x faster than Pandas for large datasets
   - Better memory efficiency
   - Explicit, type-safe operations
   - Integrates with parquet metadata system

2. **Plotly for Visualization** ✅
   - Native Polars support via Narwhals (Plotly 5.15.0+)
   - No DataFrame conversion needed
   - Interactive plots (hover, zoom, pan)
   - Publication-quality static exports
   - Supports all plot types from the paper

3. **Scipy for Metrics** ✅
   - Extract `.to_numpy()` from Polars for scipy functions
   - Pearson R, Kendall's Tau, RMSE, bootstrap CI

---

## Module Structure

### **`src/results/` Package**

```
src/results/
├── __init__.py           # Package exports
├── utils.py              # Data loading, filtering, transforms (Polars)
├── metrics.py            # Statistical metrics (Polars + Scipy)
├── plotting.py           # Visualization functions (Plotly)
└── config.py             # Plot styles, constants, color schemes
```

---

## Detailed Module Specifications

### **1. `src/results/utils.py` - Data Utilities (Polars-First)**

#### Functions:

**`load_predictions(filepath: str, format: str = 'csv') -> pl.DataFrame`**
- Load prediction files (CSV or Parquet)
- Validate required columns
- Return Polars DataFrame

**`pk_to_dg_column(df: pl.DataFrame, pk_col: str) -> pl.DataFrame`**
- Convert pK to ΔG using Polars expressions
- Formula: ΔG (kcal/mol) = RT ln(10^(-pK)) / 4184
- Vectorized operation (fast!)

**`filter_by_sample_size(df: pl.DataFrame, group_col: str, min_samples: int) -> pl.DataFrame`**
- Filter groups with insufficient samples
- Uses Polars groupby + filter
- Returns filtered DataFrame

**`split_by_target(df: pl.DataFrame, group_col: str) -> dict[str, pl.DataFrame]`**
- Split DataFrame by target/group
- Returns dict of {target_name: subset_df}

---

### **2. `src/results/metrics.py` - Statistical Metrics**

#### Functions:

**`calculate_per_target_metrics(df: pl.DataFrame, group_col: str, pred_col: str, truth_col: str) -> pl.DataFrame`**
- Compute Pearson R, Kendall's Tau, RMSE per target
- Strategy:
  ```python
  1. Polars groupby to split by target
  2. For each group: extract .to_numpy()
  3. Call scipy functions (pearsonr, kendalltau, etc.)
  4. Collect results into Polars DataFrame
  ```
- Returns: `pl.DataFrame` with columns:
  - `Target` (str)
  - `Pearson's R` (float)
  - `Kendall's Tau` (float)
  - `RMSE` (float)
  - `Sample size` (int)

**`weighted_mean_with_bootstrap_ci(values: np.ndarray | pl.Series, weights: np.ndarray | pl.Series, n_bootstrap: int = 10000) -> tuple[float, float, float]`**
- Bootstrap confidence intervals for weighted mean
- Accepts Polars Series or Numpy arrays
- Returns: (mean, lower_ci, upper_ci)

**`pearson_r(y_true: np.ndarray, y_pred: np.ndarray) -> float`**
- Wrapper for scipy.stats.pearsonr
- Returns correlation coefficient

**`kendall_tau(y_true: np.ndarray, y_pred: np.ndarray) -> float`**
- Wrapper for scipy.stats.kendalltau
- Returns tau coefficient

**`rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float`**
- Root mean squared error
- Returns RMSE value

---

### **3. `src/results/plotting.py` - Plotly Visualization**

#### Functions:

**`plot_scatter_with_marginals(df: pl.DataFrame, x_col: str, y_col: str, color_col: str = None, title: str = None, outpath: str = None) -> go.Figure`**
- Scatter plot with marginal distributions (histograms)
- Uses `plotly.graph_objects` or `plotly.express`
- Accepts Polars DataFrame natively
- Reproduces Figure 2 from paper (prediction vs truth)
- Returns: `plotly.graph_objects.Figure`

**`plot_bar_comparison(df_model: pl.DataFrame, df_baseline: pl.DataFrame, metrics: list[str], targets: list[str], title: str = None, outpath: str = None) -> go.Figure`**
- Grouped bar chart comparing model vs baseline
- Used for per-target metric comparison (Figure 4)
- Accepts Polars DataFrames
- Returns: `plotly.graph_objects.Figure`

**`plot_correlation_heatmap(df: pl.DataFrame, metrics: list[str], targets: list[str], title: str = None, outpath: str = None) -> go.Figure`**
- Heatmap of correlation metrics across targets
- Uses `plotly.graph_objects.Heatmap`
- Accepts Polars DataFrame
- Returns: `plotly.graph_objects.Figure`

**`plot_error_distribution(df: pl.DataFrame, pred_col: str, truth_col: str, bins: int = 50, title: str = None, outpath: str = None) -> go.Figure`**
- Histogram of prediction errors
- Accepts Polars DataFrame
- Returns: `plotly.graph_objects.Figure`

**`save_figure(fig: go.Figure, outpath: str, formats: list[str] = ['png', 'pdf'])`**
- Save figure in multiple formats
- PNG for viewing, PDF for publication
- Uses `fig.write_image()`

---

### **4. `src/results/config.py` - Plot Configuration**

#### Constants:

**Plot Style:**
```python
PLOT_STYLE = {
    'template': 'plotly_white',  # Clean publication style
    'font_family': 'Arial',
    'font_size': 12,
    'title_font_size': 14,
    'width': 800,
    'height': 600,
}

COLOR_SCHEMES = {
    'model_vs_baseline': ['#1f77b4', '#ff7f0e'],  # Blue vs Orange
    'per_target': px.colors.qualitative.Set2,     # Categorical
    'heatmap': 'RdBu_r',                          # Red-Blue diverging
}
```

**Figure Paths:**
```python
FIGURES_DIR = 'figures'
FIGURE_2_PATH = 'figures/figure2_predictions_vs_truth.png'
FIGURE_4_PATH = 'figures/figure4_per_target_comparison.png'
```

---

## Implementation Plan

### **Phase 1: Utils Module (Polars Data Ops)**
- [ ] `load_predictions()` - Load CSV/Parquet
- [ ] `pk_to_dg_column()` - Unit conversion
- [ ] `filter_by_sample_size()` - Filtering
- [ ] `split_by_target()` - Data splitting
- [ ] Unit tests with sample data

### **Phase 2: Metrics Module (Scipy + Polars)**
- [ ] `calculate_per_target_metrics()` - Core metrics function
- [ ] `weighted_mean_with_bootstrap_ci()` - Confidence intervals
- [ ] Helper wrappers (pearson_r, kendall_tau, rmse)
- [ ] Unit tests with known ground truth

### **Phase 3: Plotting Module (Plotly)**
- [ ] `plot_scatter_with_marginals()` - Figure 2 reproduction
- [ ] `plot_bar_comparison()` - Figure 4 reproduction
- [ ] `plot_correlation_heatmap()` - Additional analysis
- [ ] `plot_error_distribution()` - Error analysis
- [ ] `save_figure()` - Export utility
- [ ] Test with sample data

### **Phase 4: Config & Style**
- [ ] Define plot style constants
- [ ] Color schemes for different plot types
- [ ] Figure paths and naming conventions
- [ ] DPI and size settings for publication

### **Phase 5: Integration & Notebooks**
- [ ] Create `notebooks/` directory
- [ ] `notebooks/figure2_reproduction.ipynb` - Prediction scatter plots
- [ ] `notebooks/figure4_reproduction.ipynb` - Per-target comparison
- [ ] `notebooks/exploratory_analysis.ipynb` - Additional insights
- [ ] Document workflow in README

---

## Usage Example (Future)

### **Reproduce Figure 2: Predictions vs Truth**

```python
import polars as pl
from src.results import utils, plotting

# Load predictions (Polars)
baseline = utils.load_predictions("output/predictions/baseline.csv")
augmented = utils.load_predictions("output/predictions/augmented.csv")

# Convert units (Polars expression)
baseline = utils.pk_to_dg_column(baseline, pk_col='pred_pk')
augmented = utils.pk_to_dg_column(augmented, pk_col='pred_pk')

# Filter by sample size (Polars)
baseline = utils.filter_by_sample_size(baseline, group_col='target', min_samples=25)
augmented = utils.filter_by_sample_size(augmented, group_col='target', min_samples=25)

# Plot (Plotly with native Polars support - NO conversion!)
fig_baseline = plotting.plot_scatter_with_marginals(
    df=baseline,  # ← Polars DataFrame, no conversion needed!
    x_col='exp_dg',
    y_col='pred_dg',
    color_col='target',
    title='Baseline Model: Predicted vs Experimental ΔG',
    outpath='figures/figure2_baseline.png'
)

fig_augmented = plotting.plot_scatter_with_marginals(
    df=augmented,  # ← Polars DataFrame, no conversion needed!
    x_col='exp_dg',
    y_col='pred_dg',
    color_col='target',
    title='Augmented Model: Predicted vs Experimental ΔG',
    outpath='figures/figure2_augmented.png'
)

# Interactive viewing
fig_baseline.show()
fig_augmented.show()
```

### **Reproduce Figure 4: Per-Target Comparison**

```python
from src.results import metrics, plotting

# Calculate metrics per target (Polars → Scipy → Polars)
metrics_baseline = metrics.calculate_per_target_metrics(
    df=baseline,
    group_col='target',
    pred_col='pred_dg',
    truth_col='exp_dg'
)  # Returns Polars DataFrame

metrics_augmented = metrics.calculate_per_target_metrics(
    df=augmented,
    group_col='target',
    pred_col='pred_dg',
    truth_col='exp_dg'
)  # Returns Polars DataFrame

# Plot comparison (Plotly with native Polars - NO conversion!)
fig = plotting.plot_bar_comparison(
    df_model=metrics_augmented,      # ← Polars DataFrame
    df_baseline=metrics_baseline,    # ← Polars DataFrame
    metrics=['Pearson\'s R', 'Kendall\'s Tau'],
    targets=metrics_augmented['Target'].to_list(),
    title='Per-Target Performance: Augmented vs Baseline',
    outpath='figures/figure4_comparison.png'
)

fig.show()
```

**Key benefit**: 90% Polars (fast), 0% conversion overhead (Plotly native support)!

---

## Performance Considerations

### **Why Polars?**
- **Speed**: 5-10x faster than Pandas for filtering, groupby, joins
- **Memory**: Better for 100k+ prediction rows
- **Type safety**: Catches errors at expression time
- **Integration**: Matches our parquet metadata system

### **Why Plotly?**
- **Native Polars support**: No `.to_pandas()` needed (Plotly 5.15.0+)
- **Interactivity**: Hover, zoom, pan for exploration
- **Publication quality**: Export to PNG, PDF, SVG
- **Modern aesthetics**: Clean, professional default styles

### **Conversion Points** (Minimal)
- **Polars → Numpy**: Only for scipy functions (`.to_numpy()`)
- **Polars → Pandas**: NOT NEEDED for Plotly!

---

## Dependencies to Add

```toml
# Add to pyproject.toml
[project]
dependencies = [
    # ... existing deps ...
    "polars>=0.20.0",           # Fast DataFrame library
    "plotly>=5.15.0",           # Native Polars support
    "kaleido>=0.2.1",           # Static image export for Plotly
    "scipy>=1.9.0",             # Already included (statistical functions)
]
```

---

## Testing Strategy

### **Unit Tests**
- `tests/results/test_utils.py` - Data loading, filtering
- `tests/results/test_metrics.py` - Metric calculations with known ground truth
- `tests/results/test_plotting.py` - Figure generation (no visual inspection)

### **Integration Tests**
- End-to-end workflow with sample predictions
- Verify figure files are created
- Check metric values match expected ranges

### **Visual Tests**
- Generate sample figures for manual inspection
- Compare with paper figures for aesthetics
- Check plot elements (axes, labels, legends)

---

## Documentation

### **README Updates**
- Add "Results Analysis" section
- Document notebook workflows
- Provide example commands
- Link to figure reproductions

### **Docstrings**
- All functions have numpy-style docstrings
- Include parameter types (Polars, Numpy, etc.)
- Provide usage examples
- Document return types

### **Notebooks**
- Step-by-step walkthroughs
- Explanatory markdown cells
- Inline figure displays
- Export options documented

---

## Future Enhancements

### **Potential Additions**
- [ ] Additional plot types (violin, box, ridge plots)
- [ ] Statistical significance testing (t-tests, ANOVA)
- [ ] Cross-validation analysis
- [ ] Feature importance visualization
- [ ] Model ensemble uncertainty quantification
- [ ] Interactive dashboards (Dash/Streamlit)

### **Optimization**
- [ ] Lazy evaluation for large datasets (Polars LazyFrame)
- [ ] Parallel metric calculation (multiprocessing)
- [ ] Caching intermediate results (joblib)
- [ ] GPU-accelerated plotting (Datashader for 1M+ points)

---

## Timeline

**After pipeline runs successfully:**
1. **Week 1**: Implement utils.py + metrics.py + tests
2. **Week 2**: Implement plotting.py + style config + tests
3. **Week 3**: Create notebooks + reproduce Figures 2 & 4
4. **Week 4**: Documentation + integration testing + polish

**Estimated effort**: ~4 weeks (parallel with pipeline runs)

---

## Success Criteria

✅ **Functional Requirements:**
- [ ] Reproduce Figure 2 (predictions vs truth) with Plotly
- [ ] Reproduce Figure 4 (per-target comparison) with Plotly
- [ ] Calculate all metrics from paper (Pearson R, Kendall's Tau, RMSE)
- [ ] Generate publication-quality figures (PNG + PDF)
- [ ] All operations use Polars (no Pandas conversion)

✅ **Performance Requirements:**
- [ ] Load 100k predictions in <1 second
- [ ] Calculate per-target metrics for 50 targets in <5 seconds
- [ ] Generate complex plots in <2 seconds
- [ ] Export figures in <1 second

✅ **Code Quality:**
- [ ] 90%+ test coverage for results package
- [ ] All functions have docstrings
- [ ] Type hints for all parameters
- [ ] Passing linters (ruff, mypy)

---

## Notes

- **Plotly Narwhals Support**: Plotly 5.15.0+ uses Narwhals for universal DataFrame support. This means we can pass Polars DataFrames directly to Plotly Express and Graph Objects!
- **No Seaborn**: We're NOT using seaborn because it requires Pandas conversion. Plotly provides equivalent statistical plots with better interactivity and native Polars support.
- **Publication Quality**: Plotly exports to vector formats (PDF, SVG) for publication. Use `fig.write_image()` with `kaleido` backend.
- **Interactivity**: Notebooks can display interactive plots with `fig.show()`. For static exports, use `fig.write_image()`.

---

## Questions / Decisions Needed

- [ ] Which additional figures should we reproduce from the paper?
- [ ] Should we create a CLI for figure generation (e.g., `aev-plig-figures`)?
- [ ] What level of interactivity do we want in notebooks vs static exports?
- [ ] Should we support Altair as an alternative to Plotly?

---

**Last Updated:** 2026-02-10
**Status:** Planning Complete - Ready for implementation after pipeline runs
