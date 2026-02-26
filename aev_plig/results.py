"""Utilities for loading prediction results and computing metrics."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_1samp


def load_predictions(path: str) -> pl.DataFrame:
    """Load an existing prediction parquet file.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ValueError: If ``path`` is not a parquet file.
    """
    parquet_path = Path(path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Prediction parquet not found: {parquet_path}")
    if parquet_path.suffix.lower() != ".parquet":
        raise ValueError(f"Expected a parquet file, got: {parquet_path}")
    return pl.read_parquet(parquet_path)


def load_all_predictions(
    directory: str | Path,
    prediction_file_name: str,
    path_regex: str | None = (
        r"(?P<model_name>[^/\\]+)[/\\]"
        r"(?P<model_instance>[^/\\]+)[/\\]"
        r"[^/\\]+"
    ),
) -> pl.DataFrame:
    """Recursively find ``{prediction_file_name}.parquet`` under *directory*.

    Intended for comparing multiple model architectures/instances on the same
    benchmark dataset.  The *prediction_file_name* is the dataset stem produced
    by ``scripts/predict.py``, e.g.::

        "pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark_predictions"

    Under the standard output layout::

        {directory}/{model_name}/{model_instance}/{prediction_file_name}.parquet

    Named capture groups in *path_regex* (applied to each file's path relative
    to *directory*, always forward-slash separated) become additional columns.

    Parameters
    ----------
    directory:
        Root directory to scan (e.g. ``"output/predictions"``).
    prediction_file_name:
        Stem of the target parquet file (no extension, no wildcards).
    path_regex:
        Regex with named groups applied to the relative path; ``None`` disables
        provenance parsing.

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist or no matching parquet files are found.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {root}")

    glob_pattern = f"**/{prediction_file_name}.parquet"
    files = sorted(root.glob(glob_pattern))
    if not files:
        raise FileNotFoundError(
            f"No files matching '{glob_pattern}' found under {root}"
        )

    compiled = re.compile(path_regex) if path_regex else None
    frames: list[pl.DataFrame] = []
    for f in files:
        frame = pl.read_parquet(f)

        if compiled is not None:
            rel = f.relative_to(root).as_posix()
            m = compiled.search(rel)
            if m:
                for col_name, col_val in m.groupdict().items():
                    frame = frame.with_columns(pl.lit(col_val or "").alias(col_name))

        frames.append(frame)

    return pl.concat(frames, how="diagonal_relaxed")


def rmse(y_true, y_pred) -> float:
    """Compute root mean squared error."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true_arr - y_pred_arr) ** 2)))


def pearson_r(y_true, y_pred) -> float:
    """Compute Pearson correlation coefficient."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(pearsonr(y_true_arr, y_pred_arr)[0])


def kendall_tau(y_true, y_pred) -> float:
    """Compute Kendall tau correlation coefficient."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    return float(kendalltau(y_true_arr, y_pred_arr)[0])


def overall_metrics(df: pl.DataFrame, truth_col: str, pred_col: str = "preds") -> dict[str, float]:
    """Compute overall RMSE and rank correlations for a prediction dataframe."""
    if truth_col not in df.columns:
        raise KeyError(f"Missing truth column: {truth_col}")
    if pred_col not in df.columns:
        raise KeyError(f"Missing prediction column: {pred_col}")

    clean = df.select([truth_col, pred_col]).drop_nulls()
    y_true = clean[truth_col].to_numpy()
    y_pred = clean[pred_col].to_numpy()

    return {
        "n": float(clean.height),
        "rmse": rmse(y_true, y_pred),
        "pearson_r": pearson_r(y_true, y_pred),
        "kendall_tau": kendall_tau(y_true, y_pred),
    }


def per_target_metrics(
    df: pl.DataFrame,
    target_col: str,
    truth_col: str,
    pred_col: str = "preds",
    min_samples: int = 1,
) -> pl.DataFrame:
    """Compute per-target metrics.

    Returns columns: target, n, rmse, pearson_r, kendall_tau.
    """
    for col in (target_col, truth_col, pred_col):
        if col not in df.columns:
            raise KeyError(f"Missing column: {col}")

    clean = df.select([target_col, truth_col, pred_col]).drop_nulls()

    rows: list[dict[str, float | int | str]] = []
    for target_value, group_df in clean.group_by(target_col, maintain_order=True):
        if isinstance(target_value, tuple) and len(target_value) == 1:
            target_value = target_value[0]
        n_samples = group_df.height
        if n_samples < min_samples:
            continue

        y_true = group_df[truth_col].to_numpy()
        y_pred = group_df[pred_col].to_numpy()
        rows.append(
            {
                "target": str(target_value),
                "n": n_samples,
                "rmse": rmse(y_true, y_pred),
                "pearson_r": pearson_r(y_true, y_pred),
                "kendall_tau": kendall_tau(y_true, y_pred),
            }
        )

    if not rows:
        return pl.DataFrame(
            schema={
                "target": pl.Utf8,
                "n": pl.Int64,
                "rmse": pl.Float64,
                "pearson_r": pl.Float64,
                "kendall_tau": pl.Float64,
            }
        )

    return pl.DataFrame(rows)


def js_divergence(y_true, y_pred, n_bins: int = 50) -> float:
    """Jensen-Shannon divergence between predicted and true pK distributions.

    Symmetric and bounded in [0, log(2)], making it more interpretable than
    raw KL divergence for comparing prediction vs truth distributions.
    """
    from scipy.stats import entropy as _entropy
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    lo = min(float(y_true_arr.min()), float(y_pred_arr.min()))
    hi = max(float(y_true_arr.max()), float(y_pred_arr.max()))
    bins = np.linspace(lo, hi, n_bins + 1)
    p, _ = np.histogram(y_true_arr, bins=bins)
    q, _ = np.histogram(y_pred_arr, bins=bins)
    p = p.astype(float) + 1e-10
    p /= p.sum()
    q = q.astype(float) + 1e-10
    q /= q.sum()
    m = 0.5 * (p + q)
    return float(0.5 * _entropy(p, m) + 0.5 * _entropy(q, m))


def log_evaluation_to_wandb(run, df, model_cfg: dict, truth_col: str = "pK") -> None:
    """Log full evaluation metrics and plots to an active WandB run.

    Computes ensemble accuracy scalars, per-seed spread, Wilcoxon pairwise
    significance tests, optional uncertainty metrics, and parity/residual plots.

    Args:
        run: Active ``wandb.Run`` object.
        df: Predictions dataframe (pandas or polars) containing truth_col and
            ``preds`` column plus optional ``preds_{seed}`` columns.
        model_cfg: Model config dict (from ``config.json``).
        truth_col: Name of the ground-truth column (default: ``"pK"``).
    """
    from scipy.stats import wilcoxon
    from aev_plig.training import concordance_index as _ci

    # Normalise to polars for consistent column access
    if not isinstance(df, pl.DataFrame):
        pl_df = pl.from_pandas(df)
    else:
        pl_df = df

    if truth_col not in pl_df.columns or "preds" not in pl_df.columns:
        return

    clean = pl_df.select([truth_col, "preds"]).drop_nulls()
    y_true = clean[truth_col].to_numpy()
    y_pred = clean["preds"].to_numpy()
    abs_res = np.abs(y_true - y_pred)
    residuals = y_true - y_pred

    pr, pr_p = pearsonr(y_true, y_pred)
    sr, sr_p = spearmanr(y_true, y_pred)
    kt, kt_p = kendalltau(y_true, y_pred)
    _, bias_p = ttest_1samp(residuals, 0)

    summary: dict = {
        "eval_pearson_r":         float(pr),
        "eval_pearson_pvalue":    float(pr_p),
        "eval_spearman_r":        float(sr),
        "eval_spearman_pvalue":   float(sr_p),
        "eval_kendall_tau":       float(kt),
        "eval_kendall_pvalue":    float(kt_p),
        "eval_concordance_index": float(_ci(y_true, y_pred)),
        "eval_rmse":              rmse(y_true, y_pred),
        "eval_mae":               float(abs_res.mean()),
        "eval_bias_mean":         float(residuals.mean()),
        "eval_bias_pvalue":       float(bias_p),
        "eval_js_divergence":     js_divergence(y_true, y_pred),
        "eval_success_0.5pK":     float((abs_res <= 0.5).mean()),
        "eval_success_1.0pK":     float((abs_res <= 1.0).mean()),
        "eval_success_1.5pK":     float((abs_res <= 1.5).mean()),
        "eval_success_2.0pK":     float((abs_res <= 2.0).mean()),
        "eval_n":                 float(len(y_true)),
    }

    # Per-seed spread and Wilcoxon pairwise tests
    seed_cols = sorted(c for c in pl_df.columns if c.startswith("preds_") and c != "preds")
    if len(seed_cols) > 1:
        seed_pearson, seed_rmse_vals, seed_tau = [], [], []
        seed_preds: list[np.ndarray] = []
        for col in seed_cols:
            s_pred = pl_df[col].drop_nulls().to_numpy()
            # align lengths with y_true if needed
            n = min(len(y_true), len(s_pred))
            seed_preds.append(s_pred[:n])
            seed_pearson.append(float(pearsonr(y_true[:n], s_pred[:n])[0]))
            seed_rmse_vals.append(rmse(y_true[:n], s_pred[:n]))
            seed_tau.append(float(kendalltau(y_true[:n], s_pred[:n])[0]))

        summary.update({
            "eval_seed_pearson_mean": float(np.mean(seed_pearson)),
            "eval_seed_pearson_std":  float(np.std(seed_pearson)),
            "eval_seed_pearson_min":  float(np.min(seed_pearson)),
            "eval_seed_pearson_max":  float(np.max(seed_pearson)),
            "eval_seed_rmse_mean":    float(np.mean(seed_rmse_vals)),
            "eval_seed_rmse_std":     float(np.std(seed_rmse_vals)),
            "eval_seed_tau_mean":     float(np.mean(seed_tau)),
            "eval_seed_tau_std":      float(np.std(seed_tau)),
        })

        # Wilcoxon signed-rank pairwise (upper triangle)
        for i, (col_i, pi) in enumerate(zip(seed_cols, seed_preds)):
            for j, (col_j, pj) in enumerate(zip(seed_cols, seed_preds)):
                if j <= i:
                    continue
                n = min(len(pi), len(pj))
                try:
                    _, w_p = wilcoxon(pi[:n] - y_true[:n], pj[:n] - y_true[:n])
                    si = col_i.replace("preds_", "")
                    sj = col_j.replace("preds_", "")
                    summary[f"eval_wilcoxon_pvalue_{si}_vs_{sj}"] = float(w_p)
                except Exception:
                    pass

    # Optional uncertainty metrics via uncertainty-toolbox (Bayesian models)
    var_col = next((c for c in pl_df.columns if c == "preds_var"), None)
    if var_col is not None:
        try:
            import uncertainty_toolbox as uct
            y_std = np.sqrt(pl_df[var_col].to_numpy())
            summary["eval_uncertainty_mace"] = float(
                uct.mean_absolute_calibration_error(y_pred, y_std, y_true)
            )
            summary["eval_uncertainty_nll"] = float(
                uct.nll_gaussian(y_pred, y_std, y_true)
            )
        except Exception:
            pass

    run.summary.update(summary)

    # Plots
    try:
        import plotly.graph_objects as go
        import wandb as _wandb

        # Parity plot
        fig_parity = go.Figure()
        fig_parity.add_trace(go.Scatter(
            x=y_true.tolist(), y=y_pred.tolist(),
            mode="markers", marker=dict(opacity=0.5, size=5),
            name="predictions",
        ))
        lo_val = float(min(y_true.min(), y_pred.min()))
        hi_val = float(max(y_true.max(), y_pred.max()))
        fig_parity.add_trace(go.Scatter(
            x=[lo_val, hi_val], y=[lo_val, hi_val],
            mode="lines", line=dict(dash="dash", color="red"),
            name="ideal",
        ))
        fig_parity.update_layout(
            xaxis_title=truth_col,
            yaxis_title="Predicted pK",
            title=f"Parity plot  R={float(pr):.3f}  RMSE={summary['eval_rmse']:.3f}",
        )
        run.log({"eval_parity_plot": _wandb.Plotly(fig_parity)})

        # Residuals vs predicted
        fig_res = go.Figure()
        fig_res.add_trace(go.Scatter(
            x=y_pred.tolist(), y=residuals.tolist(),
            mode="markers", marker=dict(opacity=0.5, size=5),
            name="residuals",
        ))
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            xaxis_title="Predicted pK",
            yaxis_title=f"Residual ({truth_col} − Predicted)",
            title="Residuals vs Predicted",
        )
        run.log({"eval_residuals_plot": _wandb.Plotly(fig_res)})

        # Distribution comparison
        import plotly.figure_factory as ff
        fig_dist = ff.create_distplot(
            [y_true.tolist(), y_pred.tolist()],
            group_labels=[truth_col, "Predicted"],
            show_hist=False,
        )
        fig_dist.update_layout(
            xaxis_title="pK", title="True vs Predicted Distribution",
        )
        run.log({"eval_distribution_plot": _wandb.Plotly(fig_dist)})

    except Exception:
        pass
