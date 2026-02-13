"""Utilities for loading prediction results and computing metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau, pearsonr


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
