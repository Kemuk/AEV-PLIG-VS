"""Utilities for loading prediction results and computing metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import kendalltau, pearsonr, spearmanr, ttest_1samp

from aev_plig.config import RetrievalConfig


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
    trained_model_names: list[str],
    data_name: str,
    predictions_dir: str | Path | None = None,
    trained_models_dir: str | Path | None = None,
) -> pl.DataFrame:
    """Load prediction parquets for a list of trained model names.

    For each name, reads ``{trained_models_dir}/{name}/config.json`` to infer
    the architecture (``model_name``), then loads::

        {predictions_dir}/{model_name}/{name}/{data_name}_predictions.parquet

    Adds provenance columns ``model_name`` and ``trained_model_name`` to each
    loaded frame before concatenating.

    Parameters
    ----------
    trained_model_names:
        List of trained model directory names (e.g.
        ``["model_GATv2Net_run1", "model_GATv2NetBayesian_run2"]``).
    data_name:
        Dataset stem used when saving predictions, e.g. ``"fep_benchmark"``.
    predictions_dir:
        Root predictions directory. Defaults to ``Config.PREDICTIONS_DIR``.
    trained_models_dir:
        Root trained-models directory. Defaults to ``Config.TRAINED_MODELS_DIR``.

    Raises
    ------
    FileNotFoundError
        If ``predictions_dir`` does not exist or a parquet file is not found.
    ValueError
        If ``trained_model_names`` is empty.
    """
    import json as _json

    from aev_plig.config import Config

    if not trained_model_names:
        raise ValueError("trained_model_names list is empty")

    pred_root = Path(predictions_dir or Config.PREDICTIONS_DIR)
    models_root = Path(trained_models_dir or Config.TRAINED_MODELS_DIR)

    if not pred_root.exists():
        raise FileNotFoundError(f"Predictions directory not found: {pred_root}")

    frames: list[pl.DataFrame] = []
    for name in trained_model_names:
        cfg_path = models_root / name / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as fh:
                model_name = _json.load(fh).get("model", Config.MODEL_NAME)
        else:
            model_name = Config.MODEL_NAME

        parquet_path = pred_root / model_name / name / f"{data_name}_predictions.parquet"
        frame = load_predictions(str(parquet_path))
        frame = frame.with_columns(
            pl.lit(model_name).alias("model_name"),
            pl.lit(name).alias("trained_model_name"),
        )
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


# ==================== Retrieval Diagnostics ====================


def analyze_false_positives(predictions_df, sdf_paths, top_k=None):
    """
    Analyze false positives from retrieval predictions.

    For each protein target, identifies the top-k ligands ranked highest by
    the model that are NOT among the top actives by actual ranking, then
    computes RDKit molecular descriptors.

    Args:
        predictions_df: pl.DataFrame from predict_retrieval() with columns
                        [protein, ligand, predicted_score, predicted_rank, actual_rank]
        sdf_paths: dict mapping ligand ID → SDF file path
        top_k: Number of top false positives to analyze per target
               (default from RetrievalConfig)

    Returns:
        pl.DataFrame with columns:
            protein, fp_ligand, predicted_rank, actual_rank,
            tanimoto_to_best_active, mw, logp
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors, DataStructs

    if top_k is None:
        top_k = RetrievalConfig.TOP_K_FALSE_POSITIVES

    results = []

    for protein, group in predictions_df.group_by('protein'):
        protein = protein[0] if isinstance(protein, tuple) else protein
        group = group.sort('predicted_rank')
        n = len(group)
        if n < 4:
            continue

        # Top quartile by actual rank = actives
        active_cutoff = max(1, n // 4)
        actives = group.filter(pl.col('actual_rank') <= active_cutoff)
        active_ids = set(actives['ligand'].to_list())

        # Best active's fingerprint (lowest actual_rank)
        best_active_id = actives.sort('actual_rank')['ligand'][0]
        best_active_fp = None
        if best_active_id in sdf_paths:
            mol = Chem.SDMolSupplier(sdf_paths[best_active_id], removeHs=True)
            if mol and mol[0] is not None:
                best_active_fp = AllChem.GetMorganFingerprintAsBitVect(mol[0], 2, nBits=2048)

        # False positives: high predicted rank but not actual actives
        fp_count = 0
        for row in group.iter_rows(named=True):
            if fp_count >= top_k:
                break
            if row['ligand'] in active_ids:
                continue

            fp_count += 1
            ligand_id = row['ligand']

            tanimoto = None
            mw = None
            logp = None

            if ligand_id in sdf_paths:
                suppl = Chem.SDMolSupplier(sdf_paths[ligand_id], removeHs=True)
                if suppl and suppl[0] is not None:
                    fp_mol = suppl[0]
                    mw = Descriptors.MolWt(fp_mol)
                    logp = Descriptors.MolLogP(fp_mol)
                    if best_active_fp is not None:
                        fp_fp = AllChem.GetMorganFingerprintAsBitVect(fp_mol, 2, nBits=2048)
                        tanimoto = DataStructs.TanimotoSimilarity(best_active_fp, fp_fp)

            results.append({
                'protein': protein,
                'fp_ligand': ligand_id,
                'predicted_rank': row['predicted_rank'],
                'actual_rank': row['actual_rank'],
                'tanimoto_to_best_active': tanimoto,
                'mw': mw,
                'logp': logp,
            })

    if not results:
        return pl.DataFrame({
            'protein': [], 'fp_ligand': [], 'predicted_rank': [],
            'actual_rank': [], 'tanimoto_to_best_active': [],
            'mw': [], 'logp': [],
        })

    return pl.DataFrame(results)


def summarize_diagnostics(df):
    """
    Compute aggregated statistics from false positive analysis.

    Args:
        df: pl.DataFrame from analyze_false_positives()

    Returns:
        dict with keys: tanimoto_mean, tanimoto_median, frac_tanimoto_gt_0.5,
        frac_tanimoto_gt_0.3, fp_mw_mean, fp_mw_std, fp_logp_mean,
        fp_logp_std, rank_gap_mean
    """
    if df.height == 0:
        return {}

    tanimoto = df['tanimoto_to_best_active'].drop_nulls()
    mw = df['mw'].drop_nulls()
    logp = df['logp'].drop_nulls()

    summary = {}

    if tanimoto.len() > 0:
        t_arr = tanimoto.to_numpy()
        summary['tanimoto_mean'] = float(np.mean(t_arr))
        summary['tanimoto_median'] = float(np.median(t_arr))
        summary['frac_tanimoto_gt_0.5'] = float(np.mean(t_arr > 0.5))
        summary['frac_tanimoto_gt_0.3'] = float(np.mean(t_arr > 0.3))

    if mw.len() > 0:
        mw_arr = mw.to_numpy()
        summary['fp_mw_mean'] = float(np.mean(mw_arr))
        summary['fp_mw_std'] = float(np.std(mw_arr))

    if logp.len() > 0:
        logp_arr = logp.to_numpy()
        summary['fp_logp_mean'] = float(np.mean(logp_arr))
        summary['fp_logp_std'] = float(np.std(logp_arr))

    # Mean rank gap (how far off are the false positives)
    rank_gap = (df['actual_rank'] - df['predicted_rank']).drop_nulls()
    if rank_gap.len() > 0:
        summary['rank_gap_mean'] = float(rank_gap.mean())

    return summary
