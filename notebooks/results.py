import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import re

    import numpy as np
    import polars as pl
    import uncertainty_toolbox as uct
    import plotly.express as px
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.special import erf
    from scipy.stats import kendalltau as scipy_kendalltau

    from aev_plig import results

    return (
        erf,
        go,
        make_subplots,
        np,
        pl,
        px,
        re,
        results,
        scipy_kendalltau,
        uct,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # AEV-PLIG Results Analysis
    Multi-model accuracy · ensemble agreement · uncertainty calibration · ranking · outliers
    """)
    return


@app.cell
def _():
    TRAINED_MODEL_NAMES = [
        "model_GATv2Net_ligsim90_fep_benchmark",
        "GATv2NetAleatoric_20260212_173128",
        "GATv2NetBayesianMixedPrecision_2026-02-27_01-00",
        "GATv2NetMixedPrecision_2026-02-26_13-00",
    ]
    DATA_NAME          = "pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark"
    TRUTH_COL          = "pK"
    PRED_COL           = "preds"
    UID_COL            = "unique_id"
    TOP_N_OUTLIERS     = 20
    MIN_TARGET_SAMPLES = 0
    FIG_DIR            = None   # set to a Path to auto-save HTML figures
    return (
        DATA_NAME,
        MIN_TARGET_SAMPLES,
        PRED_COL,
        TOP_N_OUTLIERS,
        TRAINED_MODEL_NAMES,
        TRUTH_COL,
        UID_COL,
    )


@app.cell
def _(DATA_NAME, PRED_COL, TRAINED_MODEL_NAMES, TRUTH_COL, pl, re, results):
    df = results.load_all_predictions(TRAINED_MODEL_NAMES, data_name=DATA_NAME)

    # Auto-detect ensemble member columns (preds_0, preds_1, ...)
    pred_member_cols = sorted(
        [c for c in df.columns if re.fullmatch(r"preds_\d+", c)],
        key=lambda c: int(c.split("_")[1]),
    )
    n_models = len(pred_member_cols)

    print(f"Rows: {df.height:,} | ensemble members: {n_models}")

    # Ensemble std = std dev of predictions across checkpoints
    df = df.with_columns(
        (
            pl.concat_list([pl.col(c) for c in pred_member_cols]).list.std()
            if n_models > 1
            else pl.lit(0.0)
        ).alias("ensemble_std")
    )
    df = df.drop([c for c in df.columns if "var" in c])

    # Residual (signed: predicted − true)
    if TRUTH_COL in df.columns:
        df = df.with_columns((pl.col(PRED_COL) - pl.col(TRUTH_COL)).alias("residual"))

    df
    return df, pred_member_cols


@app.cell
def _(mo):
    mo.md("""
    ## §1 Data Overview
    """)
    return


@app.cell
def _(TRUTH_COL, df, pl, px):
    clean = df.filter(pl.col(TRUTH_COL).is_not_null())
    _color_col = "model_name" if "model_name" in clean.columns else None
    _fig = px.histogram(
        clean.to_dict(as_series=False),
        x=TRUTH_COL,
        nbins=40,
        color=_color_col,
        barmode="overlay",
        opacity=0.7,
        title=f"Distribution of true {TRUTH_COL}  (n={clean.height:,})",
    )
    print(clean.select(TRUTH_COL).describe())
    _fig
    return (clean,)


@app.cell
def _(PRED_COL, TRUTH_COL, clean, pl, px):
    _value_cols = [TRUTH_COL, PRED_COL]
    _long = (
        clean.select(["model_name"] + _value_cols)
        .drop_nulls()
        .unpivot(
            on=_value_cols,
            index="model_name",
            variable_name="source",
            value_name="value",
        )
        .with_columns(
            pl.when(pl.col("source") == TRUTH_COL)
            .then(pl.lit("true_pk"))
            .otherwise(pl.lit("pred_pk"))
            .alias("source")
        )
    )
    _fig = px.violin(
        _long.to_dict(as_series=False),
        x="source",
        y="value",
        color="source",
        facet_col="model_name",
        box=True,
        points=False,
        title="true_pk vs pred_pk per model",
    )
    _fig.update_layout(width=1200, height=600)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §2 Ensemble Member Metrics
    RMSE, Pearson R, and Kendall τ for every ensemble member and the ensemble average,
    grouped by `trained_model_name` when multiple model runs are loaded.
    """)
    return


@app.cell
def _(PRED_COL, TRUTH_COL, df, pl, pred_member_cols, px, scipy_kendalltau):
    _id_cols = [TRUTH_COL] + (
        ["trained_model_name"] if "trained_model_name" in df.columns else []
    )
    _long = (
        df.select(_id_cols + pred_member_cols + [PRED_COL])
        .drop_nulls()
        .unpivot(
            on=pred_member_cols + [PRED_COL],
            index=_id_cols,
            variable_name="model_col",
            value_name="pred",
        )
    )
    _group_keys = (
        ["trained_model_name", "model_col"]
        if "trained_model_name" in _long.columns
        else ["model_col"]
    )
    metrics_df = (
        _long.group_by(_group_keys, maintain_order=True)
        .agg(
            ((pl.col("pred") - pl.col(TRUTH_COL)).pow(2).mean().sqrt()).alias("RMSE"),
            pl.corr("pred", TRUTH_COL).alias("Pearson R"),
            pl.map_groups(
                ["pred", TRUTH_COL],
                lambda s: scipy_kendalltau(s[0].to_numpy(), s[1].to_numpy()).statistic,
                return_dtype=pl.Float64,
                returns_scalar=True,
            ).alias("Kendall τ"),
        )
        .sort(_group_keys)
    )
    _fig = px.bar(
        metrics_df.to_dict(as_series=False),
        x="model_col",
        y="RMSE",
        color="trained_model_name" if "trained_model_name" in metrics_df.columns else None,
        barmode="group",
        title="RMSE per ensemble member vs ensemble average",
        labels={"model_col": "Model"},
    )
    _fig
    return (metrics_df,)


@app.cell
def _(metrics_df):
    metrics_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §3 Predicted vs True
    Colour shows ensemble std (std dev across ensemble checkpoints).
    """)
    return


@app.cell
def _(PRED_COL, TRUTH_COL, UID_COL, df, px):
    def _plot_pred_vs_true(
        _df,
        truth_col,
        pred_col,
        uid_col,
        std_col="ensemble_std",
        facet_col="model_name",
        title=None,
        subtitle=None,
        height=600,
    ):
        _cols = [truth_col, pred_col, uid_col, std_col]
        if facet_col in _df.columns:
            _cols.append(facet_col)
        _plot_df = _df.select(_cols).drop_nulls()
        _x_min = float(_plot_df[truth_col].min())
        _x_max = float(_plot_df[truth_col].max())
        if title is None:
            title = "Predicted vs True"
        if subtitle:
            title = f"{title}<br><sup>{subtitle}</sup>"
        _fig = px.scatter(
            _plot_df.to_dict(as_series=False),
            x=truth_col,
            y=pred_col,
            color=std_col,
            color_continuous_scale="Viridis",
            facet_col=facet_col if facet_col in _plot_df.columns else None,
            hover_data={uid_col: True, truth_col: ":.4f", pred_col: ":.4f", std_col: ":.4f"},
            title=title,
            height=height,
        )
        _fig.add_shape(
            type="line",
            x0=_x_min, y0=_x_min, x1=_x_max, y1=_x_max,
            line=dict(dash="dash", color="black"),
            xref="x", yref="y",
        )
        _fig.update_xaxes(range=[_x_min, _x_max], scaleanchor="y")
        _fig.update_yaxes(range=[_x_min, _x_max])
        _fig.update_layout(
            coloraxis_colorbar=dict(title="Ensemble Std Dev"),
            margin=dict(l=40, r=40, t=80, b=40),
        )
        return _fig

    _plot_pred_vs_true(
        df, TRUTH_COL, PRED_COL, UID_COL,
        title="Model Calibration Comparison",
        subtitle="Colour indicates ensemble std",
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## §4 Residual Analysis
    """)
    return


@app.cell
def _(TRUTH_COL, UID_COL, df, go, make_subplots):
    _res = df.select([TRUTH_COL, "residual", UID_COL]).drop_nulls()
    _fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Residual distribution", f"Residual vs True {TRUTH_COL}"],
    )
    _fig.add_trace(
        go.Histogram(x=_res["residual"].to_numpy(), nbinsx=40, name="Residuals"),
        row=1, col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_res[TRUTH_COL].to_numpy(),
            y=_res["residual"].to_numpy(),
            mode="markers",
            text=_res[UID_COL].to_list(),
            marker=dict(opacity=0.5),
            name="Residual",
        ),
        row=1, col=2,
    )
    _fig.add_hline(y=0, line_dash="dash", line_color="black")
    _fig.update_layout(title="Residual Analysis", showlegend=False)
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §5 Uncertainty Calibration

    **`ensemble_std`** — std dev of predictions across ensemble checkpoints.
    Uniform for all model types. Used as the uncertainty estimate in all calibration plots.

    Diagnostics shown:
    1. **Reliability diagram** — mean |error| per `ensemble_std` bin; good calibration is monotone
    2. **Sparsification curve** — RMSE as uncertain predictions are included
    3. **Prediction interval coverage** — fraction of true values within ±kσ vs Gaussian ideal
    """)
    return


@app.cell
def _(df):
    """Pass df through; uncertainty cells group by model_name directly."""
    df_typed = df
    return (df_typed,)


@app.cell
def _(df_typed, go, np, pl, px):
    def _reliability_diagram(unc, abs_err, n_bins=10):
        bin_edges = np.percentile(unc, np.linspace(0, 100, n_bins + 1))
        bin_idx   = np.searchsorted(bin_edges[1:-1], unc)
        counts    = np.bincount(bin_idx, minlength=n_bins).astype(float)
        mean_unc  = np.bincount(bin_idx, weights=unc,     minlength=n_bins) / counts
        mean_err  = np.bincount(bin_idx, weights=abs_err, minlength=n_bins) / counts
        return mean_unc, mean_err

    _fig = go.Figure()
    _colors = px.colors.qualitative.Plotly
    _model_names = [m for m in df_typed["model_name"].unique().to_list() if m is not None]
    _color_map = {m: _colors[i % len(_colors)] for i, m in enumerate(_model_names)}

    for _mn in _model_names:
        _sub = (
            df_typed.filter(pl.col("model_name") == _mn)
            .select(["ensemble_std", "residual"])
            .drop_nulls()
        )
        _unc = _sub["ensemble_std"].to_numpy()
        _abs_err = np.abs(_sub["residual"].to_numpy())
        _mu_unc, _mu_err = _reliability_diagram(_unc, _abs_err)
        _color = _color_map[_mn]
        _fig.add_trace(go.Scatter(
            x=_mu_unc, y=_mu_err, mode="markers+lines",
            name=_mn, line=dict(color=_color), marker=dict(color=_color),
        ))
        _slope, _intercept = np.polyfit(_mu_unc, _mu_err, 1)
        _xs = np.linspace(_mu_unc.min(), _mu_unc.max(), 100)
        _fig.add_trace(go.Scatter(
            x=_xs, y=_slope * _xs + _intercept, mode="lines",
            line=dict(color=_color, dash="dash"), showlegend=False,
        ))
    _fig.update_layout(
        title="Reliability Diagram",
        xaxis_title="Mean ensemble std",
        yaxis_title="Mean |Residual|",
        width=800, height=600,
    )
    _fig
    return


@app.cell
def _(df_typed, go, np, pl):
    def _sparsification_curve(unc, residuals):
        order = np.argsort(unc)
        sq_err_sorted = residuals[order] ** 2
        cum_rmse = np.sqrt(np.cumsum(sq_err_sorted) / np.arange(1, len(sq_err_sorted) + 1))
        frac_retained = np.arange(1, len(sq_err_sorted) + 1) / len(sq_err_sorted)
        return frac_retained, cum_rmse

    _fig = go.Figure()
    for _mn2 in [m for m in df_typed["model_name"].unique().to_list() if m is not None]:
        _sub = (
            df_typed.filter(pl.col("model_name") == _mn2)
            .select(["ensemble_std", "residual"])
            .drop_nulls()
        )
        _frac, _cum_rmse = _sparsification_curve(
            _sub["ensemble_std"].to_numpy(), _sub["residual"].to_numpy()
        )
        _fig.add_trace(go.Scatter(x=_frac, y=_cum_rmse, mode="lines", name=_mn2))
    _fig.update_layout(
        title="Sparsification Curve",
        xaxis_title="Fraction retained (most confident first)",
        yaxis_title="Cumulative RMSE",
        width=800, height=600,
    )
    _fig
    return


@app.cell
def _(df_typed, erf, go, np, pl):
    def _interval_coverage(unc, residuals):
        k_vals = np.array([0.5, 1.0, 1.5, 2.0])
        obs_cov = (
            np.abs(residuals)[None, :] <= k_vals[:, None] * unc[None, :]
        ).mean(axis=1)
        expected_cov = erf(k_vals / np.sqrt(2))
        return k_vals, obs_cov, expected_cov

    _fig = go.Figure()
    _k_vals_ref = None
    _expected_ref = None
    for _mn3 in [m for m in df_typed["model_name"].unique().to_list() if m is not None]:
        _sub = (
            df_typed.filter(pl.col("model_name") == _mn3)
            .select(["ensemble_std", "residual"])
            .drop_nulls()
        )
        _k_vals, _obs_cov, _expected_cov = _interval_coverage(
            _sub["ensemble_std"].to_numpy(), _sub["residual"].to_numpy()
        )
        _k_vals_ref = _k_vals
        _expected_ref = _expected_cov
        _fig.add_trace(go.Scatter(
            x=_k_vals, y=_obs_cov, mode="lines+markers", name=_mn3
        ))
    if _k_vals_ref is not None:
        _fig.add_trace(go.Scatter(
            x=_k_vals_ref, y=_expected_ref, mode="lines",
            line=dict(dash="dash"), name="Ideal Gaussian",
        ))
    _fig.update_layout(
        title="Prediction Interval Coverage",
        xaxis_title="k (±kσ)", yaxis_title="Coverage",
        yaxis=dict(range=[0, 1.05]), width=800, height=600,
    )
    _fig
    return


@app.cell
def _(df, np, pl, uct):
    def _risk_coverage_auc(unc, residuals):
        order = np.argsort(unc)
        res_sorted = residuals[order]
        N = len(res_sorted)
        coverage = np.arange(1, N + 1) / N
        risk_curve = np.sqrt(np.cumsum(res_sorted**2) / np.arange(1, N + 1))
        return float(np.trapz(risk_curve, coverage))

    _rows = []
    for _model_name in df["model_name"].unique().to_list():
        print(f'MODEL NAME {_model_name}')
        if _model_name is None:
            continue
        _dm = df.filter(pl.col("model_name") == _model_name).drop_nulls()
        _preds_arr = _dm["preds"].to_numpy()
        _targets_arr = _dm["pK"].to_numpy()
        _unc_arr = _dm["ensemble_std"].to_numpy()
        _res_arr = _dm["residual"].to_numpy()
        _pearson_r = (
            _dm.with_columns(pl.col("residual").abs().alias("abs_res"))
            .select(pl.corr("ensemble_std", "abs_res"))
            .item()
        )
        _uct_m = uct.metrics.get_all_metrics(y_pred=_preds_arr, y_std=_unc_arr, y_true=_targets_arr)
        _rows.append({
            "model_name": _model_name,
            "Pearson_R (higher is better)": _pearson_r,
            "NLL (lower is better)": float(_uct_m["scoring_rule"]["nll"]),
            "MACE (lower is better)": float(_uct_m["avg_calibration"]["ma_cal"]),
            "Risk–Coverage AUC (lower is better)": _risk_coverage_auc(_unc_arr, _res_arr),
        })
    _uct_df = pl.DataFrame(_rows).sort("model_name") if _rows else pl.DataFrame()
    _uct_df
    return


@app.cell
def _(df, go, np, pl):
    _model_names = [m for m in df["model_name"].unique().to_list() if m is not None]

    # Calibration by bins
    _fig_cal = go.Figure()
    for _m in _model_names:
        _d = df.filter(pl.col("model_name") == _m).drop_nulls()
        _p = _d["preds"].to_numpy()
        _y = _d["pK"].to_numpy()
        _p_scaled = (_p - _p.min()) / (_p.max() - _p.min() + 1e-12)
        _bins = np.linspace(0, 1, 11)
        _idx = np.digitize(_p_scaled, _bins) - 1
        _x_pts = np.array([_p_scaled[_idx == i].mean() for i in range(10) if (_idx == i).any()])
        _y_pts = np.array([_y[_idx == i].mean() for i in range(10) if (_idx == i).any()])
        _fig_cal.add_trace(go.Scatter(x=_x_pts, y=_y_pts, mode="lines+markers", name=str(_m)))
    _fig_cal.update_layout(
        title="Calibration (by bins)",
        xaxis_title="Mean predicted (scaled)", yaxis_title="Mean observed",
    )
    _fig_cal
    return


@app.cell
def _(df, go, np, pl):
    _model_names2 = [m for m in df["model_name"].unique().to_list() if m is not None]
    _fig_int = go.Figure()
    for _m2 in _model_names2:
        _d2 = df.filter(pl.col("model_name") == _m2).drop_nulls()
        _p2 = _d2["preds"].to_numpy()
        _y2 = _d2["pK"].to_numpy()
        _u2 = _d2["ensemble_std"].to_numpy()
        _order2 = np.argsort(_u2)
        _y_s2, _p_s2, _u_s2 = _y2[_order2], _p2[_order2], _u2[_order2]
        _inside2 = (_y_s2 >= (_p_s2 - 2 * _u_s2)) & (_y_s2 <= (_p_s2 + 2 * _u_s2))
        _fig_int.add_trace(go.Scatter(
            x=np.arange(1, len(_y_s2) + 1) / len(_y_s2),
            y=np.cumsum(_inside2) / np.arange(1, len(_y_s2) + 1),
            mode="lines", name=str(_m2),
        ))
    _fig_int.update_layout(
        title="Interval Coverage vs Fraction Retained (±2σ)",
        xaxis_title="Fraction retained", yaxis_title="Empirical coverage",
    )
    _fig_int
    return


@app.cell
def _(df, go, np, pl):
    _model_names3 = [m for m in df["model_name"].unique().to_list() if m is not None]
    _fig_rc = go.Figure()
    for _m3 in _model_names3:
        _d3 = df.filter(pl.col("model_name") == _m3).drop_nulls()
        _u3 = _d3["ensemble_std"].to_numpy()
        _r3 = _d3["residual"].to_numpy()
        _o3 = np.argsort(_u3)
        _r3s = _r3[_o3]
        _n3 = len(_r3s)
        _cov3 = np.arange(1, _n3 + 1) / _n3
        _rmse3 = np.sqrt(np.cumsum(_r3s * _r3s) / np.arange(1, _n3 + 1))
        _fig_rc.add_trace(go.Scatter(x=_cov3, y=_rmse3, mode="lines", name=str(_m3)))
    _fig_rc.update_layout(
        title="Risk–Coverage Curve",
        xaxis_title="Coverage (fraction retained)", yaxis_title="RMSE",
    )
    _fig_rc
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## §6 Per-Target Kendall τ (Ranking Ability)
    Each `unique_id` is treated as its own target.
    """)
    return


@app.cell
def _(MIN_TARGET_SAMPLES, PRED_COL, TRUTH_COL, UID_COL, df, px, results):
    _target_metrics = results.per_target_metrics(
        df.drop_nulls(),
        target_col=UID_COL,
        truth_col=TRUTH_COL,
        pred_col=PRED_COL,
        min_samples=MIN_TARGET_SAMPLES,
    )
    _fig = px.histogram(
        _target_metrics.to_dict(as_series=False),
        x="kendall_tau",
        nbins=30,
        title="Per-target Kendall τ distribution",
        labels={"kendall_tau": "Kendall τ"},
    )
    _fig.add_vline(x=0, line_dash="dash", line_color="red")
    _fig
    return


@app.cell
def _(MIN_TARGET_SAMPLES, PRED_COL, TRUTH_COL, UID_COL, df, results):
    results.per_target_metrics(
        df.drop_nulls(),
        target_col=UID_COL,
        truth_col=TRUTH_COL,
        pred_col=PRED_COL,
        min_samples=MIN_TARGET_SAMPLES,
    ).sort("kendall_tau")
    return


@app.cell
def _(mo):
    mo.md("""
    ## §7 Outlier Table
    """)
    return


@app.cell
def _(PRED_COL, TOP_N_OUTLIERS, TRUTH_COL, UID_COL, df, pl):
    _outlier_cols = (
        [UID_COL, TRUTH_COL, PRED_COL, "residual", "ensemble_std"]
        + (["trained_model_name"] if "trained_model_name" in df.columns else [])
    )
    (
        df.select(_outlier_cols)
        .drop_nulls()
        .with_columns(pl.col("residual").abs().alias("abs_residual"))
        .sort("abs_residual", descending=True)
        .head(TOP_N_OUTLIERS)
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## §8 Success Rate
    """)
    return


@app.cell
def _(df, pl):
    _thresholds = pl.Series([0.5, 1.0, 1.5, 2.0])
    _abs_res = df.select("residual").drop_nulls()["residual"].abs()
    pl.DataFrame({
        "Threshold (±pK)": _thresholds,
        "N within": _thresholds.map_elements(
            lambda t: int((_abs_res <= t).sum()), return_dtype=pl.Int64
        ),
        "% within": _thresholds.map_elements(
            lambda t: round(100.0 * float((_abs_res <= t).mean()), 1), return_dtype=pl.Float64
        ),
    })
    return


if __name__ == "__main__":
    app.run()
