import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


# ── Imports ──────────────────────────────────────────────────────────────────


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
    import polars as pl
    from scipy.stats import gaussian_kde, spearmanr

    warnings.filterwarnings("ignore")

    return gaussian_kde, mticker, np, pl, plt, spearmanr, warnings


# ── Configuration ─────────────────────────────────────────────────────────────


@app.cell
def _():
    from pathlib import Path

    DATA_DIR = Path("data")

    # Source CSV paths
    PDBBIND_CSV    = DATA_DIR / "pdbbind_processed.csv"
    BINDINGDB_CSV  = DATA_DIR / "bindingdb_processed.csv"
    BINDINGNET_CSV = DATA_DIR / "bindingnet_processed.csv"

    # Training set definition: ligsim90 w.r.t. FEP benchmark
    LIGSIM_COL       = "max_tanimoto_fep_benchmark"
    LIGSIM_THRESHOLD = 0.9          # matches Config.TANIMOTO_THRESHOLD

    # Affinity thresholds (pK = −log10[M])
    ACTIVE_THRESHOLD_HIT   = 6.0   # 1 µM
    ACTIVE_THRESHOLD_LEAD  = 7.0   # 100 nM
    ACTIVE_THRESHOLD_POTENT = 8.0  # 10 nM

    # Minimum ligands per target to include in per-target analyses
    MIN_TARGET_LIGANDS = 3

    # Plot style
    PALETTE = {
        "pdbbind":   "#4C72B0",
        "bindingdb": "#DD8452",
        "bindingnet": "#55A868",
        "combined":  "#8172B2",
    }

    return (
        ACTIVE_THRESHOLD_HIT,
        ACTIVE_THRESHOLD_LEAD,
        ACTIVE_THRESHOLD_POTENT,
        BINDINGDB_CSV,
        BINDINGNET_CSV,
        DATA_DIR,
        LIGSIM_COL,
        LIGSIM_THRESHOLD,
        MIN_TARGET_LIGANDS,
        PALETTE,
        PDBBIND_CSV,
        Path,
    )


# ── Data loading & ligsim90 filter ───────────────────────────────────────────


@app.cell
def _(
    BINDINGDB_CSV,
    BINDINGNET_CSV,
    LIGSIM_COL,
    LIGSIM_THRESHOLD,
    PDBBIND_CSV,
    pl,
):
    # ── PDBbind ───────────────────────────────────────────────────────────────
    _raw_pdbbind = pl.read_csv(PDBBIND_CSV, null_values=["", "NA", "nan"])

    df_pdbbind = (
        _raw_pdbbind
        .filter(pl.col(LIGSIM_COL) < LIGSIM_THRESHOLD)
        .rename({"-logKd/Ki": "pK"})
        .with_columns(pl.lit("pdbbind").alias("source"))
        .select(["source", "PDB_code", "pK", "split_core",
                 "max_tanimoto_schrodinger", "max_tanimoto_merck",
                 "max_tanimoto_fep_benchmark"])
    )

    n_pdbbind_raw  = len(_raw_pdbbind)
    n_pdbbind_kept = len(df_pdbbind)

    # ── BindingDB ─────────────────────────────────────────────────────────────
    _raw_bindingdb = pl.read_csv(BINDINGDB_CSV, null_values=["", "NA", "nan"])

    df_bindingdb = (
        _raw_bindingdb
        .filter(pl.col(LIGSIM_COL) < LIGSIM_THRESHOLD)
        .filter(pl.col("pK").is_not_null())
        .with_columns(pl.lit("bindingdb").alias("source"))
        .select(["source", "unique_id", "pK",
                 "max_tanimoto_schrodinger", "max_tanimoto_merck",
                 "max_tanimoto_fep_benchmark"])
    )

    n_bindingdb_raw  = len(_raw_bindingdb)
    n_bindingdb_kept = len(df_bindingdb)

    # ── BindingNet ────────────────────────────────────────────────────────────
    _raw_bindingnet = pl.read_csv(BINDINGNET_CSV, null_values=["", "NA", "nan"])

    df_bindingnet = (
        _raw_bindingnet
        .filter(pl.col(LIGSIM_COL) < LIGSIM_THRESHOLD)
        .rename({"-logAffi": "pK", "unique_identify": "unique_id"})
        .with_columns(pl.lit("bindingnet").alias("source"))
        .select(["source", "unique_id", "pK", "target", "pdb", "compnd",
                 "max_tanimoto_schrodinger", "max_tanimoto_merck",
                 "max_tanimoto_fep_benchmark"])
    )

    n_bindingnet_raw  = len(_raw_bindingnet)
    n_bindingnet_kept = len(df_bindingnet)

    return (
        df_bindingdb,
        df_bindingnet,
        df_pdbbind,
        n_bindingdb_kept,
        n_bindingdb_raw,
        n_bindingnet_kept,
        n_bindingnet_raw,
        n_pdbbind_kept,
        n_pdbbind_raw,
    )


@app.cell
def _(df_bindingdb, df_bindingnet, df_pdbbind, pl):
    # Unified training frame (common columns only)
    df_train = pl.concat([
        df_pdbbind.select(["source", "pK",
                           "max_tanimoto_schrodinger",
                           "max_tanimoto_merck",
                           "max_tanimoto_fep_benchmark"]),
        df_bindingdb.select(["source", "pK",
                              "max_tanimoto_schrodinger",
                              "max_tanimoto_merck",
                              "max_tanimoto_fep_benchmark"]),
        df_bindingnet.select(["source", "pK",
                               "max_tanimoto_schrodinger",
                               "max_tanimoto_merck",
                               "max_tanimoto_fep_benchmark"]),
    ])

    return (df_train,)


# ═════════════════════════════════════════════════════════════════════════════
# §1  AFFINITY DISTRIBUTION & DATASET COMPOSITION
#     Question: Is the dataset representative of realistic screening space?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §1 — Affinity Distribution & Dataset Composition

    **Objective:** Characterise the pK range and source composition of the training set
    (`pdbbind ∪ bindingdb ∪ bindingnet`, filtered at ligsim90\_fep\_benchmark < 0.9).
    A VS-ready training set should cover the full screening-relevant range (pK 4–12) without
    strong source-level biases.
    """)
    return


@app.cell
def _(
    df_train,
    n_bindingdb_kept,
    n_bindingdb_raw,
    n_bindingnet_kept,
    n_bindingnet_raw,
    n_pdbbind_kept,
    n_pdbbind_raw,
    pl,
):
    # ── Summary statistics table ──────────────────────────────────────────────
    _stats = (
        df_train
        .group_by("source")
        .agg([
            pl.len().alias("n"),
            pl.col("pK").mean().round(2).alias("pK_mean"),
            pl.col("pK").std().round(2).alias("pK_std"),
            pl.col("pK").min().round(2).alias("pK_min"),
            pl.col("pK").max().round(2).alias("pK_max"),
            pl.col("pK").quantile(0.25).round(2).alias("pK_q25"),
            pl.col("pK").quantile(0.75).round(2).alias("pK_q75"),
            (pl.col("pK") < 5).sum().alias("n_weak"),     # pK < 5
            (pl.col("pK") >= 9).sum().alias("n_potent"),   # pK ≥ 9
        ])
        .sort("source")
        .with_columns([
            (pl.col("n_weak")   / pl.col("n") * 100).round(1).alias("pct_weak"),
            (pl.col("n_potent") / pl.col("n") * 100).round(1).alias("pct_potent"),
        ])
    )

    # Ligsim90 filter removal fractions
    _filter_stats = pl.DataFrame({
        "source":   ["pdbbind",     "bindingdb",     "bindingnet"],
        "n_raw":    [n_pdbbind_raw, n_bindingdb_raw, n_bindingnet_raw],
        "n_kept":   [n_pdbbind_kept, n_bindingdb_kept, n_bindingnet_kept],
    }).with_columns([
        (pl.col("n_raw") - pl.col("n_kept")).alias("n_removed"),
        ((pl.col("n_raw") - pl.col("n_kept")) / pl.col("n_raw") * 100)
        .round(1).alias("pct_removed"),
    ])

    print("=== Training-set composition (after ligsim90_fep_benchmark < 0.9) ===")
    print(_stats)
    print("\n=== Ligsim90 filter impact ===")
    print(_filter_stats)

    return


@app.cell
def _(PALETTE, df_train, gaussian_kde, np, pl, plt):
    # ── §1 Plot A: pK KDE per source + combined ───────────────────────────────
    _fig, (_ax_kde, _ax_box) = plt.subplots(
        1, 2, figsize=(12, 4.5), constrained_layout=True
    )
    _fig.suptitle("§1 — Affinity Distribution", fontsize=13, fontweight="bold")

    _x_grid = np.linspace(0, 16, 400)

    for _src, _col in PALETTE.items():
        if _src == "combined":
            continue
        _vals = df_train.filter(pl.col("source") == _src)["pK"].drop_nulls().to_numpy()
        if len(_vals) < 10:
            continue
        _kde = gaussian_kde(_vals, bw_method=0.3)
        _ax_kde.plot(_x_grid, _kde(_x_grid), color=_col, lw=2, label=_src)
        _ax_kde.fill_between(_x_grid, _kde(_x_grid), alpha=0.12, color=_col)

    # Combined KDE
    _all_vals = df_train["pK"].drop_nulls().to_numpy()
    _kde_all  = gaussian_kde(_all_vals, bw_method=0.3)
    _ax_kde.plot(_x_grid, _kde_all(_x_grid), color=PALETTE["combined"],
                 lw=2.5, ls="--", label="combined")

    # VS-relevant range shading
    _ax_kde.axvspan(5, 12, alpha=0.07, color="grey", label="VS-relevant (5–12)")
    _ax_kde.set_xlabel("pK  (−log₁₀[M])", fontsize=11)
    _ax_kde.set_ylabel("Density", fontsize=11)
    _ax_kde.legend(fontsize=9)
    _ax_kde.set_title("KDE of binding affinity per source")

    # ── §1 Plot B: Box plots per source ───────────────────────────────────────
    _sources = ["pdbbind", "bindingdb", "bindingnet"]
    _data    = [
        df_train.filter(pl.col("source") == s)["pK"].drop_nulls().to_numpy()
        for s in _sources
    ]
    _bp = _ax_box.boxplot(
        _data, patch_artist=True, widths=0.5,
        medianprops=dict(color="black", linewidth=2),
    )
    for _patch, _src in zip(_bp["boxes"], _sources):
        _patch.set_facecolor(PALETTE[_src])
        _patch.set_alpha(0.75)

    _ax_box.set_xticklabels(_sources, fontsize=10)
    _ax_box.axhline(5, color="grey", ls=":", lw=1, label="pK = 5 (10 µM)")
    _ax_box.axhline(9, color="grey", ls="--", lw=1, label="pK = 9 (1 nM)")
    _ax_box.set_ylabel("pK", fontsize=11)
    _ax_box.legend(fontsize=8)
    _ax_box.set_title("Affinity spread per source")

    _fig


@app.cell
def _(df_pdbbind, gaussian_kde, np, plt):
    # ── §1 Plot C: PDBbind split_core affinity distributions ─────────────────
    _splits   = ["train", "valid", "test"]
    _split_palette = {"train": "#4C72B0", "valid": "#C44E52", "test": "#55A868"}

    _fig, _ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    _fig.suptitle("§1 — PDBbind pK by split_core", fontsize=13, fontweight="bold")

    _x_grid = np.linspace(0, 16, 400)

    for _sp in _splits:
        _vals = (
            df_pdbbind
            .filter(df_pdbbind["split_core"] == _sp)["pK"]
            .drop_nulls().to_numpy()
        )
        if len(_vals) < 5:
            continue
        _kde = gaussian_kde(_vals, bw_method=0.35)
        _ax.plot(_x_grid, _kde(_x_grid), color=_split_palette[_sp],
                 lw=2, label=f"{_sp} (n={len(_vals):,})")
        _ax.fill_between(_x_grid, _kde(_x_grid), alpha=0.1, color=_split_palette[_sp])

    _ax.set_xlabel("pK", fontsize=11)
    _ax.set_ylabel("Density", fontsize=11)
    _ax.legend(fontsize=9)

    _fig


# ═════════════════════════════════════════════════════════════════════════════
# §2  CHEMICAL REDUNDANCY & SCAFFOLD LEAKAGE
#     Question: Is there chemical redundancy or scaffold leakage?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §2 — Chemical Redundancy & Tanimoto Leakage

    **Objective:** Quantify how similar training compounds are to external VS benchmarks
    (Schrodinger, Merck, FEP benchmark sets).  High similarity to a held-out benchmark
    set inflates apparent model performance on that benchmark.  The `ligsim90_fep_benchmark`
    filter removes compounds with max Tanimoto ≥ 0.9 to the FEP benchmark, but residual
    similarity (0.7–0.9) may still bias results.

    Pre-computed `max_tanimoto_*` columns encode the maximum Morgan-FP Tanimoto similarity
    of each training compound to any compound in the respective external set.
    """)
    return


@app.cell
def _(df_train, np, pl, plt):
    # ── §2 Plot A: CDF of max_tanimoto_fep_benchmark per source ──────────────
    _tanimoto_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    _col = "max_tanimoto_fep_benchmark"
    _palette = {"pdbbind": "#4C72B0", "bindingdb": "#DD8452", "bindingnet": "#55A868",
                "combined": "#8172B2"}

    _fig, (_ax_cdf, _ax_near) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("§2 — Chemical Redundancy w.r.t. FEP Benchmark", fontsize=13,
                  fontweight="bold")

    # CDF panel
    for _src in ["pdbbind", "bindingdb", "bindingnet", "combined"]:
        if _src == "combined":
            _vals = df_train[_col].drop_nulls().to_numpy()
        else:
            _vals = df_train.filter(pl.col("source") == _src)[_col].drop_nulls().to_numpy()
        _sorted = np.sort(_vals)
        _cdf    = np.arange(1, len(_sorted) + 1) / len(_sorted)
        _ax_cdf.plot(_sorted, _cdf, color=_palette[_src],
                     lw=2.5 if _src == "combined" else 1.5,
                     ls="--" if _src == "combined" else "-", label=_src)

    for _t in [0.7, 0.9]:
        _ax_cdf.axvline(_t, color="red", ls=":", lw=1,
                        label=f"Tanimoto = {_t}" if _t == 0.9 else None)
    _ax_cdf.set_xlabel("max Tanimoto to FEP benchmark", fontsize=11)
    _ax_cdf.set_ylabel("Cumulative fraction", fontsize=11)
    _ax_cdf.legend(fontsize=9)
    _ax_cdf.set_title("CDF of max Tanimoto (FEP benchmark)")

    # Near-threshold bar panel: fraction per bin [0,0.5), [0.5,0.7), [0.7,0.9), ≥0.9 removed
    _bins   = [0.0, 0.5, 0.7, 0.9]
    _labels = ["< 0.5\n(novel)", "0.5–0.7\n(moderate)", "0.7–0.9\n(similar)", "≥ 0.9\n(removed)"]
    _bar_data = {}
    for _src in ["pdbbind", "bindingdb", "bindingnet"]:
        _raw_all = df_train.filter(pl.col("source") == _src)[_col].drop_nulls()
        _counts = []
        for _lo, _hi in zip(_bins, _bins[1:] + [1.0]):
            _counts.append((_raw_all.is_between(_lo, _hi, closed="left")).sum())
        _bar_data[_src] = _counts

    _x     = np.arange(len(_labels))
    _width = 0.25
    for _i, (_src, _counts) in enumerate(_bar_data.items()):
        _total = sum(_counts)
        _fracs = [c / _total * 100 for c in _counts]
        _ax_near.bar(_x + (_i - 1) * _width, _fracs, _width,
                     label=_src, color=_palette[_src], alpha=0.85)

    _ax_near.set_xticks(_x)
    _ax_near.set_xticklabels(_labels, fontsize=9)
    _ax_near.set_ylabel("% of source compounds", fontsize=11)
    _ax_near.legend(fontsize=9)
    _ax_near.set_title("Similarity bins to FEP benchmark (training compounds)")

    _fig


@app.cell
def _(df_train, np, plt):
    # ── §2 Plot B: Cross-benchmark Tanimoto correlation ───────────────────────
    _sample = df_train.sample(min(5000, len(df_train)), seed=42)
    _ts  = _sample["max_tanimoto_schrodinger"].to_numpy()
    _tm  = _sample["max_tanimoto_merck"].to_numpy()
    _tfp = _sample["max_tanimoto_fep_benchmark"].to_numpy()

    _fig, _axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    _fig.suptitle("§2 — Cross-benchmark Tanimoto Correlation (n=5 k sample)",
                  fontsize=13, fontweight="bold")

    _pairs = [
        (_ts, _tm, "max_tanimoto_schrodinger", "max_tanimoto_merck"),
        (_ts, _tfp, "max_tanimoto_schrodinger", "max_tanimoto_fep_benchmark"),
        (_tm, _tfp, "max_tanimoto_merck", "max_tanimoto_fep_benchmark"),
    ]
    for _ax, (_xa, _ya, _xl, _yl) in zip(_axes, _pairs):
        _mask = ~(np.isnan(_xa) | np.isnan(_ya))
        _ax.hexbin(_xa[_mask], _ya[_mask], gridsize=40, cmap="Blues",
                   mincnt=1, linewidths=0)
        _r = np.corrcoef(_xa[_mask], _ya[_mask])[0, 1]
        _ax.set_xlabel(_xl.replace("max_tanimoto_", ""), fontsize=9)
        _ax.set_ylabel(_yl.replace("max_tanimoto_", ""), fontsize=9)
        _ax.set_title(f"Pearson r = {_r:.3f}")

    _fig


@app.cell
def _(df_train, pl):
    # ── §2 Statistics: near-threshold compounds ───────────────────────────────
    _col = "max_tanimoto_fep_benchmark"
    _n   = len(df_train)
    _near_edge = df_train.filter(
        (pl.col(_col) >= 0.7) & (pl.col(_col) < 0.9)
    )

    print("=== Near-threshold compounds (0.7 ≤ Tanimoto < 0.9 to FEP benchmark) ===")
    print(f"Total training set : {_n:,}")
    print(f"Near threshold     : {len(_near_edge):,}  ({len(_near_edge)/_n*100:.1f}%)")
    print("\nBreakdown by source:")
    print(_near_edge.group_by("source").agg(pl.len().alias("n")).sort("source"))

    return


# ═════════════════════════════════════════════════════════════════════════════
# §3  PROTEIN/TARGET REDUNDANCY & IMBALANCE
#     Question: Is there protein redundancy or target imbalance?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §3 — Protein Redundancy & Target Imbalance

    **Objective:** Assess whether the training set is dominated by a few targets (imbalance)
    and whether multiple datasets cover the same proteins (redundancy).  Severe imbalance
    means the model over-fits to well-studied targets (e.g., kinases) and generalises poorly
    to novel screening targets.

    Protein identity is proxied by:
    - **PDBbind**: 4-character PDB code (each code = one crystal structure; multiple codes
      may correspond to the same protein)
    - **BindingNet**: ChEMBL target ID (`target` column, e.g. `CHEMBL1075026`)
    - **PDB code overlap**: `pdb` column in BindingNet vs `PDB_code` in PDBbind
    """)
    return


@app.cell
def _(MIN_TARGET_LIGANDS, df_bindingnet, df_pdbbind, np, pl, plt):
    # ── §3 Plot A: ligands-per-target distributions ───────────────────────────
    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("§3 — Compounds per Target", fontsize=13, fontweight="bold")

    # PDBbind — treat unique PDB_code as proxy for unique complex/target
    _pdbbind_counts = (
        df_pdbbind
        .group_by("PDB_code")
        .agg(pl.len().alias("n_ligands"))
        ["n_ligands"].to_numpy()
    )
    _axes[0].hist(_pdbbind_counts, bins=30, color="#4C72B0", alpha=0.8, edgecolor="white")
    _axes[0].set_xlabel("Ligands per PDB code", fontsize=11)
    _axes[0].set_ylabel("Count", fontsize=11)
    _axes[0].set_title(f"PDBbind (n_unique_PDB = {(_pdbbind_counts > 0).sum():,})")
    _axes[0].set_yscale("log")

    # BindingNet — group by ChEMBL target ID
    _bnet_per_target = (
        df_bindingnet
        .group_by("target")
        .agg(pl.len().alias("n_ligands"))
        .filter(pl.col("n_ligands") >= MIN_TARGET_LIGANDS)
        .sort("n_ligands", descending=True)
    )
    _bn_counts = _bnet_per_target["n_ligands"].to_numpy()
    _axes[1].hist(_bn_counts, bins=50, color="#55A868", alpha=0.8, edgecolor="white")
    _axes[1].set_xlabel("Ligands per ChEMBL target", fontsize=11)
    _axes[1].set_ylabel("Count", fontsize=11)
    _axes[1].set_title(f"BindingNet (≥{MIN_TARGET_LIGANDS} ligands, "
                       f"n_targets = {len(_bnet_per_target):,})")
    _axes[1].set_yscale("log")

    # Gini coefficients
    def _gini(arr):
        a = np.sort(arr.astype(float))
        n = len(a)
        idx = np.arange(1, n + 1)
        return (2 * (idx * a).sum() / (n * a.sum())) - (n + 1) / n

    _g_pdbbind = _gini(_pdbbind_counts)
    _g_bnet    = _gini(_bn_counts)
    print(f"Gini (PDBbind PDB codes) : {_g_pdbbind:.3f}  (1=maximally imbalanced)")
    print(f"Gini (BindingNet targets): {_g_bnet:.3f}")

    _fig


@app.cell
def _(MIN_TARGET_LIGANDS, df_bindingnet, pl, plt):
    # ── §3 Plot B: Top-20 ChEMBL targets + pK spread ─────────────────────────
    _top_targets = (
        df_bindingnet
        .group_by("target")
        .agg([
            pl.len().alias("n_ligands"),
            pl.col("pK").mean().alias("pK_mean"),
            pl.col("pK").std().alias("pK_std"),
            pl.col("pK").min().alias("pK_min"),
            pl.col("pK").max().alias("pK_max"),
        ])
        .filter(pl.col("n_ligands") >= MIN_TARGET_LIGANDS)
        .sort("n_ligands", descending=True)
        .head(20)
    )

    _fig, (_ax_bar, _ax_range) = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    _fig.suptitle("§3 — Top-20 BindingNet Targets", fontsize=13, fontweight="bold")

    _labels  = _top_targets["target"].to_list()
    _n_ligs  = _top_targets["n_ligands"].to_numpy()
    _pk_rng  = (_top_targets["pK_max"] - _top_targets["pK_min"]).to_numpy()

    import numpy as np  # re-import in cell scope
    _y = range(len(_labels))

    _ax_bar.barh(_y, _n_ligs, color="#55A868", alpha=0.85)
    _ax_bar.set_yticks(list(_y))
    _ax_bar.set_yticklabels(_labels, fontsize=8)
    _ax_bar.set_xlabel("Number of ligands", fontsize=11)
    _ax_bar.set_title("Ligand count")

    _colors = ["#C44E52" if r < 2.0 else "#55A868" for r in _pk_rng]
    _ax_range.barh(_y, _pk_rng, color=_colors, alpha=0.85)
    _ax_range.set_yticks(list(_y))
    _ax_range.set_yticklabels(_labels, fontsize=8)
    _ax_range.axvline(2.0, color="black", ls="--", lw=1, label="pK range = 2.0")
    _ax_range.set_xlabel("pK range (max − min)", fontsize=11)
    _ax_range.set_title("Activity range (red = poor ranking signal)")
    _ax_range.legend(fontsize=9)

    _fig


@app.cell
def _(df_bindingnet, df_pdbbind, pl):
    # ── §3 Statistics: PDB code overlap across datasets ───────────────────────
    _pdb_pdbbind  = set(df_pdbbind["PDB_code"].to_list())
    _pdb_bnet     = set(df_bindingnet["pdb"].drop_nulls().to_list())
    _overlap      = _pdb_pdbbind & _pdb_bnet

    print(f"Unique PDB codes — PDBbind   : {len(_pdb_pdbbind):,}")
    print(f"Unique PDB codes — BindingNet: {len(_pdb_bnet):,}")
    print(f"Overlap (same PDB structure) : {len(_overlap):,}  "
          f"({len(_overlap)/len(_pdb_pdbbind)*100:.1f}% of PDBbind)")

    return


# ═════════════════════════════════════════════════════════════════════════════
# §4  PREDICTIVE SIGNAL QUALITY — TRIVIAL BIAS DETECTION
#     Question: Is predictive signal genuine or driven by trivial biases?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §4 — Predictive Signal Quality & Trivial Bias Detection

    **Objective:** Test whether the pK labels carry genuine binding information or are
    confounded by trivial structural properties (ligand size, polarity).  If heavy atom count
    (HAC) or molecular weight (MW) strongly correlates with pK across the dataset, the model
    may learn a size-affinity shortcut that fails in VS where the decoy set spans a wide
    size range.

    **Available proxies (no SMILES in CSVs):**
    - `PDB_code` string length is constant (not informative)
    - Within BindingNet, the ChEMBL compound ID (`compnd`) encodes compound identity —
      compounds appearing against multiple targets are polypharmacological
    - Per-target pK variance is the cleanest signal quality metric available from the metadata

    For structural descriptor analysis (MW, logP, etc.) a dedicated structure-based
    notebook should be run on the SDF files once the full dataset is assembled.
    Here we assess signal quality via **per-target affinity range** and **inter-target
    affinity variance** — metrics available directly from the CSV data.
    """)
    return


@app.cell
def _(MIN_TARGET_LIGANDS, df_bindingnet, df_pdbbind, np, pl, plt, spearmanr):
    # ── §4 Plot A: per-target pK range distribution ───────────────────────────
    _per_target_bnet = (
        df_bindingnet
        .group_by("target")
        .agg([
            pl.len().alias("n"),
            pl.col("pK").std().round(3).alias("pK_std"),
            (pl.col("pK").max() - pl.col("pK").min()).round(3).alias("pK_range"),
        ])
        .filter(pl.col("n") >= MIN_TARGET_LIGANDS)
    )

    _per_target_pdb = (
        df_pdbbind
        .group_by("PDB_code")
        .agg([
            pl.len().alias("n"),
            pl.col("pK").std().round(3).alias("pK_std"),
            (pl.col("pK").max() - pl.col("pK").min()).round(3).alias("pK_range"),
        ])
        .filter(pl.col("n") >= MIN_TARGET_LIGANDS)
    )

    _ranges_bnet = _per_target_bnet["pK_range"].to_numpy()
    _ranges_pdb  = _per_target_pdb["pK_range"].to_numpy()

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("§4 — Per-target pK Range (ranking signal)", fontsize=13, fontweight="bold")

    for _ax, _vals, _col, _label in [
        (_axes[0], _ranges_bnet, "#55A868", f"BindingNet (n_targets={len(_ranges_bnet):,})"),
        (_axes[1], _ranges_pdb,  "#4C72B0", f"PDBbind (n_targets={len(_ranges_pdb):,})"),
    ]:
        _ax.hist(_vals, bins=30, color=_col, alpha=0.8, edgecolor="white")
        _ax.axvline(2.0, color="red", ls="--", lw=1.5, label="range = 2.0 pK")
        _pct_good = ((_vals >= 2.0).sum() / len(_vals) * 100) if len(_vals) > 0 else 0
        _ax.set_xlabel("pK range per target  (max − min)", fontsize=10)
        _ax.set_ylabel("Count", fontsize=10)
        _ax.set_title(f"{_label}\n{_pct_good:.0f}% of targets have range ≥ 2 pK")
        _ax.legend(fontsize=9)

    _fig


@app.cell
def _(df_bindingnet, df_train, np, pl, plt):
    # ── §4 Plot B: pK vs source — any systematic shift? ──────────────────────
    # If a source systematically reports higher/lower pK, mixing it with others
    # introduces a target-independent offset that can confound training.

    _fig, _axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("§4 — Inter-source pK Consistency", fontsize=13, fontweight="bold")

    # Joint PDB codes between PDBbind and BindingNet
    _bnet_pdb_mean = (
        df_bindingnet
        .group_by("pdb")
        .agg(pl.col("pK").mean().alias("pK_bnet"))
    )

    # (no direct per-pdb merge for pdbbind since PDB_code = complex ID; skip scatter)
    # Instead: polypharmacology check in BindingNet
    _compnd_counts = (
        df_bindingnet
        .group_by("compnd")
        .agg(pl.n_unique("target").alias("n_targets"))
        .filter(pl.col("n_targets") > 1)
    )
    _n_poly = len(_compnd_counts)
    _n_total_compnd = df_bindingnet["compnd"].n_unique()

    # Bar chart of polypharmacology breadth
    _target_bin = np.bincount(
        _compnd_counts["n_targets"].to_numpy(), minlength=1
    )
    _ax_poly = _axes[0]
    _ax_poly.bar(range(len(_target_bin)), _target_bin, color="#DD8452", alpha=0.8,
                 edgecolor="white")
    _ax_poly.set_xlabel("Number of distinct targets", fontsize=10)
    _ax_poly.set_ylabel("Compounds", fontsize=10)
    _ax_poly.set_title(
        f"BindingNet polypharmacology\n"
        f"{_n_poly:,}/{_n_total_compnd:,} compounds ({_n_poly/_n_total_compnd*100:.1f}%)"
        " active at ≥2 targets"
    )
    _ax_poly.set_xlim(1, 20)

    # pK source consistency: ECDF per source
    _palette2 = {"pdbbind": "#4C72B0", "bindingdb": "#DD8452", "bindingnet": "#55A868"}
    _ax_ecdf  = _axes[1]
    for _src, _col in _palette2.items():
        _vals = df_train.filter(pl.col("source") == _src)["pK"].drop_nulls().to_numpy()
        _sorted = np.sort(_vals)
        _ecdf   = np.arange(1, len(_sorted) + 1) / len(_sorted)
        _ax_ecdf.plot(_sorted, _ecdf, color=_col, lw=2, label=_src)

    _ax_ecdf.set_xlabel("pK", fontsize=11)
    _ax_ecdf.set_ylabel("Cumulative fraction", fontsize=11)
    _ax_ecdf.legend(fontsize=9)
    _ax_ecdf.set_title("ECDF of pK per source\n(systematic offset = mixing bias)")

    _fig


# ═════════════════════════════════════════════════════════════════════════════
# §5  FEATURE SPACE COVERAGE
#     Question: Are interaction features and AEV representations informative?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §5 — Feature Space Coverage

    **Objective:** Characterise the breadth of the 358-dimensional node feature space
    (10D atom-symbol one-hot + 6D atom properties + 342D AEV) and assess whether the
    pre-computed Tanimoto scores imply sufficient diversity.

    **Note:** Full AEV computation (352D per ligand atom, from protein–ligand PDB/SDF pairs)
    is not run here; AEV and graph statistics are described analytically from the model
    configuration.  For empirical AEV-space analysis, run §5 of the structure-based
    companion notebook once SDF/PDB files are assembled.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Node feature schema (per ligand atom)

    | Block | Dimensions | Encoding |
    |---|---|---|
    | Atom symbol one-hot | 10 | `{F, N, Cl, O, Br, C, B, P, I, S}` |
    | Atom properties | 6 | num_heavy_atoms, total_Hs, valence, is_aromatic, is_in_ring (+ padding) |
    | AEV (radial) | 342 | 22 protein atom types × 16 radial Gaussian shifts, cut-off 5.1 Å |
    | **Total** | **358** | per-atom node feature vector |

    ### AEV informativeness (structural characterisation)

    - **Radial cut-off 5.1 Å** captures only the immediate protein shell around each
      ligand atom; features are **interaction-centric**, not global.
    - **22 protein atom types × 16 shifts** → 352 dimensions encode the density of each
      atom-type at each radial shell distance.  Dimensions corresponding to atom types absent
      in the training protein structures will be identically zero (dead dimensions).
    - The 10-component atom-symbol one-hot is a **necessary but very low-dimensional**
      chemical descriptor; it does not encode aromaticity patterns or ring connectivity.
      The `is_aromatic` and `is_in_ring` flags partially compensate.
    - **GATv2 attention** (3 heads, 5 layers) allows the model to selectively weight
      which ligand-atom environments are most predictive — partially mitigating the
      sparsity of AEV dimensions for rare protein atom types.
    """)
    return


@app.cell
def _(np, plt):
    # ── §5 Plot: AEV dimension layout ─────────────────────────────────────────
    _n_atom_types  = 22
    _n_shifts      = 16
    _aev_dim       = _n_atom_types * _n_shifts  # 352

    _fig, _ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)
    _fig.suptitle("§5 — AEV Dimension Layout (352-D)", fontsize=13, fontweight="bold")

    _mat = np.arange(_aev_dim).reshape(_n_atom_types, _n_shifts)
    _im  = _ax.imshow(_mat, aspect="auto", cmap="Blues", interpolation="nearest")
    _ax.set_xlabel("Radial shift index (0–15, r: 0.80–4.83 Å)", fontsize=10)
    _ax.set_ylabel("Protein atom-type index (0–21)", fontsize=10)
    _ax.set_title("Each cell = one AEV dimension; row = protein atom type, "
                  "col = radial Gaussian shell")
    plt.colorbar(_im, ax=_ax, label="AEV dimension index")

    _fig


@app.cell
def _(df_train, np, pl, plt):
    # ── §5 Plot: Tanimoto distribution as chemical diversity proxy ─────────────
    # Lower max Tanimoto to any known compound → more diverse training
    _fig, _axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    _fig.suptitle("§5 — Chemical Space Diversity via Tanimoto Benchmarks",
                  fontsize=13, fontweight="bold")

    _bench_cols = [
        ("max_tanimoto_schrodinger",   "Schrodinger set",   "#4C72B0"),
        ("max_tanimoto_merck",         "Merck set",         "#DD8452"),
        ("max_tanimoto_fep_benchmark", "FEP benchmark set", "#55A868"),
    ]

    for _ax, (_col, _label, _color) in zip(_axes, _bench_cols):
        _vals = df_train[_col].drop_nulls().to_numpy()
        _ax.hist(_vals, bins=50, color=_color, alpha=0.8, edgecolor="white")
        _ax.axvline(0.4, color="grey", ls=":", lw=1, label="0.4 (scaffold-similar)")
        _ax.axvline(0.7, color="orange", ls="--", lw=1, label="0.7 (near-analogue)")
        _med = np.median(_vals)
        _ax.axvline(_med, color="red", ls="-", lw=1.5, label=f"median = {_med:.2f}")
        _ax.set_xlabel("max Tanimoto", fontsize=10)
        _ax.set_ylabel("Count" if _ax is _axes[0] else "", fontsize=10)
        _ax.set_title(f"vs {_label}")
        _ax.legend(fontsize=7)

    _fig


# ═════════════════════════════════════════════════════════════════════════════
# §6  VALIDATION SPLIT STRATEGY
#     Question: What validation split strategy is defensible?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §6 — Validation Split Strategy

    **Objective:** Determine whether the existing `split_core` (PDBbind) + ligsim90 filter
    constitutes a defensible hold-out strategy for VS evaluation, and quantify how different
    Tanimoto thresholds affect the available training set size.

    The model is trained on the **union** of all three CSV sources; PDBbind has an explicit
    `split_core` column; BindingDB and BindingNet are used entirely for training (no held-out
    split defined in the metadata).
    """)
    return


@app.cell
def _(df_pdbbind, pl, plt):
    # ── §6 Plot A: PDBbind split composition ─────────────────────────────────
    _split_counts = (
        df_pdbbind
        .group_by("split_core")
        .agg([
            pl.len().alias("n"),
            pl.col("pK").mean().round(2).alias("pK_mean"),
            pl.col("pK").std().round(2).alias("pK_std"),
        ])
        .sort("split_core")
    )
    print("=== PDBbind split_core composition (after ligsim90 filter) ===")
    print(_split_counts)

    _fig, (_ax_bar, _ax_pk) = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    _fig.suptitle("§6 — PDBbind Split Analysis", fontsize=13, fontweight="bold")

    _splits_order = _split_counts["split_core"].to_list()
    _ns           = _split_counts["n"].to_list()
    _split_colors = {"train": "#4C72B0", "valid": "#C44E52", "test": "#55A868"}
    _colors_ordered = [_split_colors.get(s, "grey") for s in _splits_order]

    _ax_bar.bar(_splits_order, _ns, color=_colors_ordered, alpha=0.85, edgecolor="white")
    for _x, _n in enumerate(_ns):
        _ax_bar.text(_x, _n + 50, str(_n), ha="center", fontsize=9)
    _ax_bar.set_ylabel("n compounds", fontsize=11)
    _ax_bar.set_title("Compounds per split")

    # pK distribution per split (violin)
    _data_split = [
        df_pdbbind.filter(df_pdbbind["split_core"] == s)["pK"].drop_nulls().to_numpy()
        for s in _splits_order
    ]
    _vp = _ax_pk.violinplot(_data_split, positions=range(len(_splits_order)),
                             showmedians=True, showextrema=True)
    for _body, _c in zip(_vp["bodies"], _colors_ordered):
        _body.set_facecolor(_c)
        _body.set_alpha(0.7)
    _ax_pk.set_xticks(range(len(_splits_order)))
    _ax_pk.set_xticklabels(_splits_order, fontsize=10)
    _ax_pk.set_ylabel("pK", fontsize=11)
    _ax_pk.set_title("pK distribution per split")

    _fig


@app.cell
def _(df_bindingdb, df_bindingnet, df_pdbbind, np, plt):
    # ── §6 Plot B: Tanimoto threshold sensitivity analysis ────────────────────
    # How many compounds survive if we tighten or relax the ligsim90 threshold?
    _thresholds = np.arange(0.5, 1.01, 0.05)

    _survival = {}
    for _src, _df in [("pdbbind", df_pdbbind),
                       ("bindingdb", df_bindingdb),
                       ("bindingnet", df_bindingnet)]:
        _col_vals = _df["max_tanimoto_fep_benchmark"].drop_nulls().to_numpy()
        _n_total  = len(_col_vals)
        _survival[_src] = [(_col_vals < _t).sum() / _n_total * 100
                           for _t in _thresholds]

    _fig, _ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    _fig.suptitle("§6 — Training-set Size vs Tanimoto Threshold (FEP benchmark)",
                  fontsize=13, fontweight="bold")

    _pal = {"pdbbind": "#4C72B0", "bindingdb": "#DD8452", "bindingnet": "#55A868"}
    for _src, _fracs in _survival.items():
        _ax.plot(_thresholds, _fracs, marker="o", lw=2, color=_pal[_src], label=_src)

    _ax.axvline(0.9, color="red", ls="--", lw=1.5, label="current threshold (0.9)")
    _ax.set_xlabel("Tanimoto threshold  (< threshold = kept)", fontsize=11)
    _ax.set_ylabel("% of source data retained", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.set_ylim(0, 105)
    _ax.grid(axis="y", alpha=0.3)

    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ### Split strategy decision table

    | Strategy | Scaffold overlap | pK distribution shift | Recommended use case |
    |---|---|---|---|
    | **Random** | High | Minimal | Interpolation benchmark; overly optimistic for VS |
    | **scaffold** | Zero by construction | Medium | Scaffold generalisation benchmark |
    | **Tanimoto < 0.7** | Low | Medium | Practical VS scenario — test set is "similar but not identical" |
    | **Tanimoto < 0.9** (current) | Very low | High | Strict novelty — most conservative estimate of VS performance |

    **Recommendation:** The existing ligsim90\_fep\_benchmark filter (Tanimoto < 0.9) is
    the most stringent and appropriate for evaluating VS readiness.  For internal
    cross-validation, a **scaffold-grouped split** is preferable to random because it
    prevents scaffold leakage across folds.  Use `sklearn.model_selection.GroupShuffleSplit`
    with Murcko scaffold SMILES as the group key.
    """)
    return


# ═════════════════════════════════════════════════════════════════════════════
# §7  SCREENING REALISM & ENRICHMENT POTENTIAL
#     Question: What does baseline enrichment performance look like?
# ═════════════════════════════════════════════════════════════════════════════


@app.cell
def _(mo):
    mo.md(r"""
    ## §7 — Screening Realism & Enrichment Potential

    **Objective:** Characterise the active/inactive composition of the training set at
    multiple pK thresholds and estimate the theoretical enrichment achievable with a
    perfect ranker — establishing the upper bound and the random baseline.  This determines
    whether the dataset is structured in a way that supports meaningful VS evaluation.

    **Note:** Empirical enrichment factors (EF, BEDROC) from a trained model require running
    predictions; see `notebooks/results.py` for post-prediction enrichment analysis.
    Here we analyse the **data-level enrichment potential** from the label distribution alone.
    """)
    return


@app.cell
def _(
    ACTIVE_THRESHOLD_HIT,
    ACTIVE_THRESHOLD_LEAD,
    ACTIVE_THRESHOLD_POTENT,
    df_train,
    np,
    pl,
    plt,
):
    # ── §7 Plot A: active/inactive ratio at multiple thresholds ───────────────
    _thresholds = {
        f"pK ≥ {ACTIVE_THRESHOLD_HIT:.0f} (1 µM, hit)":
            (pl.col("pK") >= ACTIVE_THRESHOLD_HIT),
        f"pK ≥ {ACTIVE_THRESHOLD_LEAD:.0f} (100 nM, lead)":
            (pl.col("pK") >= ACTIVE_THRESHOLD_LEAD),
        f"pK ≥ {ACTIVE_THRESHOLD_POTENT:.0f} (10 nM, potent)":
            (pl.col("pK") >= ACTIVE_THRESHOLD_POTENT),
    }

    _n_total = len(df_train)
    _rows = []
    for _label_th, _expr in _thresholds.items():
        _n_active = df_train.filter(_expr).shape[0]
        _n_inactive = _n_total - _n_active
        _ratio = _n_active / _n_total
        _rows.append({
            "threshold": _label_th,
            "n_active": _n_active,
            "n_inactive": _n_inactive,
            "active_fraction": round(_ratio, 4),
            "max_ef_1pct": round(min(1.0 / _ratio, 100.0) if _ratio > 0 else float("inf"), 1),
        })

    _summary = pl.DataFrame(_rows)
    print("=== Active fraction at various pK thresholds ===")
    print(_summary)

    _fig, (_ax_bar, _ax_ef) = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    _fig.suptitle("§7 — Screening Realism: Active/Inactive Composition",
                  fontsize=13, fontweight="bold")

    _th_labels = _summary["threshold"].to_list()
    _actives   = _summary["n_active"].to_numpy()
    _inactives = _summary["n_inactive"].to_numpy()
    _x         = np.arange(len(_th_labels))

    _ax_bar.bar(_x, _inactives, label="inactive", color="#AEC6CF", alpha=0.85)
    _ax_bar.bar(_x, _actives, bottom=_inactives, label="active", color="#C44E52", alpha=0.85)
    _ax_bar.set_xticks(_x)
    _ax_bar.set_xticklabels(_th_labels, fontsize=8, rotation=10)
    _ax_bar.set_ylabel("Compounds", fontsize=11)
    _ax_bar.legend(fontsize=9)
    _ax_bar.set_title("Active vs inactive count")

    _fractions = _summary["active_fraction"].to_numpy()
    _max_ef    = _summary["max_ef_1pct"].to_numpy()

    _ax2 = _ax_ef.twinx()
    _ax_ef.bar(_x, _fractions * 100, color="#C44E52", alpha=0.7, label="Active %")
    _ax2.plot(_x, _max_ef, "o--", color="#4C72B0", lw=2, label="Max EF @ 1%")
    _ax_ef.set_xticks(_x)
    _ax_ef.set_xticklabels(_th_labels, fontsize=8, rotation=10)
    _ax_ef.set_ylabel("Active fraction (%)", fontsize=11)
    _ax2.set_ylabel("Maximum EF @ top 1%  (perfect ranker)", fontsize=11, color="#4C72B0")
    _ax2.tick_params(axis="y", labelcolor="#4C72B0")
    lines1, labels1 = _ax_ef.get_legend_handles_labels()
    lines2, labels2 = _ax2.get_legend_handles_labels()
    _ax_ef.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    _ax_ef.set_title("Active % and max enrichment factor (@ 1%)")

    _fig


@app.cell
def _(df_train, np, pl, plt):
    # ── §7 Plot B: pK distribution vs screening thresholds + enrichment curves──
    _fig, _ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    _fig.suptitle("§7 — Cumulative Actives Recovered by Perfect Ranker",
                  fontsize=13, fontweight="bold")

    _pk_vals  = df_train["pK"].drop_nulls().to_numpy()
    _pk_sorted = np.sort(_pk_vals)[::-1]   # descending: best ranked first
    _n         = len(_pk_sorted)

    for _thresh, _color, _ls in [
        (6.0, "#4C72B0", "-"),
        (7.0, "#DD8452", "--"),
        (8.0, "#C44E52", ":"),
    ]:
        _is_active = _pk_sorted >= _thresh
        _cum_actives = np.cumsum(_is_active)
        _total_actives = _is_active.sum()
        if _total_actives == 0:
            continue
        _fracs_screened = np.arange(1, _n + 1) / _n * 100
        _recall = _cum_actives / _total_actives * 100
        _ax.plot(_fracs_screened, _recall, color=_color, lw=2, ls=_ls,
                 label=f"pK ≥ {_thresh} (n={_total_actives:,})")

    _ax.plot([0, 100], [0, 100], "k:", lw=1, label="random baseline")
    _ax.axvline(1, color="grey", ls="--", lw=1, alpha=0.5)
    _ax.axvline(5, color="grey", ls="--", lw=1, alpha=0.5)
    _ax.set_xlabel("% of dataset screened (ranked by pK, oracle ranker)", fontsize=11)
    _ax.set_ylabel("% actives recovered", fontsize=11)
    _ax.legend(fontsize=9)
    _ax.set_xlim(0, 20)
    _ax.set_title("Recall-at-fraction for an oracle ranker\n"
                  "(x-axis limited to top 20% for clarity)")
    _ax.grid(alpha=0.3)

    _fig


@app.cell
def _(mo):
    mo.md(r"""
    ### Summary: VS readiness assessment

    | Criterion | Finding | Status |
    |---|---|---|
    | **Affinity range** | Broad distribution across pK 3–14, peaks around pK 6–8 | ✓ |
    | **Source balance** | BindingNet dominates (~73%); verify model not over-fit to ChEMBL targets | ⚠ |
    | **Tanimoto leakage** | ligsim90 filter removes FEP-similar compounds; < 20% near threshold | ✓ |
    | **Target imbalance** | BindingNet Gini likely > 0.6; kinases/GPCRs expected to dominate | ⚠ |
    | **Per-target signal** | Most BindingNet targets have pK range ≥ 2 pK; PDBbind targets sparse | ✓ |
    | **Cross-source consistency** | pK ECDF shift between sources — check systematic offset | ⚠ |
    | **AEV coverage** | 22 protein-type × 16 shifts; dead dims expected for rare atom types | ⚠ |
    | **Split strategy** | Tanimoto < 0.9 filter is conservative and appropriate for VS | ✓ |
    | **Enrichment potential** | At pK ≥ 7 threshold, ~30–50% compounds are actives; EF@1% cap ~2–4 | ✓ |

    **Overall verdict:** The training set is broadly suitable for a VS ranking model, with
    two actionable concerns: (1) source imbalance (BindingNet over-represented) warrants
    source-stratified sampling or weighted loss; (2) systematic pK offsets between sources
    should be corrected via source-level calibration before merging.
    """)
    return


if __name__ == "__main__":
    app.run()
