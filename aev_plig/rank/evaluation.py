from __future__ import annotations

import numpy as np
import polars as pl
from rdkit.ML.Scoring.Scoring import CalcBEDROC, CalcEnrichment
from sklearn.metrics import roc_auc_score

from .dataset import RankDataset
from .model import LambdaMARTModel


def per_target_enrichment(
    model: LambdaMARTModel,
    dataset: RankDataset,
    split: str = "test",
    fractions: tuple = (0.01, 0.05, 0.10),
    bedroc_alpha: float = 80.5,
) -> pl.DataFrame:
    """Per-query EF, BEDROC and AUROC. Returns a Polars DataFrame."""
    X, y, groups = dataset.get_arrays(split)
    scores = model.predict(X)
    tids = dataset.target_ids(split)

    rows, offset = [], 0
    for tid, size in zip(tids, groups):
        s, l = scores[offset:offset + size], y[offset:offset + size]
        offset += size
        if l.sum() == 0:
            continue
        order = np.argsort(-s)
        scored = [[float(s[i]), int(l[i])] for i in order]
        efs = CalcEnrichment(scored, 0, list(fractions))
        rows.append({
            "target_id": tid,
            "n_actives": int(l.sum()),
            "n_total":   int(size),
            **{f"ef_{int(f*100)}pct": efs[i] for i, f in enumerate(fractions)},
            "bedroc": CalcBEDROC(scored, 0, bedroc_alpha),
            "auroc":  float(roc_auc_score(l, s)),
        })
    return pl.DataFrame(rows)


def overall_enrichment(model: LambdaMARTModel, dataset: RankDataset,
                       split: str = "test") -> dict:
    df = per_target_enrichment(model, dataset, split)
    return {c: float(df[c].mean()) for c in df.columns if c != "target_id"}
