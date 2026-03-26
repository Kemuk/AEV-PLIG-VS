from pathlib import Path

from .config import RankConfig
from .dataset import RankDataset, load_records
from .featurisers import ECFP4Featuriser, LigandFeaturiser
from .model import LambdaMARTModel
from .negatives import RandomNegativeGenerator


def train_rank(
    sources: list[str] = ("pdbbind", "bindingnet"),
    pool_sources: list[str] | None = None,
    data_dir: str = "data",
    featuriser: LigandFeaturiser | None = None,
    config: RankConfig | None = None,
    output_dir: str | Path | None = None,
    n_jobs: int = -1,
    cache_dir: str | None = None,
) -> tuple[LambdaMARTModel, RankDataset]:
    cfg = config or RankConfig()
    feat = featuriser or ECFP4Featuriser(cfg.ECFP4_RADIUS, cfg.ECFP4_NBITS)

    print(f"[1/4] Loading records from {list(sources)}...")
    actives_df = load_records(sources, data_dir)
    pool_df = load_records(pool_sources, data_dir) if pool_sources else None
    print(f"      {len(actives_df)} actives loaded.")

    print("[2/4] Preparing dataset...")
    dataset = RankDataset(actives_df, feat, RandomNegativeGenerator(cfg.N_NEGATIVES),
                          cfg.NEGATIVE_SEED, pool_df=pool_df, n_jobs=n_jobs, cache_dir=cache_dir)
    dataset.prepare()

    X_tr, y_tr, g_tr = dataset.get_arrays("train")
    val = dataset._data.get("valid")
    X_va, y_va, g_va = (val[0], val[1], val[2]) if val is not None else (None, None, None)
    print(f"      Train: {len(g_tr)} queries | Valid: {len(g_va) if g_va is not None else 0} queries")

    print(f"[3/4] Training LambdaMART ({cfg.N_ESTIMATORS} trees, lr={cfg.LEARNING_RATE})...")
    model = LambdaMARTModel(cfg)
    model.fit(X_tr, y_tr, g_tr, X_va, y_va, g_va)

    if output_dir:
        print(f"[4/4] Saving model to {output_dir}...")
        model.save(output_dir)

    return model, dataset
