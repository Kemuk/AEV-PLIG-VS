from pathlib import Path

from .config import RankConfig
from .dataset import RankDataset, load_records
from .featurisers import ECFP4Featuriser, LigandFeaturiser
from .model import LambdaMARTModel
from .negatives import RandomNegativeGenerator


def train_rank(
    sources: list[str] = ("pdbbind", "bindingnet"),
    data_dir: str = "data",
    featuriser: LigandFeaturiser | None = None,
    config: RankConfig | None = None,
    output_dir: str | Path | None = None,
) -> tuple[LambdaMARTModel, RankDataset]:
    cfg = config or RankConfig()
    feat = featuriser or ECFP4Featuriser(cfg.ECFP4_RADIUS, cfg.ECFP4_NBITS)

    df = load_records(sources, data_dir)
    dataset = RankDataset(df, feat, RandomNegativeGenerator(cfg.N_NEGATIVES), cfg.NEGATIVE_SEED)
    dataset.prepare()

    X_tr, y_tr, g_tr = dataset.get_arrays("train")
    X_va, y_va, g_va = dataset.get_arrays("valid")

    model = LambdaMARTModel(cfg)
    model.fit(X_tr, y_tr, g_tr, X_va, y_va, g_va)

    if output_dir:
        model.save(output_dir)

    return model, dataset
