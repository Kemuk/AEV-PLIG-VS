from pathlib import Path

import lightgbm as lgb
import numpy as np

from .config import RankConfig


class LambdaMARTModel:
    def __init__(self, config: RankConfig | None = None):
        self._cfg = config or RankConfig()
        self._booster: lgb.Booster | None = None

    def fit(self, X_train, y_train, groups_train,
            X_val=None, y_val=None, groups_val=None) -> None:
        params = dict(
            objective="lambdarank", metric="ndcg",
            eval_at=self._cfg.NDCG_EVAL_AT,
            num_leaves=self._cfg.NUM_LEAVES,
            min_child_samples=self._cfg.MIN_CHILD_SAMPLES,
            learning_rate=self._cfg.LEARNING_RATE,
            n_estimators=self._cfg.N_ESTIMATORS,
            verbose=50,
        )
        train_ds = lgb.Dataset(X_train, label=y_train, group=groups_train)
        valid_sets = ([lgb.Dataset(X_val, label=y_val, group=groups_val)]
                      if X_val is not None else [])
        callbacks = ([lgb.early_stopping(self._cfg.EARLY_STOPPING_ROUNDS, verbose=False)]
                     if valid_sets else [])
        self._booster = lgb.train(params, train_ds,
                                  valid_sets=valid_sets or None,
                                  callbacks=callbacks)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._booster.predict(X)

    def save(self, path: str | Path) -> None:
        Path(path).mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(Path(path) / "model.lgb"))

    @classmethod
    def load(cls, path: str | Path) -> "LambdaMARTModel":
        obj = cls()
        obj._booster = lgb.Booster(model_file=str(Path(path) / "model.lgb"))
        return obj
