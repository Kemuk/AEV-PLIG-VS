from abc import ABC, abstractmethod

import numpy as np
import polars as pl


class NegativeGenerator(ABC):
    @abstractmethod
    def generate(self, target_id: str, pool: pl.DataFrame,
                 rng: np.random.Generator) -> pl.DataFrame:
        """Return rows sampled from pool (already split-filtered), excluding target_id."""


class RandomNegativeGenerator(NegativeGenerator):
    def __init__(self, n: int = 50):
        self.n = n

    def generate(self, target_id, pool, rng):
        candidates = pool.filter(pl.col("target_id") != target_id)
        n = min(self.n, len(candidates))
        return candidates.sample(n=n, seed=int(rng.integers(0, 2**32)))
