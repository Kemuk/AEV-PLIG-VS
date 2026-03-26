from abc import ABC, abstractmethod

import numpy as np


class NegativeGenerator(ABC):
    @abstractmethod
    def sample_indices(self, target_id: str, pool_target_ids: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
        """Return integer indices into the pool array, excluding target_id."""


class RandomNegativeGenerator(NegativeGenerator):
    def __init__(self, n: int = 50):
        self.n = n

    def sample_indices(self, target_id, pool_target_ids, rng) -> np.ndarray:
        eligible = np.where(pool_target_ids != target_id)[0]
        n = min(self.n, len(eligible))
        return rng.choice(eligible, size=n, replace=False) if n > 0 else np.empty(0, int)
