from abc import ABC, abstractmethod

import numpy as np


class NegativeGenerator(ABC):
    @abstractmethod
    def sample(self, eligible: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Return integer indices drawn from pre-filtered eligible pool indices."""


class RandomNegativeGenerator(NegativeGenerator):
    def __init__(self, n: int = 50):
        self.n = n

    def sample(self, eligible, rng) -> np.ndarray:
        n = min(self.n, len(eligible))
        return rng.choice(eligible, size=n, replace=False) if n > 0 else np.empty(0, int)
