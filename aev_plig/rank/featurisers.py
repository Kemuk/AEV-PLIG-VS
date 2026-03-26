from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator


class LigandFeaturiser(ABC):
    @abstractmethod
    def featurise(self, mol, protein_path: Path | None = None) -> np.ndarray: ...

    def featurise_batch(self, mols, protein_paths=None) -> np.ndarray:
        paths = protein_paths or [None] * len(mols)
        return np.array([self.featurise(m, p) for m, p in zip(mols, paths)])


class ECFP4Featuriser(LigandFeaturiser):
    def __init__(self, radius: int = 2, n_bits: int = 2048):
        self._gen = GetMorganGenerator(radius=radius, fpSize=n_bits)

    def featurise(self, mol, protein_path=None) -> np.ndarray:
        return self._gen.GetFingerprintAsNumPy(mol)
