from __future__ import annotations

import numpy as np
import polars as pl
from rdkit import Chem

from .featurisers import LigandFeaturiser
from .negatives import NegativeGenerator

TANIMOTO_MAX = 0.9  # mirror existing pipeline filter


def _read_mol(path: str):
    """Read SDF or MOL2 via RDKit; return None on failure."""
    if path.endswith(".mol2"):
        return Chem.MolFromMol2File(path, sanitize=True)
    suppl = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
    return next(iter(suppl), None)


def load_records(
    sources: list[str] = ("pdbbind", "bindingnet", "bindingdb"),
    data_dir: str = "data",
) -> pl.DataFrame:
    """
    Load actives from the processed CSVs.

    Returns a DataFrame with columns:
        unique_id, target_id, mol_path, protein_path, pk, split

    sources: subset of {"pdbbind", "bindingnet", "bindingdb"}
    """
    frames = []

    if "pdbbind" in sources:
        df = pl.read_csv(f"{data_dir}/pdbbind_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )
        uid, refined = df["PDB_code"].to_list(), df["refined"].to_list()
        base = [f"{data_dir}/pdbbind/{'refined' if r else 'general'}-set/{u}/{u}"
                for u, r in zip(uid, refined)]
        frames.append(pl.DataFrame({
            "unique_id":    uid,
            "target_id":    uid,
            "mol_path":     [b + "_ligand.mol2" for b in base],
            "protein_path": [b + "_protein.pdb"  for b in base],
            "pk":           df["-logKd/Ki"],
            "split":        df["split_core"],
        }))

    if "bindingnet" in sources:
        df = pl.read_csv(f"{data_dir}/bindingnet_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )
        bn = f"{data_dir}/bindingnet/from_chembl_client"
        pdb, tgt, cmp = df["pdb"].to_list(), df["target"].to_list(), df["compnd"].to_list()
        frames.append(pl.DataFrame({
            "unique_id":    df["unique_identify"],
            "target_id":    tgt,
            "mol_path":     [f"{bn}/{p}/target_{t}/{c}/{p}_{t}_{c}.sdf" for p, t, c in zip(pdb, tgt, cmp)],
            "protein_path": [f"{bn}/{p}/rec_h_opt.pdb" for p in pdb],
            "pk":           df["-logAffi"],
            "split":        pl.Series(["train"] * len(df)),
        }))

    if "bindingdb" in sources:
        df = pl.read_csv(f"{data_dir}/bindingdb_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )
        bdb = f"{data_dir}/bindingdb/surflex"
        fld, mol2, pdb = df["folder"].to_list(), df["mol2_file"].to_list(), df["pdb_file"].to_list()
        frames.append(pl.DataFrame({
            "unique_id":    df["unique_id"],
            "target_id":    fld,
            "mol_path":     [f"{bdb}/{f}/{m}" for f, m in zip(fld, mol2)],
            "protein_path": [f"{bdb}/{f}/{p}" for f, p in zip(fld, pdb)],
            "pk":           df["pK"],
            "split":        pl.Series(["train"] * len(df)),
        }))

    return pl.concat(frames)


class RankDataset:
    """
    Loads actives, samples negatives per query, computes fingerprints.
    Call prepare() once before get_arrays().
    """

    def __init__(self, df: pl.DataFrame, featuriser: LigandFeaturiser,
                 neg_gen: NegativeGenerator, seed: int = 42):
        self._df = df
        self._featuriser = featuriser
        self._neg_gen = neg_gen
        self._seed = seed
        self._data: dict = {}  # split -> (X, y, group_sizes, target_ids)

    def prepare(self) -> None:
        rng = np.random.default_rng(self._seed)
        fps = {row["unique_id"]: self._featuriser.featurise(mol, row["protein_path"])
               for row in self._df.iter_rows(named=True)
               if (mol := _read_mol(row["mol_path"])) is not None}
        valid = self._df.filter(pl.col("unique_id").is_in(list(fps)))
        for split in valid["split"].unique():
            split_df = valid.filter(pl.col("split") == split)
            groups = [g for row in split_df.iter_rows(named=True)
                      if (g := self._make_group(row, split_df, fps, rng))]
            if groups:
                Xs, ys, sizes, tids = zip(*groups)
                self._data[split] = np.vstack(Xs), np.concatenate(ys), np.array(sizes), list(tids)

    def _make_group(self, row, split_df, fps, rng):
        negs = [r["unique_id"] for r in
                self._neg_gen.generate(row["target_id"], split_df, rng).iter_rows(named=True)
                if r["unique_id"] in fps]
        if not negs:
            return None
        return (np.vstack([fps[row["unique_id"]]] + [fps[n] for n in negs]),
                np.array([1] + [0] * len(negs), dtype=np.uint8),
                1 + len(negs), row["target_id"])

    def get_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X, y, group_sizes for LightGBM."""
        if split not in self._data:
            raise KeyError(f"Split '{split}' not prepared or contains no data.")
        return self._data[split][:3]

    def target_ids(self, split: str) -> list[str]:
        return self._data[split][3]
