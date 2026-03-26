from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import polars as pl
from rdkit import Chem
from tqdm import tqdm

from .featurisers import LigandFeaturiser
from .negatives import NegativeGenerator

TANIMOTO_MAX = 0.9  # mirror existing pipeline filter


def _read_mol(path: str):
    if path.endswith(".mol2"):
        return Chem.MolFromMol2File(path, sanitize=True)
    suppl = Chem.SDMolSupplier(path, sanitize=True, removeHs=False)
    return next(iter(suppl), None)


def _read_and_featurise(args: tuple) -> tuple[str, np.ndarray | None]:
    uid, mol_path, protein_path, featuriser = args
    mol = _read_mol(mol_path)
    return uid, (featuriser.featurise(mol, protein_path) if mol is not None else None)


def _featurise_df(df: pl.DataFrame, featuriser: LigandFeaturiser,
                  n_jobs: int = -1) -> dict[str, np.ndarray]:
    """Read and featurise all rows in parallel. Returns {unique_id: fingerprint}."""
    n_workers = os.cpu_count() if n_jobs < 1 else n_jobs
    tasks = [(r["unique_id"], r["mol_path"], r["protein_path"], featuriser)
             for r in df.select(["unique_id", "mol_path", "protein_path"]).iter_rows(named=True)]
    print(f"  Featurising {len(tasks)} molecules ({n_workers} workers)...")
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        results = tqdm(ex.map(_read_and_featurise, tasks), total=len(tasks),
                       desc="  featurise", unit="mol", leave=False)
        return {uid: fp for uid, fp in results if fp is not None}


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

    pool_df: optional separate DataFrame for negatives. If None, actives are used as pool.
    n_jobs:  parallelism for mol reading + featurisation (-1 = all CPUs).
    """

    def __init__(self, actives_df: pl.DataFrame, featuriser: LigandFeaturiser,
                 neg_gen: NegativeGenerator, seed: int = 42,
                 pool_df: pl.DataFrame | None = None, n_jobs: int = -1):
        self._actives_df = actives_df
        self._featuriser = featuriser
        self._neg_gen = neg_gen
        self._seed = seed
        self._pool_df = pool_df
        self._n_jobs = n_jobs
        self._data: dict = {}  # split -> (X, y, group_sizes, target_ids)

    def prepare(self) -> None:
        rng = np.random.default_rng(self._seed)
        pool = self._pool_df if self._pool_df is not None else self._actives_df

        # Parallel featurisation of all unique mols (actives + pool)
        fps = _featurise_df(
            pl.concat([self._actives_df, pool]).unique("unique_id"),
            self._featuriser, self._n_jobs,
        )

        # Pre-build pool arrays once — O(1) lookup in inner loop
        pool_valid   = pool.filter(pl.col("unique_id").is_in(list(fps)))
        pool_uids    = pool_valid["unique_id"].to_list()
        pool_matrix  = np.vstack([fps[u] for u in pool_uids])       # (P, D)
        pool_targets = np.array(pool_valid["target_id"].to_list())   # (P,)

        for split in self._actives_df["split"].unique():
            actives = self._actives_df.filter(
                (pl.col("split") == split) & pl.col("unique_id").is_in(list(fps))
            )
            act_uids    = actives["unique_id"].to_list()
            act_targets = actives["target_id"].to_list()
            act_matrix  = np.vstack([fps[u] for u in act_uids])          # (A, D)
            eligible_cache = {t: np.where(pool_targets != t)[0]
                              for t in np.unique(act_targets)}            # K calls, not N
            print(f"  Building {split} queries ({len(act_uids)} actives, "
                  f"{len(eligible_cache)} unique targets)...")
            Xs, ys, sizes, tids = [], [], [], []
            for i, (uid, tid) in enumerate(tqdm(
                    zip(act_uids, act_targets), total=len(act_uids),
                    desc=f"  {split}", unit="query", leave=False)):
                idxs = self._neg_gen.sample(eligible_cache[tid], rng)
                if len(idxs) == 0:
                    continue
                Xs.append(np.vstack([act_matrix[i:i+1], pool_matrix[idxs]]))
                ys.append(np.array([1] + [0] * len(idxs), dtype=np.uint8))
                sizes.append(1 + len(idxs))
                tids.append(tid)
            if Xs:
                self._data[split] = np.vstack(Xs), np.concatenate(ys), np.array(sizes), tids

    def get_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return X, y, group_sizes for LightGBM."""
        if split not in self._data:
            raise KeyError(f"Split '{split}' not prepared or contains no data.")
        return self._data[split][:3]

    def target_ids(self, split: str) -> list[str]:
        return self._data[split][3]
