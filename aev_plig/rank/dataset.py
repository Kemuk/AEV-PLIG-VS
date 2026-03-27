from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
import sys

import numpy as np
import polars as pl
from rdkit import Chem, RDLogger
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map

from .featurisers import LigandFeaturiser
from .negatives import NegativeGenerator

RDLogger.DisableLog("rdApp.*")

TANIMOTO_MAX = 0.9


def _read_molecule(file_path: str):
    if file_path.endswith(".mol2"):
        return Chem.MolFromMol2File(file_path, sanitize=True)
    supplier = Chem.SDMolSupplier(file_path, sanitize=True, removeHs=False)
    return next(iter(supplier), None)


def _read_and_compute_features(args: tuple) -> tuple[str, np.ndarray | None]:
    uid, mol_path, protein_path, featuriser = args
    if not os.path.exists(mol_path):
        return uid, None
    mol = _read_molecule(mol_path)
    return uid, (featuriser.featurise(mol, protein_path) if mol is not None else None)


def _compute_fingerprints_for_df(
    records_df: pl.DataFrame,
    featuriser: LigandFeaturiser,
    n_jobs: int = -1,
    cache_dir: str | None = None,
) -> dict[str, np.ndarray]:

    n_workers = os.cpu_count() if n_jobs < 1 else n_jobs

    featurisation_tasks = [
        (row["unique_id"], row["mol_path"], row["protein_path"], featuriser)
        for row in records_df.select(
            ["unique_id", "mol_path", "protein_path"]
        ).iter_rows(named=True)
    ]

    print(f"  First 3 paths: {[t[1] for t in featurisation_tasks[:3]]}", flush=True)

    if cache_dir is not None:
        cache_hash = hashlib.md5(
            str(sorted(task[1] for task in featurisation_tasks)).encode()
        ).hexdigest()
        cache_path = Path(cache_dir) / f"fps_{cache_hash}.pkl"

        if cache_path.exists():
            print(f"  Loading {len(featurisation_tasks)} fingerprints from cache ({cache_path.name})...")
            with open(cache_path, "rb") as f:
                return pickle.load(f)

    print(f"  Featurising {len(featurisation_tasks)} molecules ({n_workers} workers)...")

    featurisation_results = thread_map(
        _read_and_compute_features,
        featurisation_tasks,
        total=len(featurisation_tasks),
        max_workers=n_workers,
        desc="  featurise",
        unit="mol",
    )

    fingerprints_by_id = {
        record_id: fp
        for record_id, fp in featurisation_results
        if fp is not None
    }

    if cache_dir is not None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(fingerprints_by_id, f)
        print(f"  Fingerprints cached to {cache_path.name}")

    return fingerprints_by_id


def load_records(
    sources: list[str] = ("pdbbind", "bindingnet", "bindingdb"),
    data_dir: str = "data",
) -> pl.DataFrame:

    source_frames = []

    if "pdbbind" in sources:
        records_df = pl.read_csv(f"{data_dir}/pdbbind_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )

        pdb_codes = records_df["PDB_code"].to_list()
        is_refined_flags = records_df["refined"].to_list()

        base_paths = [
            f"{data_dir}/pdbbind/{'refined' if is_refined else 'general'}-set/{code}/{code}"
            for code, is_refined in zip(pdb_codes, is_refined_flags)
        ]

        source_frames.append(
            pl.DataFrame({
                "unique_id": pdb_codes,
                "target_id": pdb_codes,
                "mol_path": [p + "_ligand.mol2" for p in base_paths],
                "protein_path": [p + "_protein.pdb" for p in base_paths],
                "pk": records_df["-logKd/Ki"],
                "split": records_df["split_core"],
            })
        )

    if "bindingnet" in sources:
        records_df = pl.read_csv(f"{data_dir}/bindingnet_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )

        bindingnet_root = f"{data_dir}/bindingnet/from_chembl_client"

        pdb_codes = records_df["pdb"].to_list()
        target_ids = records_df["target"].to_list()
        compound_ids = records_df["compnd"].to_list()

        source_frames.append(
            pl.DataFrame({
                "unique_id": records_df["unique_identify"],
                "target_id": target_ids,
                "mol_path": [
                    f"{bindingnet_root}/{p}/target_{t}/{c}/{p}_{t}_{c}.sdf"
                    for p, t, c in zip(pdb_codes, target_ids, compound_ids)
                ],
                "protein_path": [
                    f"{bindingnet_root}/{p}/rec_h_opt.pdb"
                    for p in pdb_codes
                ],
                "pk": records_df["-logAffi"],
                "split": pl.Series(["train"] * len(records_df)),
            })
        )

    if "bindingdb" in sources:
        records_df = pl.read_csv(f"{data_dir}/bindingdb_processed.csv").filter(
            pl.col("max_tanimoto_fep_benchmark") < TANIMOTO_MAX
        )

        bindingdb_root = f"{data_dir}/bindingdb/surflex"

        folders = records_df["folder"].to_list()
        mol2_files = records_df["mol2_file"].to_list()
        pdb_files = records_df["pdb_file"].to_list()

        source_frames.append(
            pl.DataFrame({
                "unique_id": records_df["unique_id"],
                "target_id": folders,
                "mol_path": [
                    f"{bindingdb_root}/{folder}/{mol}"
                    for folder, mol in zip(folders, mol2_files)
                ],
                "protein_path": [
                    f"{bindingdb_root}/{folder}/{pdb}"
                    for folder, pdb in zip(folders, pdb_files)
                ],
                "pk": records_df["pK"],
                "split": pl.Series(["train"] * len(records_df)),
            })
        )

    print(source_frames[0].row(0))

    return pl.concat(source_frames)


class RankDataset:

    def __init__(
        self,
        actives_df: pl.DataFrame,
        featuriser: LigandFeaturiser,
        neg_gen: NegativeGenerator,
        seed: int = 42,
        pool_df: pl.DataFrame | None = None,
        n_jobs: int = -1,
        cache_dir: str | None = None,
    ):
        self._actives_df = actives_df
        self._featuriser = featuriser
        self._neg_gen = neg_gen
        self._seed = seed
        self._pool_df = pool_df
        self._n_jobs = n_jobs
        self._cache_dir = cache_dir
        self._data: dict = {}

    def prepare(self) -> None:
        random_generator = np.random.default_rng(self._seed)
        negative_pool_df = self._pool_df if self._pool_df is not None else self._actives_df

        if self._pool_df is not None:
            featurise_df = pl.concat([self._actives_df, negative_pool_df]).unique("unique_id")
        else:
            featurise_df = self._actives_df

        fingerprints_by_id = _compute_fingerprints_for_df(
            featurise_df, self._featuriser, self._n_jobs, self._cache_dir,
        )
        valid_ids = set(fingerprints_by_id)

        actives_valid = self._actives_df.filter(pl.col("unique_id").is_in(valid_ids))
        pool_valid = negative_pool_df.filter(pl.col("unique_id").is_in(valid_ids))

        for split in actives_valid["split"].unique():
            split_actives = actives_valid.filter(pl.col("split") == split)
            split_pool = (
                pool_valid.filter(pl.col("split") == split)
                if self._pool_df is None else pool_valid
            )

            pool_ids = split_pool["unique_id"].to_list()
            pool_target_ids = split_pool["target_id"].to_numpy()
            pool_feature_matrix = np.vstack([fingerprints_by_id[rid] for rid in pool_ids])
            pool_idx_all = np.arange(len(pool_ids))

            active_ids = split_actives["unique_id"].to_list()
            active_target_ids = split_actives["target_id"].to_numpy()
            active_feature_matrix = np.vstack([fingerprints_by_id[rid] for rid in active_ids])

            # One rng.choice per unique target; broadcast sampled indices to all its actives
            n_neg = self._neg_gen.n
            neg_index_matrix = np.empty((len(active_ids), n_neg), dtype=np.intp)
            for t in np.unique(active_target_ids):
                mask = active_target_ids == t
                own = np.where(pool_target_ids == t)[0]
                eligible = np.setdiff1d(pool_idx_all, own, assume_unique=True)
                draws = random_generator.choice(eligible, size=(mask.sum(), n_neg), replace=False)
                neg_index_matrix[mask] = draws

            # (Q, 1+n_neg, F) → (Q*(1+n_neg), F)
            n_queries = len(active_ids)
            feature_matrix = np.concatenate(
                [active_feature_matrix[:, np.newaxis, :], pool_feature_matrix[neg_index_matrix]],
                axis=1,
            ).reshape(n_queries * (1 + n_neg), -1)

            labels = np.zeros(n_queries * (1 + n_neg), dtype=np.uint8)
            labels[:: 1 + n_neg] = 1

            self._data[split] = (
                feature_matrix,
                labels,
                np.full(n_queries, 1 + n_neg, dtype=np.intp),
                active_target_ids.tolist(),
            )

    def get_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split not in self._data:
            raise KeyError(f"Split '{split}' not prepared or contains no data.")
        return self._data[split][:3]

    def target_ids(self, split: str) -> list[str]:
        return self._data[split][3]