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
    record_id, mol_path, protein_path, featuriser = args
    mol = _read_molecule(mol_path)
    if mol is None:
        return record_id, None
    return record_id, featuriser.featurise(mol, protein_path)


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
                    f"{bindingnet_root}/{p}/target_{t}/{c}/rec_addcharge_pocket_6A.mol2"
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

        fingerprints_by_id = _compute_fingerprints_for_df(
            pl.concat([self._actives_df, negative_pool_df]).unique("unique_id"),
            self._featuriser,
            self._n_jobs,
            self._cache_dir,
        )

        for split in self._actives_df["split"].unique():

            active_records = self._actives_df.filter(
                (pl.col("split") == split)
                & pl.col("unique_id").is_in(list(fingerprints_by_id))
            )

            current_pool_df = (
                negative_pool_df.filter(pl.col("split") == split)
                if self._pool_df is None else negative_pool_df
            )

            valid_pool_df = current_pool_df.filter(
                pl.col("unique_id").is_in(list(fingerprints_by_id))
            )

            pool_record_ids = valid_pool_df["unique_id"].to_list()
            pool_feature_matrix = np.vstack([
                fingerprints_by_id[rid] for rid in pool_record_ids
            ])
            pool_target_ids = np.array(valid_pool_df["target_id"].to_list())

            active_record_ids = active_records["unique_id"].to_list()
            active_target_ids = active_records["target_id"].to_list()
            active_feature_matrix = np.vstack([
                fingerprints_by_id[rid] for rid in active_record_ids
            ])

            eligible_pool_indices_by_target = {
                t: np.where(pool_target_ids != t)[0]
                for t in np.unique(active_target_ids)
            }

            print(f"  Building {split} queries ({len(active_record_ids)} actives, {len(pool_record_ids)} pool)...")

            feature_blocks, label_blocks, group_sizes, group_target_ids = [], [], [], []

            for i, (record_id, target_id) in enumerate(tqdm(
                zip(active_record_ids, active_target_ids),
                total=len(active_record_ids),
                desc=f"  {split}",
                unit="query",
                leave=False,
                file=sys.stdout,
            )):
                negative_indices = self._neg_gen.sample(
                    eligible_pool_indices_by_target[target_id],
                    random_generator,
                )

                if len(negative_indices) == 0:
                    continue

                feature_blocks.append(
                    np.vstack([
                        active_feature_matrix[i:i+1],
                        pool_feature_matrix[negative_indices],
                    ])
                )

                label_blocks.append(
                    np.array([1] + [0] * len(negative_indices), dtype=np.uint8)
                )

                group_sizes.append(1 + len(negative_indices))
                group_target_ids.append(target_id)

            if feature_blocks:
                self._data[split] = (
                    np.vstack(feature_blocks),
                    np.concatenate(label_blocks),
                    np.array(group_sizes),
                    group_target_ids,
                )

    def get_arrays(self, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if split not in self._data:
            raise KeyError(f"Split '{split}' not prepared or contains no data.")
        return self._data[split][:3]

    def target_ids(self, split: str) -> list[str]:
        return self._data[split][3]