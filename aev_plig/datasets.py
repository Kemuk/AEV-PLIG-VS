"""
PyTorch Geometric dataset utilities for protein-ligand graphs.
"""

import json
import math
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import ConcatDataset, Sampler
from torch_geometric.data import Data
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def init_weights(layer):
    """
    Initialize weights for neural network layers.

    Uses Xavier normal initialization for weights and zeros for biases.

    Args:
        layer: PyTorch layer to initialize
    """
    if hasattr(layer, "weight") and "BatchNorm" not in str(layer):
        torch.nn.init.xavier_normal_(layer.weight)
    if hasattr(layer, "bias"):
        if layer.bias is not None:
            torch.nn.init.zeros_(layer.bias)


def create_dataset(ids, targets, graphs_dict, scale=False, y_scaler=None,
                   sdf_files=None, pdb_files=None):
    """
    Convert graph tuples into a list of PyG Data objects.

    Args:
        ids: List of graph IDs used to look up graph tuples.
        targets: List of labels or graph_ids to store in `Data.y`.
        graphs_dict: Mapping of unique_id -> (c_size, features, edge_index, edge_features).
        scale: If True, apply StandardScaler to targets.
        y_scaler: Optional pre-fit scaler for reuse across splits.
        sdf_files: Optional list of ligand file paths (stored as Data.sdf_file).
        pdb_files: Optional list of protein file paths (stored as Data.pdb_file).

    Returns:
        tuple[list[Data], StandardScaler | None]: data list and scaler.
    """
    if len(ids) != len(targets):
        raise ValueError("Number of datapoints and targets must be the same")

    y_values = np.asarray(targets).reshape(-1, 1)
    scaler = y_scaler

    if scale:
        if scaler is None:
            scaler = StandardScaler()
            scaler.fit(y_values)
        transformed = scaler.transform(y_values).flatten()
        y_out = transformed.tolist()
    else:
        y_out = targets

    data_list = []
    missing_count = 0

    for idx, unique_id in enumerate(tqdm(ids, desc="Creating graphs", unit="graphs")):
        graph_tuple = graphs_dict.get(unique_id)
        if graph_tuple is None:
            missing_count += 1
            continue

        _, features, edge_index, edge_features = graph_tuple
        value = y_out[idx]
        y_dtype = torch.float32 if scale else torch.int32

        data_point = Data(
            x=torch.tensor(np.array(features), dtype=torch.float32),
            edge_index=torch.tensor(np.array(edge_index), dtype=torch.long).T,
            edge_attr=torch.tensor(np.array(edge_features), dtype=torch.float32),
            y=torch.tensor([value], dtype=y_dtype),
        )
        data_point.unique_id = unique_id
        data_point.pK = float(targets[idx])
        if sdf_files is not None:
            data_point.sdf_file = sdf_files[idx]
        if pdb_files is not None:
            data_point.pdb_file = pdb_files[idx]
        data_list.append(data_point)

    if missing_count > 0:
        print(f"⚠️  {missing_count}/{len(ids)} graphs not found in graphs_dict")
    print(f"✓ Processed {len(data_list)}/{len(ids)} graphs")

    return data_list, scaler


def load_split(dataset_name, split):
    """Load split data from chunked manifest format, with flat-file fallback."""
    dataset_root = _PROJECT_ROOT / "data" / "processed" / dataset_name
    split_dir = dataset_root / split
    manifest_path = split_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        parts = [
            torch.load(split_dir / part_name, weights_only=False)
            for part_name in manifest["parts"]
        ]
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    legacy_path = _PROJECT_ROOT / "data" / "processed" / f"{dataset_name}_{split}.pt"
    if legacy_path.exists():
        return torch.load(legacy_path, weights_only=False)

    raise FileNotFoundError(
        f"No dataset artifacts found for split '{split}'. "
        f"Checked {manifest_path} and {legacy_path}."
    )


def get_target_labels(dataset):
    """
    Extract target labels from a dataset of PyG Data objects.

    Args:
        dataset: Iterable of PyG Data objects, each with a `.unique_id` attribute
                 (e.g. "3ao4").

    Returns:
        list[str]: Target label per complex.
    """
    return [data.unique_id for data in dataset]


class TargetAwareBatchSampler(Sampler):
    """
    Batch sampler that guarantees each batch contains multiple complexes per
    target, enabling pairwise ranking loss computation within each batch.

    Groups dataset indices by target label, then builds batches by sampling
    `complexes_per_target` indices from each of several targets until the
    batch is full. Targets with fewer than 2 complexes are excluded.

    Args:
        target_labels: One target label per dataset element.
        complexes_per_target: Number of complexes to sample per target in each batch.
        batch_size: Maximum batch size (will be rounded down to a multiple of
                    complexes_per_target).
        seed: Random seed for reproducibility.
    """

    def __init__(self, target_labels, complexes_per_target=4, batch_size=64, seed=42):
        self.complexes_per_target = complexes_per_target
        self.seed = seed

        # Group indices by target, keep only targets with ≥2 complexes
        target_to_indices = defaultdict(list)
        for idx, label in enumerate(target_labels):
            target_to_indices[label].append(idx)
        self.target_to_indices = {
            t: idxs for t, idxs in target_to_indices.items() if len(idxs) >= 2
        }
        self.targets = list(self.target_to_indices.keys())

        # How many targets fit in one batch
        self.targets_per_batch = max(1, batch_size // complexes_per_target)
        self.total = sum(len(v) for v in self.target_to_indices.values())

    def __iter__(self):
        rng = np.random.RandomState(self.seed)

        # Shuffle targets, cycle through them building batches
        targets = self.targets.copy()
        rng.shuffle(targets)

        # For each target, create a shuffled pool of indices
        pools = {}
        for t in targets:
            idxs = self.target_to_indices[t].copy()
            rng.shuffle(idxs)
            pools[t] = idxs

        target_idx = 0
        while target_idx < len(targets):
            batch = []
            batch_targets = targets[target_idx:target_idx + self.targets_per_batch]
            for t in batch_targets:
                pool = pools[t]
                n = min(self.complexes_per_target, len(pool))
                if n < 2:
                    continue
                # Sample without replacement from this target's pool
                sampled = pool[:n]
                pool[:n] = []  # remove used indices
                # Refill pool if exhausted
                if len(pool) < 2:
                    refill = self.target_to_indices[t].copy()
                    rng.shuffle(refill)
                    pools[t] = refill
                batch.extend(sampled)
            if batch:
                yield batch
            target_idx += self.targets_per_batch

    def __len__(self):
        return math.ceil(len(self.targets) / self.targets_per_batch)
