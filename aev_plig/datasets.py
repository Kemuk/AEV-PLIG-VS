"""
PyTorch Geometric dataset utilities for protein-ligand graphs.
"""

import warnings

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch_geometric.data import Data
from tqdm import tqdm

warnings.simplefilter(action='ignore', category=FutureWarning)


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
