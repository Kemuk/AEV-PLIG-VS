"""
PyTorch Geometric dataset classes for protein-ligand graphs.

This module provides dataset classes for loading and processing molecular graphs
for use with PyTorch Geometric.

Optimizations:
- Parallel graph processing with multiprocessing
- Progress bars with tqdm
"""

import os
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
import torch
from sklearn.preprocessing import StandardScaler
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm


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


# =============================================================================
# Parallel processing helper functions (module-level for picklability)
# =============================================================================

def _process_single_graph_train(args):
    """
    Process a single graph for training (with labels).

    Args:
        args: Tuple of (label, graph_tuple) where graph_tuple is (c_size, features, edge_index, edge_features)
              OR None if graph not found

    Returns:
        Data object or None if graph not found
    """
    label, graph_tuple = args

    if graph_tuple is None:
        return None

    c_size, features, edge_index, edge_features = graph_tuple

    data_point = Data(
        x=torch.Tensor(np.array(features)),
        edge_index=torch.LongTensor(np.array(edge_index)).T,
        edge_attr=torch.Tensor(np.array(edge_features)),
        y=torch.FloatTensor(np.array([label]))
    )
    return data_point


def _process_single_graph_predict(args):
    """
    Process a single graph for prediction (with graph IDs).

    Args:
        args: Tuple of (graph_id, graph_tuple) where graph_tuple is (c_size, features, edge_index, edge_features)
              OR None if graph not found

    Returns:
        Data object or None if graph not found
    """
    graph_id, graph_tuple = args

    if graph_tuple is None:
        return None

    c_size, features, edge_index, edge_features = graph_tuple

    data_point = Data(
        x=torch.Tensor(np.array(features)),
        edge_index=torch.LongTensor(np.array(edge_index)).T,
        edge_attr=torch.Tensor(np.array(edge_features)),
        y=torch.IntTensor(np.array([graph_id]))
    )
    return data_point


# =============================================================================
# Dataset Classes
# =============================================================================

class GraphDataset(InMemoryDataset):
    """
    PyTorch Geometric dataset for training/validation/test sets with labels.

    Loads molecular graphs from a dictionary and creates PyTorch Data objects.
    Automatically applies StandardScaler normalization to target values (pK).

    Args:
        root: Root directory for data storage (default: 'data')
        dataset: Dataset name (used for file naming)
        ids: List of molecule IDs
        y: List of target values (binding affinities)
        graphs_dict: Dictionary mapping IDs to graph tuples
        y_scaler: Optional StandardScaler (if None, creates new one from training data)
    """

    def __init__(self, root='data', dataset=None,
                 ids=None, y=None, graphs_dict=None, y_scaler=None):

        super(GraphDataset, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])

        if os.path.isfile(self.processed_paths[0]):
            # Load existing processed data
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])
        else:
            # Process raw data and save
            self.process(ids, y, graphs_dict)
            self.load(self.processed_paths[0])

        # Apply StandardScaler normalization to target values
        if y_scaler is None:
            y_scaler = StandardScaler()
            y_scaler.fit(np.reshape(self._data.y, (self.__len__(), 1)))
        self.y_scaler = y_scaler
        self._data.y = [
            torch.tensor(element[0]).float()
            for element in self.y_scaler.transform(np.reshape(self._data.y, (self.__len__(), 1)))
        ]

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, y, graphs_dict):
        """
        Process molecular graphs and create PyTorch Data objects (parallelized).

        Args:
            ids: List of molecule IDs
            y: List of target values
            graphs_dict: Dictionary mapping IDs to (c_size, features, edge_index, edge_features)
        """
        assert (len(ids) == len(y)), 'Number of datapoints and labels must be the same'

        data_len = len(ids)

        # Limit workers to prevent memory exhaustion (each worker needs memory for graph processing)
        # Use at most 16 workers, or fewer if less CPUs available
        max_workers = min(os.cpu_count(), 16)
        print(f"Processing {data_len} graphs using {max_workers} threads...")

        # PRE-LOOKUP: Get graph data for each ID (or None if missing)
        # This way we only pass small tuples to workers, not the entire graphs_dict
        print("Looking up graph data...")
        args_list = [
            (y[i], graphs_dict.get(ids[i], None))
            for i in range(data_len)
        ]

        missing_count = sum(1 for _, g in args_list if g is None)
        if missing_count > 0:
            print(f"⚠️  {missing_count}/{data_len} graphs not found in graphs_dict")

        # Process graphs with threads (no IPC serialization overhead)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data_list = list(tqdm(
                executor.map(_process_single_graph_train, args_list, chunksize=1000),
                total=data_len,
                desc="Processing graphs",
                unit="graphs"
            ))

        # Filter out None values (skipped entries)
        original_count = len(data_list)
        data_list = [d for d in data_list if d is not None]
        skipped_count = original_count - len(data_list)

        if skipped_count > 0:
            print(f"✓ Processed {len(data_list)}/{data_len} graphs ({skipped_count} skipped)")
        else:
            print(f"✓ Processed all {len(data_list)} graphs successfully")

        print(f'Saving {len(data_list)} graphs to file...')
        self.save(data_list, self.processed_paths[0])


class GraphDatasetPredict(InMemoryDataset):
    """
    PyTorch Geometric dataset for prediction (no labels, uses graph IDs).

    Used for inference on new data without known binding affinities.
    Stores graph ID in the y field for tracking predictions.

    Args:
        root: Root directory for data storage (default: 'data')
        dataset: Dataset name (used for file naming)
        ids: List of molecule IDs
        graph_ids: List of graph IDs (for tracking predictions)
        graphs_dict: Dictionary mapping IDs to graph tuples
    """

    def __init__(self, root='data', dataset=None,
                 ids=None, graph_ids=None, graphs_dict=None):

        super(GraphDatasetPredict, self).__init__(root)
        self.dataset = dataset
        torch.serialization.add_safe_globals([Data])

        if os.path.isfile(self.processed_paths[0]):
            self.load(self.processed_paths[0])
            print("processed paths:")
            print(self.processed_paths[0])
        else:
            self.process(ids, graph_ids, graphs_dict)
            self.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, ids, graph_ids, graphs_dict):
        """
        Process molecular graphs for prediction (parallelized).

        Args:
            ids: List of molecule IDs
            graph_ids: List of graph IDs (stored in y field)
            graphs_dict: Dictionary mapping IDs to (c_size, features, edge_index, edge_features)
        """
        assert (len(ids) == len(graph_ids)), 'Number of datapoints and graph IDs must be the same'

        data_len = len(ids)

        # Limit workers to prevent memory exhaustion
        max_workers = min(os.cpu_count(), 16)
        print(f"Processing {data_len} graphs using {max_workers} threads...")

        # PRE-LOOKUP: Get graph data for each ID (or None if missing)
        print("Looking up graph data...")
        args_list = [
            (graph_ids[i], graphs_dict.get(ids[i], None))
            for i in range(data_len)
        ]

        missing_count = sum(1 for _, g in args_list if g is None)
        if missing_count > 0:
            print(f"⚠️  {missing_count}/{data_len} graphs not found in graphs_dict")

        # Process graphs with threads (no IPC serialization overhead)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            data_list = list(tqdm(
                executor.map(_process_single_graph_predict, args_list, chunksize=1000),
                total=data_len,
                desc="Processing graphs",
                unit="graphs"
            ))

        # Filter out None values (skipped entries)
        original_count = len(data_list)
        data_list = [d for d in data_list if d is not None]
        skipped_count = original_count - len(data_list)

        if skipped_count > 0:
            print(f"✓ Processed {len(data_list)}/{data_len} graphs ({skipped_count} skipped)")
        else:
            print(f"✓ Processed all {len(data_list)} graphs successfully")

        print(f'Saving {len(data_list)} graphs to file...')
        self.save(data_list, self.processed_paths[0])
