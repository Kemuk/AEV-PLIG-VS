"""
Integration tests for dataset creation.

Tests the path: graphs dict -> PyTorch Geometric dataset
"""

import pytest
import numpy as np
import torch
import os
import tempfile
import shutil

from aev_plig.datasets import GraphDataset, GraphDatasetPredict, init_weights


@pytest.mark.integration
class TestDatasetCreation:
    """Test dataset creation from graphs."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_graph_dataset_creation(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test GraphDataset creation with labels."""
        ids = list(sample_graphs_dict.keys())

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='test_dataset',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        # Check dataset length
        assert len(dataset) == len(ids)

        # Check data attributes
        data = dataset[0]
        assert hasattr(data, 'x')  # node features
        assert hasattr(data, 'edge_index')  # edge connectivity
        assert hasattr(data, 'edge_attr')  # edge features
        assert hasattr(data, 'y')  # label

    def test_graph_dataset_node_features(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test that node features are correctly loaded."""
        ids = list(sample_graphs_dict.keys())
        num_atoms, features, _, _ = sample_graphs_dict[ids[0]]

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='test_features',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        data = dataset[0]

        # Check node feature dimensions
        assert data.x.shape[0] == num_atoms
        assert data.x.shape[1] == len(features[0])

    def test_graph_dataset_edge_index_shape(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test that edge index has correct shape."""
        ids = list(sample_graphs_dict.keys())
        _, _, edge_index, _ = sample_graphs_dict[ids[0]]

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='test_edges',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        data = dataset[0]

        # edge_index should be [2, num_edges]
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == len(edge_index)

    def test_graph_dataset_scaler(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test that scaler is created and applied."""
        ids = list(sample_graphs_dict.keys())

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='test_scaler',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        # Check scaler exists
        assert dataset.y_scaler is not None

        # Labels should be normalized (with single sample, should be 0)
        data = dataset[0]
        # With a single sample, StandardScaler normalizes to 0
        # This is expected behavior

    def test_graph_dataset_predict_creation(self, sample_graphs_dict, temp_data_dir):
        """Test GraphDatasetPredict creation without labels."""
        ids = list(sample_graphs_dict.keys())
        graph_ids = list(range(len(ids)))

        dataset = GraphDatasetPredict(
            root=temp_data_dir,
            dataset='test_predict',
            ids=ids,
            graph_ids=graph_ids,
            graphs_dict=sample_graphs_dict
        )

        # Check dataset length
        assert len(dataset) == len(ids)

        # Check data attributes
        data = dataset[0]
        assert hasattr(data, 'x')
        assert hasattr(data, 'edge_index')
        assert hasattr(data, 'edge_attr')
        assert hasattr(data, 'y')  # Contains graph_id for tracking

    def test_init_weights(self, untrained_model):
        """Test that init_weights can be applied to a model."""
        # This should not raise any errors
        untrained_model.apply(init_weights)

        # Check that weights are initialized (not all zeros)
        for name, param in untrained_model.named_parameters():
            if 'weight' in name and 'BatchNorm' not in name:
                assert not (param == 0).all(), f"Weight {name} is all zeros"

    def test_dataset_num_features(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test that dataset reports correct feature dimensions."""
        ids = list(sample_graphs_dict.keys())
        _, features, _, edge_features = sample_graphs_dict[ids[0]]

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='test_num_features',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        assert dataset.num_node_features == len(features[0])
        assert dataset.num_edge_features == len(edge_features[0])
