"""
Integration tests for Bayesian model.

Minimal tests for GATv2NetBayesian:
- Output shape (mean, var tuple)
- Variance positivity
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from aev_plig.models import get_model, GATv2NetBayesian
from aev_plig.datasets import GraphDataset, init_weights


@pytest.mark.integration
class TestBayesianModel:
    """Minimal tests for Bayesian model."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_dataset(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Create a test dataset."""
        ids = list(sample_graphs_dict.keys())
        return GraphDataset(
            root=temp_data_dir,
            dataset='bayesian_test',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    @pytest.fixture
    def bayesian_model(self, mock_config, node_feature_dim, edge_feature_dim):
        """Create Bayesian model for testing."""
        return get_model(
            'GATv2NetBayesian',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config
        )

    def test_bayesian_output_shape(self, bayesian_model, test_dataset, device):
        """Test that Bayesian model returns (mean, var) tuple with correct shapes."""
        bayesian_model.to(device)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        with torch.no_grad():
            output = bayesian_model(batch)

        # Should return a tuple of (mean, var)
        assert isinstance(output, tuple), "Output should be a tuple"
        assert len(output) == 2, "Output tuple should have 2 elements (mean, var)"

        mean, var = output

        # Check shapes
        assert mean.shape == (1, 1), f"Mean shape should be (1, 1), got {mean.shape}"
        assert var.shape == (1, 1), f"Variance shape should be (1, 1), got {var.shape}"

    def test_variance_positivity(self, bayesian_model, test_dataset, device):
        """Test that variance is always positive."""
        bayesian_model.to(device)
        bayesian_model.apply(init_weights)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                mean, var = bayesian_model(batch)

                # Variance must be strictly positive
                assert (var > 0).all(), f"Variance must be positive, got {var}"
                assert not torch.isnan(var).any(), "Variance contains NaN"
                assert not torch.isinf(var).any(), "Variance contains Inf"

    def test_variance_positivity_multiple_inputs(self, mock_config, device):
        """Test variance positivity with varied synthetic inputs."""
        node_dim = 100
        edge_dim = 4
        batch_size = 5

        model = get_model(
            'GATv2NetBayesian',
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            config=mock_config
        )
        model.to(device)
        model.apply(init_weights)
        model.eval()

        # Test with multiple random graphs
        for seed in [42, 123, 456]:
            torch.manual_seed(seed)

            data_list = []
            for i in range(batch_size):
                num_nodes = 10 + i * 3
                num_edges = 20 + i * 6

                data = Data(
                    x=torch.randn(num_nodes, node_dim),
                    edge_index=torch.randint(0, num_nodes, (2, num_edges)),
                    edge_attr=torch.randn(num_edges, edge_dim),
                    y=torch.tensor([1.0])
                )
                data_list.append(data)

            loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
            batch = next(iter(loader))
            batch = batch.to(device)

            with torch.no_grad():
                mean, var = model(batch)

            # All variances must be positive
            assert (var > 0).all(), f"Seed {seed}: Variance must be positive, got min={var.min()}"

    def test_bayesian_model_in_registry(self):
        """Test that GATv2NetBayesian is in the model registry."""
        from aev_plig.models import MODEL_REGISTRY, list_models

        assert 'GATv2NetBayesian' in MODEL_REGISTRY
        assert 'GATv2NetBayesian' in list_models()

    def test_bayesian_model_differentiable(self, bayesian_model, test_dataset, device):
        """Test that both mean and variance are differentiable."""
        bayesian_model.to(device)
        bayesian_model.train()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        mean, var = bayesian_model(batch)

        # Both outputs should be differentiable
        loss = mean.mean() + var.mean()
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in bayesian_model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "Model should have gradients after backward pass"
