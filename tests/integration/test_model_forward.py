"""
Integration tests for model forward pass.

Tests the path: dataset batch -> model -> output
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from aev_plig.models import (
    get_model, list_models, MODEL_REGISTRY, BaseGATv2Net, GATv2Net,
    GATv2NetAleatoric, GATv2NetBayesian, GATv2NetMixedPrecision,
    GATv2NetBayesianMixedPrecision,
)
from aev_plig.datasets import GraphDataset, init_weights


@pytest.mark.integration
class TestModelForward:
    """Test model forward pass."""

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
            dataset='test_model',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    def test_model_registry_contains_gatv2net(self):
        """Test that GATv2Net is in the model registry."""
        assert 'GATv2Net' in MODEL_REGISTRY
        assert MODEL_REGISTRY['GATv2Net'] == GATv2Net

    def test_list_models(self):
        """Test list_models function."""
        models = list_models()
        assert isinstance(models, list)
        assert 'GATv2Net' in models

    def test_get_model(self, mock_config, node_feature_dim, edge_feature_dim):
        """Test get_model factory function."""
        model = get_model(
            'GATv2Net',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config
        )

        assert isinstance(model, GATv2Net)

    def test_get_model_invalid_name(self, mock_config, node_feature_dim, edge_feature_dim):
        """Test get_model with invalid model name."""
        with pytest.raises(KeyError):
            get_model(
                'InvalidModel',
                node_feature_dim=node_feature_dim,
                edge_feature_dim=edge_feature_dim,
                config=mock_config
            )

    def test_model_forward_output_shape(self, untrained_model, test_dataset, device):
        """Test model forward pass returns correct shape."""
        untrained_model.to(device)
        untrained_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        with torch.no_grad():
            output = untrained_model(batch)

        # Output should be [batch_size, 1]
        assert output.shape == (1, 1)

    def test_model_forward_no_nan(self, untrained_model, test_dataset, device):
        """Test model forward pass produces no NaN values."""
        untrained_model.to(device)
        untrained_model.apply(init_weights)
        untrained_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        with torch.no_grad():
            output = untrained_model(batch)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_forward_differentiable(self, untrained_model, test_dataset, device):
        """Test model forward pass is differentiable."""
        untrained_model.to(device)
        untrained_model.train()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        output = untrained_model(batch)
        loss = output.mean()
        loss.backward()

        # Check that gradients exist
        has_grad = False
        for param in untrained_model.parameters():
            if param.grad is not None:
                has_grad = True
                break

        assert has_grad, "Model should have gradients after backward pass"

    def test_model_batch_processing(self, mock_config, device):
        """Test model handles batch of multiple graphs."""
        # Create synthetic batch
        node_dim = 100
        edge_dim = 4
        batch_size = 3

        model = get_model(
            'GATv2Net',
            node_feature_dim=node_dim,
            edge_feature_dim=edge_dim,
            config=mock_config
        )
        model.to(device)
        model.eval()

        # Create batch of graphs
        data_list = []
        for i in range(batch_size):
            num_nodes = 10 + i * 2
            num_edges = 20 + i * 4

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
            output = model(batch)

        assert output.shape == (batch_size, 1)

    def test_model_deterministic_with_seed(self, mock_config, node_feature_dim, edge_feature_dim, test_dataset, device):
        """Test model produces deterministic output with same seed."""
        torch.manual_seed(42)
        model1 = get_model(
            'GATv2Net',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config
        )
        model1.to(device)
        model1.eval()

        torch.manual_seed(42)
        model2 = get_model(
            'GATv2Net',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config
        )
        model2.to(device)
        model2.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader))
        batch = batch.to(device)

        with torch.no_grad():
            output1 = model1(batch)
            output2 = model2(batch)

        torch.testing.assert_close(output1, output2)


@pytest.mark.integration
class TestMixedPrecisionModelForward:
    """Test mixed precision model forward pass (CPU-safe)."""

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
            dataset='test_mp_model',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    def test_base_class_is_abstract(self):
        """Test that BaseGATv2Net cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseGATv2Net(node_feature_dim=25, edge_feature_dim=4)

    def test_all_models_inherit_base(self):
        """Test all model classes inherit from BaseGATv2Net."""
        assert issubclass(GATv2Net, BaseGATv2Net)
        assert issubclass(GATv2NetAleatoric, BaseGATv2Net)
        assert issubclass(GATv2NetBayesian, BaseGATv2Net)
        assert issubclass(GATv2NetMixedPrecision, BaseGATv2Net)
        assert issubclass(GATv2NetBayesianMixedPrecision, BaseGATv2Net)

    def test_mp_models_share_base_forward(self):
        """Test MP models share forward with their parent."""
        assert GATv2NetMixedPrecision.forward is GATv2Net.forward
        assert GATv2NetBayesianMixedPrecision.forward is GATv2NetBayesian.forward

    def test_mp_models_override_mlp_forward(self):
        """Test MP models override _mlp_forward to force fp32 in MLP layers."""
        assert GATv2NetMixedPrecision._mlp_forward is not GATv2Net._mlp_forward
        assert GATv2NetBayesianMixedPrecision._mlp_forward is not GATv2NetBayesian._mlp_forward

    def test_mp_model_forward_on_cpu(self, mock_config, node_feature_dim, edge_feature_dim, test_dataset, device):
        """Test mixed precision model forward pass works on CPU."""
        model = get_model(
            'GATv2NetMixedPrecision',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config
        )
        model.to(device)
        model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            output = model(batch)

        assert output.shape == (1, 1)
        assert not torch.isnan(output).any()

    def test_mp_bayesian_model_forward_on_cpu(self, mock_config, node_feature_dim, edge_feature_dim, test_dataset, device):
        """Test mixed precision VBLL Bayesian model forward pass on CPU."""
        model = get_model(
            'GATv2NetBayesianMixedPrecision',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config,
            dataset_size=100,
        )
        model.to(device)
        model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            output = model(batch)

        # GATv2NetBayesianMixedPrecision is now VBLL: returns VBLLReturn, not a tuple
        assert not isinstance(output, tuple), "VBLL output should not be a tuple"
        assert output.predictive.mean.shape == (1, 1)
        assert output.predictive.variance.shape == (1, 1)
        assert (output.predictive.variance > 0).all()
