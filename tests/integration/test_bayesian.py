"""
Integration tests for the true Bayesian model (GATv2NetBayesian with VBLL).

GATv2NetBayesian uses a Variational Bayesian Last Layer (VBLL) to provide
both epistemic and aleatoric uncertainty in a single sampling-free pass.
Forward output is a VBLLReturn dataclass, not a (mean, var) tuple.
"""

import pytest
import torch
import tempfile
import shutil

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from aev_plig.models import get_model, GATv2NetBayesian
from aev_plig.datasets import GraphDataset, init_weights


@pytest.mark.integration
class TestBayesianModel:
    """Tests for GATv2NetBayesian (VBLL)."""

    @pytest.fixture
    def temp_data_dir(self):
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def test_dataset(self, sample_graphs_dict, sample_labels, temp_data_dir):
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
        return get_model(
            'GATv2NetBayesian',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config,
            dataset_size=100,
        )

    def test_bayesian_output_is_vbll_return(self, bayesian_model, test_dataset, device):
        """Test that GATv2NetBayesian returns a VBLLReturn dataclass, not a tuple."""
        bayesian_model.to(device)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            output = bayesian_model(batch)

        assert not isinstance(output, tuple), "VBLL output should not be a tuple"
        assert hasattr(output, 'predictive'), "VBLL output must have .predictive attribute"
        assert hasattr(output, 'train_loss_fn'), "VBLL output must have .train_loss_fn"

    def test_bayesian_predictive_mean_shape(self, bayesian_model, test_dataset, device):
        """Test that predictive.mean has shape (batch, 1)."""
        bayesian_model.to(device)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            output = bayesian_model(batch)

        assert output.predictive.mean.shape == (1, 1), (
            f"Expected shape (1, 1), got {output.predictive.mean.shape}"
        )
        assert output.predictive.variance.shape == (1, 1), (
            f"Expected shape (1, 1), got {output.predictive.variance.shape}"
        )

    def test_variance_positivity(self, bayesian_model, test_dataset, device):
        """Test that predictive.variance is always positive."""
        bayesian_model.to(device)
        bayesian_model.apply(init_weights)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                output = bayesian_model(batch)
                var = output.predictive.variance

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
            config=mock_config,
            dataset_size=500,
        )
        model.to(device)
        model.apply(init_weights)
        model.eval()

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
            batch = next(iter(loader)).to(device)

            with torch.no_grad():
                output = model(batch)

            var = output.predictive.variance
            assert (var > 0).all(), f"Seed {seed}: variance must be positive, min={var.min()}"

    def test_train_loss_fn(self, bayesian_model, test_dataset, device):
        """Test that train_loss_fn returns a scalar tensor."""
        bayesian_model.to(device)
        bayesian_model.train()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        output = bayesian_model(batch)
        target = batch.y.view(-1, 1).to(device)
        loss = output.train_loss_fn(target)

        assert loss.ndim == 0, "train_loss_fn should return a scalar tensor"
        assert not torch.isnan(loss), "train_loss_fn returned NaN"
        assert not torch.isinf(loss), "train_loss_fn returned Inf"

    def test_bayesian_model_in_registry(self):
        """Test that GATv2NetBayesian is in the model registry."""
        from aev_plig.models import MODEL_REGISTRY, list_models

        assert 'GATv2NetBayesian' in MODEL_REGISTRY
        assert MODEL_REGISTRY['GATv2NetBayesian'] is GATv2NetBayesian
        assert 'GATv2NetBayesian' in list_models()

    def test_bayesian_model_differentiable(self, bayesian_model, test_dataset, device):
        """Test that the VBLL loss is differentiable."""
        bayesian_model.to(device)
        bayesian_model.train()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        output = bayesian_model(batch)
        target = batch.y.view(-1, 1).to(device)
        loss = output.train_loss_fn(target)
        loss.backward()

        has_grad = any(p.grad is not None for p in bayesian_model.parameters())
        assert has_grad, "Model should have gradients after backward pass"

    def test_is_bayesian_flag(self, mock_config, node_feature_dim, edge_feature_dim):
        """Test that is_bayesian=True for VBLL model."""
        model = get_model(
            'GATv2NetBayesian',
            node_feature_dim=node_feature_dim,
            edge_feature_dim=edge_feature_dim,
            config=mock_config,
            dataset_size=100,
        )
        assert model.is_bayesian is True

    def test_predict_method_returns_tensor(self, bayesian_model, test_dataset, device):
        """Test that predict() returns a plain tensor (not VBLLReturn)."""
        bayesian_model.to(device)
        bayesian_model.eval()

        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
        batch = next(iter(loader)).to(device)

        with torch.no_grad():
            pred = bayesian_model.predict(batch)

        assert isinstance(pred, torch.Tensor), "predict() should return a tensor"
        assert pred.shape == (1, 1), f"Expected (1,1), got {pred.shape}"
