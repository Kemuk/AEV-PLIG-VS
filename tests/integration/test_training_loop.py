"""
Integration tests for training loop.

Tests: model + dataset -> Trainer -> training step + validation
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
import os

from torch_geometric.loader import DataLoader

from aev_plig.models import get_model
from aev_plig.datasets import GraphDataset, init_weights
from aev_plig.training import Trainer, rmse, mse, pearson, spearman, concordance_index


@pytest.mark.integration
class TestTrainingLoop:
    """Test training loop integration."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def train_dataset(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Create training dataset."""
        ids = list(sample_graphs_dict.keys())
        return GraphDataset(
            root=temp_data_dir,
            dataset='train',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    @pytest.fixture
    def valid_dataset(self, sample_graphs_dict, sample_labels, train_dataset, temp_data_dir):
        """Create validation dataset with same scaler."""
        ids = list(sample_graphs_dict.keys())
        return GraphDataset(
            root=temp_data_dir,
            dataset='valid',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=train_dataset.y_scaler
        )

    @pytest.fixture
    def trainer(self, mock_config, train_dataset, valid_dataset, device):
        """Create a Trainer instance."""
        model = get_model(
            'GATv2Net',
            node_feature_dim=train_dataset.num_node_features,
            edge_feature_dim=train_dataset.num_edge_features,
            config=mock_config
        )
        model.apply(init_weights)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

        return Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            y_scaler=train_dataset.y_scaler,
            learning_rate=0.001
        )


class TestMetrics:
    """Test metric functions."""

    def test_rmse_known_values(self):
        """Test RMSE with known values."""
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0, 3.0])
        assert rmse(y, f) == 0.0

        y = np.array([0.0, 0.0])
        f = np.array([1.0, 1.0])
        assert rmse(y, f) == 1.0

    def test_mse_known_values(self):
        """Test MSE with known values."""
        y = np.array([1.0, 2.0, 3.0])
        f = np.array([1.0, 2.0, 3.0])
        assert mse(y, f) == 0.0

        y = np.array([0.0, 0.0])
        f = np.array([1.0, 1.0])
        assert mse(y, f) == 1.0

    def test_pearson_perfect_correlation(self):
        """Test Pearson with perfect correlation."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(pearson(y, f), 1.0)

        # Perfect negative correlation
        f = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert np.isclose(pearson(y, f), -1.0)

    def test_pearson_no_correlation(self):
        """Test Pearson with uncorrelated data."""
        y = np.array([1.0, 2.0, 3.0, 4.0])
        f = np.array([1.0, 1.0, 1.0, 1.0])  # Constant
        # Pearson is undefined for constant arrays (NaN)
        result = pearson(y, f)
        assert np.isnan(result)

    def test_spearman_perfect_rank_correlation(self):
        """Test Spearman with perfect rank correlation."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([10.0, 20.0, 30.0, 40.0, 50.0])  # Same ranking
        assert np.isclose(spearman(y, f), 1.0)

    def test_concordance_index_perfect(self):
        """Test C-index with perfect prediction."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert np.isclose(concordance_index(y, f), 1.0)

    def test_concordance_index_inverse(self):
        """Test C-index with inverse prediction."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        f = np.array([5.0, 4.0, 3.0, 2.0, 1.0])
        assert np.isclose(concordance_index(y, f), 0.0)

    def test_concordance_index_random(self):
        """Test C-index is between 0 and 1."""
        np.random.seed(42)
        y = np.random.rand(100)
        f = np.random.rand(100)
        ci = concordance_index(y, f)
        assert 0.0 <= ci <= 1.0


@pytest.mark.integration
class TestTrainer:
    """Test Trainer class."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    @pytest.fixture
    def train_dataset(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Create training dataset."""
        ids = list(sample_graphs_dict.keys())
        return GraphDataset(
            root=temp_data_dir,
            dataset='train_trainer',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    @pytest.fixture
    def trainer(self, mock_config, train_dataset, device):
        """Create a Trainer instance."""
        model = get_model(
            'GATv2Net',
            node_feature_dim=train_dataset.num_node_features,
            edge_feature_dim=train_dataset.num_edge_features,
            config=mock_config
        )
        model.apply(init_weights)

        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
        valid_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)  # Same for simplicity

        return Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            y_scaler=train_dataset.y_scaler,
            learning_rate=0.001
        )

    def test_trainer_initialization(self, trainer):
        """Test Trainer initializes correctly."""
        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.loss_fn is not None
        assert trainer.y_scaler is not None
        assert trainer.best_pc == -1.1
        assert trainer.pcs == []

    def test_train_epoch_returns_loss(self, trainer):
        """Test train_epoch returns a loss value."""
        loss = trainer.train_epoch(epoch=1, log_interval=1000)

        assert isinstance(loss, float)
        assert loss >= 0

    def test_validate_returns_predictions(self, trainer):
        """Test validate returns true values and predictions."""
        y_true, y_pred = trainer.validate()

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)

    def test_fit_creates_checkpoint(self, trainer, temp_data_dir):
        """Test fit saves model checkpoint."""
        model_path = os.path.join(temp_data_dir, 'test_model.pt')

        history = trainer.fit(n_epochs=2, model_save_path=model_path)

        assert os.path.exists(model_path)
        assert 'train_loss' in history
        assert 'val_pc' in history
        assert 'val_rmse' in history
        assert len(history['train_loss']) == 2

    def test_predict_returns_values(self, trainer, train_dataset):
        """Test predict returns values."""
        test_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)

        y_true, y_pred = trainer.predict(test_loader)

        assert isinstance(y_true, np.ndarray)
        assert isinstance(y_pred, np.ndarray)
        assert len(y_true) == len(y_pred)
