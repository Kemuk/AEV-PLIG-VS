"""
Integration tests for ensemble prediction.

Tests: multiple model outputs -> ensemble averaging
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import shutil
import os

from aev_plig.models import get_model
from aev_plig.datasets import GraphDataset, init_weights
from aev_plig.training import Trainer
from torch_geometric.loader import DataLoader


@pytest.mark.integration
class TestEnsembleAggregation:
    """Test ensemble prediction aggregation."""

    def test_ensemble_averaging_basic(self):
        """Test basic ensemble averaging."""
        # 3 models, 2 samples each
        predictions = [
            [1.0, 2.0],
            [1.2, 2.1],
            [0.9, 1.9]
        ]

        ensemble_pred = np.mean(predictions, axis=0)

        assert ensemble_pred.shape == (2,)
        np.testing.assert_almost_equal(ensemble_pred[0], 1.033, decimal=2)
        np.testing.assert_almost_equal(ensemble_pred[1], 2.0, decimal=2)

    def test_ensemble_dataframe_aggregation(self):
        """Test ensemble averaging in DataFrame (as done in Predictor)."""
        df = pd.DataFrame({
            'graph_id': [0, 1, 2],
            'preds_0': [1.0, 2.0, 3.0],
            'preds_1': [1.1, 2.1, 3.1],
            'preds_2': [0.9, 1.9, 2.9]
        })

        pred_cols = [c for c in df.columns if c.startswith('preds_')]
        df['preds'] = df[pred_cols].mean(axis=1)

        expected = [1.0, 2.0, 3.0]
        np.testing.assert_array_almost_equal(df['preds'].values, expected, decimal=10)

    def test_ensemble_reduces_variance(self):
        """Test that ensemble averaging reduces prediction variance."""
        np.random.seed(42)

        # Simulate 10 models with noisy predictions
        n_samples = 100
        true_values = np.random.rand(n_samples) * 10

        individual_variances = []
        all_predictions = []

        for i in range(10):
            noise = np.random.randn(n_samples) * 0.5
            pred = true_values + noise
            all_predictions.append(pred)
            individual_variances.append(np.var(pred - true_values))

        # Ensemble prediction
        ensemble_pred = np.mean(all_predictions, axis=0)
        ensemble_variance = np.var(ensemble_pred - true_values)

        # Ensemble variance should be lower than average individual variance
        avg_individual_var = np.mean(individual_variances)
        assert ensemble_variance < avg_individual_var

    def test_ensemble_with_identical_models(self):
        """Test ensemble with identical predictions (edge case)."""
        predictions = [
            [5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0]
        ]

        ensemble_pred = np.mean(predictions, axis=0)

        expected = [5.0, 6.0, 7.0]
        np.testing.assert_array_equal(ensemble_pred, expected)


@pytest.mark.integration
class TestMultiModelPrediction:
    """Test predictions from multiple model instances."""

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
            dataset='ensemble_test',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

    def test_different_seeds_produce_different_predictions(
        self, mock_config, test_dataset, device
    ):
        """Test that different seeds produce different model predictions."""
        predictions = []

        for seed in [42, 123, 456]:
            torch.manual_seed(seed)

            model = get_model(
                'GATv2Net',
                node_feature_dim=test_dataset.num_node_features,
                edge_feature_dim=test_dataset.num_edge_features,
                config=mock_config
            )
            model.apply(init_weights)
            model.to(device)
            model.eval()

            loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output = model(data)
                    predictions.append(output.item())

        # Different seeds should produce different predictions
        assert len(set(predictions)) > 1, "Different seeds should produce different predictions"

    def test_same_seed_produces_same_predictions(
        self, mock_config, test_dataset, device
    ):
        """Test that same seed produces identical model predictions."""
        predictions = []

        for _ in range(2):
            torch.manual_seed(42)

            model = get_model(
                'GATv2Net',
                node_feature_dim=test_dataset.num_node_features,
                edge_feature_dim=test_dataset.num_edge_features,
                config=mock_config
            )
            model.apply(init_weights)
            model.to(device)
            model.eval()

            loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output = model(data)
                    predictions.append(output.item())

        # Same seed should produce identical predictions
        assert predictions[0] == predictions[1], "Same seed should produce identical predictions"

    def test_ensemble_from_multiple_trained_models(
        self, mock_config, test_dataset, device, temp_data_dir
    ):
        """Test ensemble prediction from multiple saved models."""
        model_paths = []
        n_models = 3

        # Train and save multiple models
        for i, seed in enumerate([100, 200, 300]):
            torch.manual_seed(seed)

            model = get_model(
                'GATv2Net',
                node_feature_dim=test_dataset.num_node_features,
                edge_feature_dim=test_dataset.num_edge_features,
                config=mock_config
            )
            model.apply(init_weights)

            # Save model
            model_path = os.path.join(temp_data_dir, f'model_{i}.pt')
            torch.save(model.state_dict(), model_path)
            model_paths.append(model_path)

        # Load models and make ensemble prediction
        all_predictions = []
        loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for model_path in model_paths:
            model = get_model(
                'GATv2Net',
                node_feature_dim=test_dataset.num_node_features,
                edge_feature_dim=test_dataset.num_edge_features,
                config=mock_config
            )
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.to(device)
            model.eval()

            model_preds = []
            with torch.no_grad():
                for data in loader:
                    data = data.to(device)
                    output = model(data)
                    model_preds.append(output.item())

            all_predictions.append(model_preds)

        # Compute ensemble average
        ensemble_pred = np.mean(all_predictions, axis=0)

        assert len(ensemble_pred) == len(test_dataset)
        assert not np.isnan(ensemble_pred).any()
