"""
Integration tests for scaler functionality.

Tests: scaler creation, transformation, serialization, and round-trip.
"""

import pytest
import numpy as np
import pickle
import tempfile
import shutil
import os

from sklearn.preprocessing import StandardScaler

from aev_plig.datasets import GraphDataset


@pytest.mark.integration
class TestScalerRoundTrip:
    """Test scaler round-trip transformations."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_scaler_transform_inverse_transform(self):
        """Test that scaler transform/inverse_transform preserves values."""
        y = np.array([2.0, 5.0, 8.0, 10.0])

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))

        transformed = scaler.transform(y.reshape(-1, 1))
        recovered = scaler.inverse_transform(transformed)

        np.testing.assert_array_almost_equal(y, recovered.flatten(), decimal=10)

    def test_scaler_transform_normalizes(self):
        """Test that scaler normalizes to zero mean, unit variance."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))
        transformed = scaler.transform(y.reshape(-1, 1)).flatten()

        # Mean should be ~0, std should be ~1
        assert np.abs(transformed.mean()) < 1e-10
        assert np.abs(transformed.std() - 1.0) < 0.1  # ddof difference

    def test_scaler_serialization(self, temp_data_dir):
        """Test that scaler can be pickled and unpickled."""
        y = np.array([2.0, 5.0, 8.0, 10.0])

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))

        # Save
        scaler_path = os.path.join(temp_data_dir, 'scaler.pickle')
        with open(scaler_path, 'wb') as f:
            pickle.dump(scaler, f)

        # Load
        with open(scaler_path, 'rb') as f:
            loaded_scaler = pickle.load(f)

        # Verify they produce same results
        test_values = np.array([3.0, 7.0]).reshape(-1, 1)
        original_transform = scaler.transform(test_values)
        loaded_transform = loaded_scaler.transform(test_values)

        np.testing.assert_array_almost_equal(original_transform, loaded_transform)

    def test_scaler_from_dataset(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test scaler created by GraphDataset."""
        ids = list(sample_graphs_dict.keys())

        dataset = GraphDataset(
            root=temp_data_dir,
            dataset='scaler_test',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        scaler = dataset.y_scaler

        # Verify scaler is fitted
        assert hasattr(scaler, 'mean_')
        assert hasattr(scaler, 'scale_')

    def test_scaler_shared_between_datasets(self, sample_graphs_dict, sample_labels, temp_data_dir):
        """Test that scaler can be shared between train/valid datasets."""
        ids = list(sample_graphs_dict.keys())

        # Create train dataset (creates scaler)
        train_dataset = GraphDataset(
            root=temp_data_dir,
            dataset='scaler_train',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=None
        )

        # Create valid dataset (uses train scaler)
        valid_dataset = GraphDataset(
            root=temp_data_dir,
            dataset='scaler_valid',
            ids=ids,
            y=sample_labels,
            graphs_dict=sample_graphs_dict,
            y_scaler=train_dataset.y_scaler
        )

        # Both should use same scaler parameters
        assert train_dataset.y_scaler.mean_[0] == valid_dataset.y_scaler.mean_[0]
        assert train_dataset.y_scaler.scale_[0] == valid_dataset.y_scaler.scale_[0]

    def test_scaler_handles_single_value(self):
        """Test scaler behavior with single value (edge case)."""
        y = np.array([5.0])

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))

        # With single value, variance is 0, so scale_ will be 1 (sklearn default)
        transformed = scaler.transform(y.reshape(-1, 1))

        # Should transform to 0 (mean subtraction)
        assert transformed[0, 0] == 0.0

        # Round-trip should work
        recovered = scaler.inverse_transform(transformed)
        np.testing.assert_array_almost_equal(y, recovered.flatten())

    def test_scaler_with_varied_values(self):
        """Test scaler with diverse value range."""
        # Typical pKd range
        y = np.array([2.5, 4.0, 5.5, 7.0, 8.5, 10.0, 11.5])

        scaler = StandardScaler()
        scaler.fit(y.reshape(-1, 1))

        # Test round-trip
        transformed = scaler.transform(y.reshape(-1, 1))
        recovered = scaler.inverse_transform(transformed).flatten()

        np.testing.assert_array_almost_equal(y, recovered, decimal=10)

        # Test with new values
        new_values = np.array([3.0, 6.0, 9.0]).reshape(-1, 1)
        new_transformed = scaler.transform(new_values)
        new_recovered = scaler.inverse_transform(new_transformed)

        np.testing.assert_array_almost_equal(new_values, new_recovered, decimal=10)
