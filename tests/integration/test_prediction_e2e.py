"""
End-to-end integration tests for prediction pipeline.

Tests the full path: CSV -> validation -> graph generation -> prediction
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import shutil
import os

from aev_plig.prediction import Validator, GraphProcessor
from aev_plig.datasets import GraphDatasetPredict
from aev_plig.config import Config


@pytest.mark.integration
class TestValidator:
    """Test Validator class."""

    def test_validator_initialization(self, atom_keys):
        """Test Validator initializes correctly."""
        validator = Validator(atom_keys=atom_keys)

        assert validator.atom_keys is not None
        assert validator.allowed_atoms == set(Config.ALLOWED_ATOMS)
        assert validator.skip_protein_validation is False

    def test_validator_with_custom_atoms(self, atom_keys):
        """Test Validator with custom allowed atoms."""
        custom_atoms = {'C', 'N', 'O'}
        validator = Validator(atom_keys=atom_keys, allowed_atoms=custom_atoms)

        assert validator.allowed_atoms == custom_atoms

    def test_validate_ligands_valid_molecule(self, atom_keys, sample_dataset_df):
        """Test validate_ligands with valid molecule."""
        validator = Validator(atom_keys=atom_keys)

        result_df = validator.validate_ligands(sample_dataset_df)

        # Valid molecule should pass validation
        assert len(result_df) == len(sample_dataset_df)

    def test_validate_ligands_filters_invalid(self, atom_keys, temp_data_dir):
        """Test validate_ligands filters non-readable molecules."""
        validator = Validator(atom_keys=atom_keys)

        # Create DataFrame with non-existent file
        df = pd.DataFrame({
            'unique_id': ['nonexistent'],
            'sdf_file': ['/nonexistent/path.sdf'],
            'pdb_file': ['/nonexistent/path.pdb']
        })

        # This should raise an error or filter out the invalid entry
        with pytest.raises(Exception):
            validator.validate_ligands(df)

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.mark.integration
class TestGraphProcessor:
    """Test GraphProcessor class."""

    @pytest.fixture
    def graph_processor(self, atom_keys, atom_map, radial_coefs):
        """Create GraphProcessor instance."""
        return GraphProcessor(
            atom_keys=atom_keys,
            atom_map=atom_map,
            radial_coefs=radial_coefs
        )

    def test_processor_initialization(self, graph_processor, atom_keys):
        """Test GraphProcessor initializes correctly."""
        assert graph_processor.atom_keys is not None
        assert graph_processor.atom_map is not None
        assert graph_processor.radial_coefs is not None

    def test_process_single(self, graph_processor, sample_dataset_df):
        """Test processing a single molecule."""
        row_dict = sample_dataset_df.iloc[0].to_dict()

        unique_id, graph = graph_processor.process_single(row_dict)

        assert unique_id == row_dict['unique_id']
        assert len(graph) == 4  # (num_atoms, features, edge_index, edge_features)

        num_atoms, features, edge_index, edge_features = graph
        assert num_atoms > 0
        assert len(features) == num_atoms
        assert len(edge_features) == len(edge_index)

    def test_process_batch(self, graph_processor, sample_dataset_df):
        """Test batch processing of molecules."""
        mol_graphs = graph_processor.process_batch(sample_dataset_df, num_workers=1)

        assert len(mol_graphs) == len(sample_dataset_df)

        for unique_id in sample_dataset_df['unique_id']:
            assert unique_id in mol_graphs


@pytest.mark.integration
class TestGraphDatasetPredict:
    """Test GraphDatasetPredict for inference."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_predict_dataset_creation(self, sample_graphs_dict, temp_data_dir):
        """Test creating prediction dataset."""
        ids = list(sample_graphs_dict.keys())
        graph_ids = list(range(len(ids)))

        dataset = GraphDatasetPredict(
            root=temp_data_dir,
            dataset='predict_test',
            ids=ids,
            graph_ids=graph_ids,
            graphs_dict=sample_graphs_dict
        )

        assert len(dataset) == len(ids)

    def test_predict_dataset_stores_graph_id(self, sample_graphs_dict, temp_data_dir):
        """Test that prediction dataset stores graph ID in y field."""
        ids = list(sample_graphs_dict.keys())
        graph_ids = [42]  # Custom graph ID

        dataset = GraphDatasetPredict(
            root=temp_data_dir,
            dataset='predict_id_test',
            ids=ids,
            graph_ids=graph_ids,
            graphs_dict=sample_graphs_dict
        )

        data = dataset[0]
        assert data.y.item() == 42


@pytest.mark.integration
@pytest.mark.slow
class TestEndToEndPrediction:
    """Full end-to-end prediction pipeline test."""

    @pytest.fixture
    def temp_data_dir(self):
        """Create a temporary directory for test data."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir, ignore_errors=True)

    def test_full_prediction_pipeline(
        self, atom_keys, atom_map, radial_coefs, sample_dataset_df, temp_data_dir
    ):
        """Test complete prediction pipeline from CSV to dataset."""
        # Step 1: Validate ligands
        validator = Validator(atom_keys=atom_keys, skip_protein_validation=True)
        df = validator.validate_ligands(sample_dataset_df)

        assert len(df) > 0, "Validation should pass for valid molecules"

        # Step 2: Generate graphs
        processor = GraphProcessor(
            atom_keys=atom_keys,
            atom_map=atom_map,
            radial_coefs=radial_coefs
        )
        mol_graphs = processor.process_batch(df, num_workers=1)

        assert len(mol_graphs) == len(df), "All molecules should be processed"

        # Step 3: Create prediction dataset
        ids = df['unique_id'].tolist()
        graph_ids = list(range(len(ids)))

        dataset = GraphDatasetPredict(
            root=temp_data_dir,
            dataset='e2e_test',
            ids=ids,
            graph_ids=graph_ids,
            graphs_dict=mol_graphs
        )

        assert len(dataset) == len(df), "Dataset should contain all molecules"

        # Verify dataset structure
        data = dataset[0]
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None
