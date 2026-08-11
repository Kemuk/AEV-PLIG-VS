"""
Shared fixtures for AEV-PLIG tests.

Provides common test data, configurations, and helper objects.
"""

import pytest
import pandas as pd
import numpy as np
import torch
import os
from pathlib import Path
from rdkit import Chem

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from aev_plig.config import Config
from aev_plig.models import get_model
from aev_plig.loaders import load_ligand_atoms, compute_aevs
from aev_plig.graphs import create_graph


# Paths
FIXTURES_DIR = Path(__file__).parent / "fixtures"
MOLECULES_DIR = FIXTURES_DIR / "molecules"
PROJECT_ROOT = Path(__file__).parent.parent


@pytest.fixture(scope="session")
def atom_keys():
    """Load atom keys DataFrame."""
    return pd.read_csv(PROJECT_ROOT / Config.ATOM_KEYS_FILE, sep=",")


@pytest.fixture(scope="session")
def atom_map(atom_keys):
    """Create atom map from atom keys."""
    return atom_keys[["ATOM_TYPE", "ATOM_NR"]].drop_duplicates()


@pytest.fixture(scope="session")
def radial_coefs():
    """Get radial coefficients from config."""
    return Config.get_radial_coefs()


@pytest.fixture(scope="session")
def sample_sdf_path():
    """Path to sample ligand SDF file."""
    return str(MOLECULES_DIR / "3ao4.sdf")


@pytest.fixture(scope="session")
def sample_pdb_path():
    """Path to sample protein PDB file."""
    return str(MOLECULES_DIR / "3ao4.pdb")


@pytest.fixture(scope="session")
def sample_mol(sample_sdf_path):
    """Load a sample RDKit molecule."""
    suppl = Chem.SDMolSupplier(sample_sdf_path, removeHs=False)
    return suppl[0]


@pytest.fixture(scope="session")
def sample_mol_df(sample_mol):
    """Load ligand atoms as DataFrame."""
    return load_ligand_atoms(sample_mol)


@pytest.fixture(scope="session")
def sample_aevs(sample_pdb_path, sample_mol, atom_keys, radial_coefs, atom_map):
    """Compute AEVs for sample molecule."""
    mol_df, aevs = compute_aevs(
        sample_pdb_path, sample_mol, atom_keys, radial_coefs, atom_map
    )
    return mol_df, aevs


@pytest.fixture(scope="session")
def sample_graph(sample_mol, sample_aevs):
    """Create graph from sample molecule."""
    mol_df, aevs = sample_aevs
    return create_graph(sample_mol, mol_df, aevs)


@pytest.fixture(scope="session")
def sample_graphs_dict(sample_graph):
    """Dictionary of sample graphs for dataset creation."""
    return {"3ao4": sample_graph}


@pytest.fixture
def mock_config():
    """Mock config for model creation."""
    class MockConfig:
        hidden_dim = 64  # Smaller for faster tests
        head = 2
        activation_function = 'leaky_relu'
    return MockConfig()


@pytest.fixture
def node_feature_dim(sample_graph):
    """Get node feature dimension from sample graph."""
    num_atoms, features, edge_index, edge_attr = sample_graph
    return len(features[0])


@pytest.fixture
def edge_feature_dim():
    """Edge feature dimension (bond types)."""
    return 4  # single, aromatic, double, triple


@pytest.fixture
def untrained_model(mock_config, node_feature_dim, edge_feature_dim):
    """Create untrained GATv2Net for testing."""
    return get_model(
        'GATv2Net',
        node_feature_dim=node_feature_dim,
        edge_feature_dim=edge_feature_dim,
        config=mock_config
    )


@pytest.fixture(scope="session")
def device():
    """Get test device (CPU for CI)."""
    return torch.device("cpu")


@pytest.fixture
def sample_dataset_df():
    """Create a minimal dataset DataFrame for testing."""
    return pd.DataFrame({
        'unique_id': ['3ao4'],
        'sdf_file': [str(MOLECULES_DIR / '3ao4.sdf')],
        'pdb_file': [str(MOLECULES_DIR / '3ao4.pdb')]
    })


@pytest.fixture
def sample_labels():
    """Sample binding affinity labels for testing."""
    return [7.5]  # Example pKd value


@pytest.fixture
def cleanup_processed_files():
    """Fixture to clean up processed .pt files after tests."""
    created_files = []
    yield created_files
    # Cleanup after test
    for f in created_files:
        if os.path.exists(f):
            os.remove(f)
