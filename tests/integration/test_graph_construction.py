"""
Integration tests for graph construction pipeline.

Tests the full path: PDB + SDF -> loaders -> features -> graphs
"""

import pytest
import numpy as np
import torch

from aev_plig.loaders import load_ligand_atoms, load_protein_atoms, compute_aevs
from aev_plig.graphs import create_graph
from aev_plig.config import Config


@pytest.mark.integration
class TestGraphConstruction:
    """Test the graph construction pipeline."""

    def test_load_ligand_atoms(self, sample_mol):
        """Test ligand atom loading from RDKit molecule."""
        mol_df = load_ligand_atoms(sample_mol)

        # Check DataFrame structure
        assert 'ATOM_INDEX' in mol_df.columns
        assert 'ATOM_TYPE' in mol_df.columns
        assert 'X' in mol_df.columns
        assert 'Y' in mol_df.columns
        assert 'Z' in mol_df.columns

        # Check we only have heavy atoms (no hydrogens)
        assert 'H' not in mol_df['ATOM_TYPE'].values

        # Check we have atoms
        assert len(mol_df) > 0
        assert len(mol_df) == sample_mol.GetNumHeavyAtoms()

    def test_load_protein_atoms(self, sample_pdb_path, atom_keys):
        """Test protein atom loading from PDB file."""
        protein_df = load_protein_atoms(sample_pdb_path, atom_keys)

        # Check DataFrame structure
        assert 'ATOM_INDEX' in protein_df.columns
        assert 'ATOM_TYPE' in protein_df.columns
        assert 'X' in protein_df.columns
        assert 'Y' in protein_df.columns
        assert 'Z' in protein_df.columns

        # Check we have atoms
        assert len(protein_df) > 0

    def test_compute_aevs_shape(self, sample_pdb_path, sample_mol, atom_keys, radial_coefs, atom_map):
        """Test AEV computation returns correct shape."""
        mol_df, aevs = compute_aevs(
            sample_pdb_path, sample_mol, atom_keys, radial_coefs, atom_map
        )

        # AEVs should be [num_ligand_atoms, 352]
        # 352 = 22 atom types * 16 radial shifts
        expected_aev_dim = Config.AEV_DIM
        assert aevs.shape[0] == len(mol_df)
        assert aevs.shape[1] == expected_aev_dim

        # AEVs should be non-negative (radial basis functions)
        assert (aevs >= 0).all()

    def test_compute_aevs_returns_tensor(self, sample_aevs):
        """Test that compute_aevs returns a torch tensor."""
        mol_df, aevs = sample_aevs
        assert isinstance(aevs, torch.Tensor)

    def test_create_graph_structure(self, sample_mol, sample_aevs):
        """Test graph creation returns correct structure."""
        mol_df, aevs = sample_aevs
        num_atoms, node_features, edge_index, edge_features = create_graph(
            sample_mol, mol_df, aevs
        )

        # Check num_atoms matches
        assert num_atoms == sample_mol.GetNumHeavyAtoms()
        assert num_atoms == len(node_features)

        # Check node features dimension
        # atom_features (10 atom types + 4 other features = 14 or 16) + AEV (352)
        for feat in node_features:
            assert len(feat) > Config.AEV_DIM  # Has both atom features and AEVs

    def test_create_graph_edges_bidirectional(self, sample_mol, sample_aevs):
        """Test that edges are bidirectional (undirected graph)."""
        mol_df, aevs = sample_aevs
        num_atoms, node_features, edge_index, edge_features = create_graph(
            sample_mol, mol_df, aevs
        )

        # Edges should come in pairs (a->b and b->a)
        assert len(edge_index) % 2 == 0
        assert len(edge_features) == len(edge_index)

        # Check edge features are valid one-hot encodings
        for feat in edge_features:
            assert len(feat) == 4  # single, aromatic, double, triple
            assert sum(feat) == 1.0  # exactly one bond type

    def test_create_graph_edge_indices_valid(self, sample_mol, sample_aevs):
        """Test that edge indices are within valid range."""
        mol_df, aevs = sample_aevs
        num_atoms, node_features, edge_index, edge_features = create_graph(
            sample_mol, mol_df, aevs
        )

        for src, dst in edge_index:
            assert 0 <= src < num_atoms
            assert 0 <= dst < num_atoms

    def test_full_pipeline_integration(self, sample_pdb_path, sample_sdf_path, atom_keys, radial_coefs, atom_map):
        """Test the full pipeline from files to graph."""
        from rdkit import Chem

        # Load molecule
        suppl = Chem.SDMolSupplier(sample_sdf_path, removeHs=False)
        mol = suppl[0]
        assert mol is not None

        # Compute AEVs
        mol_df, aevs = compute_aevs(
            sample_pdb_path, mol, atom_keys, radial_coefs, atom_map
        )

        # Create graph
        num_atoms, node_features, edge_index, edge_features = create_graph(
            mol, mol_df, aevs
        )

        # Validate complete output
        assert num_atoms > 0
        assert len(node_features) == num_atoms
        assert len(edge_index) > 0  # Molecule should have bonds
        assert len(edge_features) == len(edge_index)
