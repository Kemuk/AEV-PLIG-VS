"""
Graph construction functions for molecular structures.

This module provides functions to convert molecular structures into graph representations
suitable for graph neural network models.
"""

import pandas as pd
import numpy as np
from aev_plig.features import atom_features, one_of_k_encoding
from aev_plig.config import Config


def create_graph(mol, mol_df, aevs, extra_features=None):
    """
    Convert molecule to graph representation.

    Creates a graph with:
    - Nodes: Heavy atoms with features (atom properties + AEVs)
    - Edges: Bonds with features (bond type one-hot encoding)

    Args:
        mol: RDKit molecule object
        mol_df: DataFrame of molecule atoms (from load_ligand_atoms)
        aevs: Tensor of AEVs for each atom (from compute_aevs)
        extra_features: List of feature names to extract. If None, uses Config.ATOM_FEATURES

    Returns:
        tuple: (num_atoms, node_features, edge_index, edge_features)
            - num_atoms: Number of atoms in the molecule
            - node_features: List of feature arrays for each node
            - edge_index: List of [source, target] pairs for edges
            - edge_features: List of feature arrays for each edge
    """
    if extra_features is None:
        extra_features = Config.ATOM_FEATURES

    features = []
    heavy_atom_index = []
    idx_to_idx = {}
    counter = 0

    # Generate nodes (heavy atoms only)
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            idx_to_idx[atom.GetIdx()] = counter
            aev_idx = mol_df[mol_df['ATOM_INDEX'] == atom.GetIdx()].index
            heavy_atom_index.append(atom.GetIdx())
            # Concatenate atom features with AEVs
            feature = np.append(atom_features(atom, extra_features), aevs[aev_idx, :])
            features.append(feature)
            counter += 1

    # Generate edges (bonds between heavy atoms)
    edges = []
    for bond in mol.GetBonds():
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
            # One-hot encode bond type
            bond_type = one_of_k_encoding(bond.GetBondType(), Config.BOND_TYPES)
            bond_type = [float(b) for b in bond_type]

            # Add edge in both directions (undirected graph)
            edge1 = [idx_to_idx[idx1], idx_to_idx[idx2]]
            edge1.extend(bond_type)
            edge2 = [idx_to_idx[idx2], idx_to_idx[idx1]]
            edge2.extend(bond_type)
            edges.append(edge1)
            edges.append(edge2)

    # Convert to DataFrame for sorting
    df = pd.DataFrame(edges, columns=['atom1', 'atom2', 'single', 'aromatic', 'double', 'triple'])
    df = df.sort_values(by=['atom1', 'atom2'])

    edge_index = df[['atom1', 'atom2']].to_numpy().tolist()
    edge_attr = df[['single', 'aromatic', 'double', 'triple']].to_numpy().tolist()

    return len(mol_df), features, edge_index, edge_attr
