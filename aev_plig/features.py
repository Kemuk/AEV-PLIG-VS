"""
Feature extraction functions for atoms and bonds.

This module provides functions to extract features from molecular structures
for use in graph neural network models.
"""

import numpy as np
from aev_plig.config import Config


def one_of_k_encoding(x, allowable_set):
    """
    One-hot encode a value given an allowable set.

    Args:
        x: Value to encode
        allowable_set: List of allowed values

    Returns:
        list: One-hot encoded binary list

    Raises:
        Exception: If x is not in allowable_set
    """
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def atom_features(atom, features=None):
    """
    Extract features from an RDKit atom for graph node construction.

    Default features include:
    - atom_symbol: One-hot encoding of atom type
    - num_heavy_atoms: Number of heavy atom neighbors
    - total_num_Hs: Number of hydrogen neighbors
    - explicit_valence: Explicit valence of the atom
    - is_aromatic: Boolean (1 if aromatic, 0 otherwise)
    - is_in_ring: Boolean (1 if in ring, 0 otherwise)

    Args:
        atom: RDKit atom object
        features: List of feature names to extract. If None, uses Config.ATOM_FEATURES

    Returns:
        np.ndarray: Array of atom features
    """
    if features is None:
        features = Config.ATOM_FEATURES

    feature_list = []

    if "atom_symbol" in features:
        feature_list.extend(one_of_k_encoding(atom.GetSymbol(), Config.ALLOWED_ATOMS))

    if "num_heavy_atoms" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() != "H"]))

    if "total_num_Hs" in features:
        feature_list.append(len([x.GetSymbol() for x in atom.GetNeighbors() if x.GetSymbol() == "H"]))

    if "explicit_valence" in features:
        feature_list.append(atom.GetExplicitValence())

    if "is_aromatic" in features:
        if atom.GetIsAromatic():
            feature_list.append(1)
        else:
            feature_list.append(0)

    if "is_in_ring" in features:
        if atom.IsInRing():
            feature_list.append(1)
        else:
            feature_list.append(0)

    return np.array(feature_list)


def bond_features(bond):
    """
    Extract features from an RDKit bond for graph edge construction.

    Returns one-hot encoding of bond type:
    - SINGLE (1)
    - AROMATIC (12)
    - DOUBLE (2)
    - TRIPLE (3)

    Args:
        bond: RDKit bond object

    Returns:
        list: One-hot encoded bond type features
    """
    bond_type = one_of_k_encoding(bond.GetBondType(), Config.BOND_TYPES)
    return [float(b) for b in bond_type]
