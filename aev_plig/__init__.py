"""
AEV-PLIG: Graph Neural Network-based Scoring Function for Protein-Ligand Binding Affinity Prediction

This package provides tools for:
- Loading and parsing protein-ligand structures
- Computing Atomic Environment Vectors (AEVs)
- Generating molecular graphs
- Training GNN models for binding affinity prediction
- Making predictions on new protein-ligand complexes
"""

__version__ = "2.0.0"

# Expose key classes and functions at package level
from aev_plig.models import GATv2Net, get_model
from aev_plig.config import Config

__all__ = [
    'GATv2Net',
    'get_model',
    'Config',
]
