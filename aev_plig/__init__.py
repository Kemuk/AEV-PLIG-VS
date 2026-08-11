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

__all__ = [
    'GATv2Net',
    'get_model',
    'Config',
    'results',
]


def __getattr__(name):
    if name == 'GATv2Net':
        from aev_plig.models import GATv2Net
        return GATv2Net
    if name == 'get_model':
        from aev_plig.models import get_model
        return get_model
    if name == 'Config':
        from aev_plig.config import Config
        return Config
    if name == 'results':
        import importlib
        return importlib.import_module('aev_plig.results')
    raise AttributeError(f"module 'aev_plig' has no attribute {name!r}")
