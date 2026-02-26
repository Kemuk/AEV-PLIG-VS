"""
Graph Neural Network models for binding affinity prediction.

This module provides the model architecture and model registry for easy model selection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import global_max_pool as gmp
from torch_geometric.nn import global_mean_pool as gap
from torch_geometric.nn import BatchNorm
from aev_plig.config import Config


# Activation function registry
ACTIVATION_FUNCTIONS = {
    "relu": F.relu,
    "leaky_relu": F.leaky_relu,
}


class BaseGATv2Net(torch.nn.Module, ABC):
    """
    Abstract base class for GATv2-based graph neural network models.

    Architecture:
    - N GATv2Conv layers with batch normalization
    - Global pooling (concatenation of max and mean pooling)
    - 3 fully connected layers with batch normalization
    - Output head defined by subclasses

    Args:
        node_feature_dim: Dimension of node features
        edge_feature_dim: Dimension of edge features
        config: Configuration object or namespace with model parameters (optional, defaults to Config)
    """

    is_bayesian: bool = False

    def __init__(self, node_feature_dim, edge_feature_dim, config=None):
        super().__init__()

        # Get configuration parameters
        if config is None:
            config = Config

        if hasattr(config, 'activation_function'):
            self.act = config.activation_function
        else:
            self.act = Config.ACTIVATION_FUNCTION

        if hasattr(config, 'hidden_dim'):
            hidden_dim = config.hidden_dim
        else:
            hidden_dim = Config.HIDDEN_DIM

        if hasattr(config, 'head'):
            head = config.head
        else:
            head = Config.NUM_ATTENTION_HEADS

        self.number_GNN_layers = Config.NUM_GNN_LAYERS
        self.activation = ACTIVATION_FUNCTIONS[self.act]

        # GNN layers
        self.GNN_layers = nn.ModuleList()
        self.BN_layers = nn.ModuleList()

        input_dim = node_feature_dim

        # First GNN layer
        self.GNN_layers.append(GATv2Conv(input_dim, hidden_dim, heads=head, edge_dim=edge_feature_dim))
        self.BN_layers.append(BatchNorm(hidden_dim * head))

        # Remaining GNN layers
        for i in range(1, self.number_GNN_layers):
            self.GNN_layers.append(GATv2Conv(hidden_dim * head, hidden_dim, heads=head, edge_dim=edge_feature_dim))
            self.BN_layers.append(BatchNorm(hidden_dim * head))

        final_dim = hidden_dim * head

        # Fully connected layers (MLP)
        mlp_dims = Config.MLP_DIMS
        self.fc1 = nn.Linear(final_dim * 2, mlp_dims[0])  # *2 for concatenated pooling
        self.bn_connect1 = nn.BatchNorm1d(mlp_dims[0])
        self.fc2 = nn.Linear(mlp_dims[0], mlp_dims[1])
        self.bn_connect2 = nn.BatchNorm1d(mlp_dims[1])
        self.fc3 = nn.Linear(mlp_dims[1], mlp_dims[2])
        self.bn_connect3 = nn.BatchNorm1d(mlp_dims[2])

    def _gnn_forward(self, x, edge_index, edge_attr):
        """Run input through GNN layers with batch normalization."""
        for layer, bn in zip(self.GNN_layers, self.BN_layers):
            x = layer(x, edge_index, edge_attr)
            x = self.activation(x)
            x = bn(x)
        return x

    def _mlp_forward(self, x):
        """Run input through fully connected layers with batch normalization."""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.bn_connect1(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.bn_connect2(x)
        x = self.fc3(x)
        x = self.activation(x)
        x = self.bn_connect3(x)
        return x

    @abstractmethod
    def _output_head(self, x):
        """Apply the final output head. Subclasses define their output layer(s)."""
        pass

    def forward(self, data):
        """
        Forward pass through the network.

        Args:
            data: PyTorch Geometric Data object with x, edge_index, edge_attr, batch

        Returns:
            Output from the subclass output head
        """
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self._gnn_forward(x, edge_index, edge_attr)

        # Global pooling (concatenate max and mean pooling)
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = self._mlp_forward(x)

        return self._output_head(x)

    @abstractmethod
    def predict(self, data):
        """Return point predictions for inference."""
        pass


class GATv2Net(BaseGATv2Net):
    """
    Graph Attention Network v2 for protein-ligand binding affinity prediction.

    Output: Single value (predicted binding affinity)
    """

    def __init__(self, node_feature_dim, edge_feature_dim, config=None):
        super().__init__(node_feature_dim, edge_feature_dim, config)
        mlp_dims = Config.MLP_DIMS
        self.out = nn.Linear(mlp_dims[2], 1)

    def _output_head(self, x):
        return self.out(x)

    def predict(self, data):
        """Return point predictions for inference."""
        return self.forward(data)


class GATv2NetBayesian(BaseGATv2Net):
    """
    Bayesian Graph Attention Network v2 for protein-ligand binding affinity prediction.

    Output: (mean, variance) tuple for uncertainty quantification
    """

    is_bayesian: bool = True

    def __init__(self, node_feature_dim, edge_feature_dim, config=None):
        super().__init__(node_feature_dim, edge_feature_dim, config)
        mlp_dims = Config.MLP_DIMS
        self.mean_head = nn.Linear(mlp_dims[2], 1)
        self.logvar_head = nn.Linear(mlp_dims[2], 1)

    def _output_head(self, x):
        mean = self.mean_head(x)
        var = F.softplus(self.logvar_head(x)) + 1e-6
        return mean, var

    def predict(self, data):
        """Return point predictions for inference (mean only)."""
        mean, _ = self.forward(data)
        return mean


class GATv2NetMixedPrecision(GATv2Net):
    """
    Mixed precision GATv2Net.

    GNN layers run in fp16 under the Trainer's autocast context.
    MLP layers are forced to fp32 to avoid numerical instability in the
    fully connected layers.
    """

    def _mlp_forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            return super()._mlp_forward(x.float())


class GATv2NetBayesianMixedPrecision(GATv2NetBayesian):
    """
    Mixed precision Bayesian GATv2Net.

    GNN layers run in fp16 under the Trainer's autocast context.
    MLP layers are forced to fp32 to avoid numerical instability in the
    fully connected layers.
    """

    def _mlp_forward(self, x):
        with torch.amp.autocast('cuda', enabled=False):
            return super()._mlp_forward(x.float())


# Model registry for easy model selection
MODEL_REGISTRY = {
    'GATv2Net': GATv2Net,
    'GATv2NetBayesian': GATv2NetBayesian,
    'GATv2NetMixedPrecision': GATv2NetMixedPrecision,
    'GATv2NetBayesianMixedPrecision': GATv2NetBayesianMixedPrecision,
}


def get_model(name, **kwargs):
    """
    Get a model from the registry.

    Args:
        name: Model name (e.g., 'GATv2Net')
        **kwargs: Arguments to pass to model constructor. Common arguments:
            - node_feature_dim (int): Dimension of node features
            - edge_feature_dim (int): Dimension of edge features
            - config (optional): Configuration object or namespace with model parameters.
                               If not provided, uses defaults from Config class.

    Returns:
        torch.nn.Module: Instantiated model

    Raises:
        KeyError: If model name is not in registry

    Example:
        >>> # Create model with default config
        >>> model = get_model('GATv2Net', node_feature_dim=25, edge_feature_dim=4)
        >>>
        >>> # Create model with custom config
        >>> class CustomConfig:
        ...     hidden_dim = 128
        ...     head = 4
        >>> model = get_model('GATv2Net', node_feature_dim=25, edge_feature_dim=4, config=CustomConfig())
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not found in registry. Available models: {list(MODEL_REGISTRY.keys())}")

    return MODEL_REGISTRY[name](**kwargs)


def list_models():
    """
    List all available models in the registry.

    Returns:
        list: List of model names
    """
    return list(MODEL_REGISTRY.keys())
