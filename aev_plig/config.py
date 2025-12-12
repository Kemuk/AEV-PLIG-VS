"""
Configuration management for AEV-PLIG

Centralizes all configuration parameters, paths, and constants.
"""

import torch
import os


class Config:
    """
    Centralized configuration for AEV-PLIG package.

    All hardcoded parameters, paths, and constants are defined here.
    Can be easily extended or overridden for different experiments.
    """

    # ==================== Paths ====================
    DATA_DIR = "data"
    OUTPUT_DIR = "output"
    TRAINED_MODELS_DIR = os.path.join("output", "trained_models")
    PREDICTIONS_DIR = os.path.join("output", "predictions")
    PROCESSED_DATA_DIR = os.path.join("data", "processed")
    ATOM_KEYS_FILE = os.path.join("data", "PDB_Atom_Keys.csv")

    # ==================== Atom Types ====================
    # Allowed ligand atom types for one-hot encoding
    ALLOWED_ATOMS = ['F', 'N', 'Cl', 'O', 'Br', 'C', 'B', 'P', 'I', 'S']

    # Number of atom types for protein (used in AEV computation)
    NUM_PROTEIN_ATOM_TYPES = 22  # Atoms 1-22 in periodic table

    # ==================== AEV Parameters (ANI-2x) ====================
    # Radial coefficients
    AEV_RADIAL_CUTOFF = 5.1  # RcR - Radial cutoff in Angstroms
    AEV_RADIAL_ETA = 19.7    # EtaR - Radial decay parameter
    AEV_RADIAL_SHIFTS = [0.80, 1.07, 1.34, 1.61, 1.88, 2.14, 2.41, 2.68,
                         2.95, 3.22, 3.49, 3.76, 4.03, 4.29, 4.56, 4.83]

    # Angular coefficients (not used, but kept for compatibility)
    AEV_ANGULAR_CUTOFF = 2.0
    AEV_ANGULAR_ZETA = 1.0
    AEV_ANGULAR_TS = 1.0
    AEV_ANGULAR_ETA = 1.0
    AEV_ANGULAR_RS = 1.0

    # Computed AEV dimension (22 atom types * 16 radial shifts = 352)
    AEV_DIM = NUM_PROTEIN_ATOM_TYPES * len(AEV_RADIAL_SHIFTS)

    # Distance cutoff for protein atoms filtering (adds small buffer to radial cutoff)
    DISTANCE_CUTOFF_BUFFER = 0.1

    # ==================== Feature Extraction ====================
    # Features to extract for each atom
    ATOM_FEATURES = [
        "atom_symbol",      # One-hot encoding of atom type
        "num_heavy_atoms",  # Number of heavy atom neighbors
        "total_num_Hs",     # Number of hydrogen neighbors
        "explicit_valence", # Explicit valence
        "is_aromatic",      # Boolean: is aromatic
        "is_in_ring"        # Boolean: is in ring
    ]

    # Bond types for edge features (RDKit bond type codes)
    BOND_TYPES = [1, 12, 2, 3]  # SINGLE=1, AROMATIC=12, DOUBLE=2, TRIPLE=3

    # ==================== Model Architecture (GATv2Net) ====================
    HIDDEN_DIM = 256
    NUM_GNN_LAYERS = 5
    NUM_ATTENTION_HEADS = 3
    DROPOUT = 0.0

    # MLP (fully connected) layers after GNN
    MLP_DIMS = [1024, 512, 256]

    # Activation function (options: 'relu', 'leaky_relu')
    ACTIVATION_FUNCTION = 'leaky_relu'

    # Pooling method (concat of max and mean pooling)
    POOLING_METHOD = 'concat'  # max + mean

    # ==================== Training Parameters ====================
    BATCH_SIZE = 128
    LEARNING_RATE = 0.00012291937615434127
    WEIGHT_DECAY = 0.0
    NUM_EPOCHS = 200

    # Early stopping based on rolling average Pearson correlation
    EARLY_STOPPING_WINDOW = 8  # Rolling window size

    # Ensemble training
    ENSEMBLE_SIZE = 10
    ENSEMBLE_SEEDS = [100, 123, 15, 257, 2, 2012, 3752, 350, 843, 621]

    # Loss function
    LOSS_FUNCTION = 'mse'  # Mean squared error

    # ==================== Data Split ====================
    TRAIN_RATIO = 0.8
    VALID_RATIO = 0.1
    TEST_RATIO = 0.1

    # Tanimoto similarity threshold for train/test separation
    TANIMOTO_THRESHOLD = 0.9

    # ==================== Multiprocessing ====================
    DEFAULT_NUM_WORKERS = 0  # 0 means use all available CPU cores

    # ==================== Device ====================
    @staticmethod
    def get_device(device_param='auto'):
        """
        Get PyTorch device based on parameter.

        Args:
            device_param: 'auto' (use CUDA if available), 'cpu', or CUDA device index

        Returns:
            torch.device
        """
        if device_param.lower() == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif device_param.lower() == "cpu":
            return torch.device("cpu")
        else:
            # Assume the user provided a valid CUDA device index
            device_idx = int(device_param)
            if device_idx >= torch.cuda.device_count():
                raise ValueError(f"CUDA device {device_param} doesn't exist!")
            return torch.device(f"cuda:{device_param}")

    @staticmethod
    def get_radial_coefs():
        """
        Get radial coefficients as tensors for AEV computation.

        Returns:
            list: [RcR, EtaR, RsR] where RsR is a tensor
        """
        return [
            Config.AEV_RADIAL_CUTOFF,
            torch.tensor([Config.AEV_RADIAL_ETA]),
            torch.tensor(Config.AEV_RADIAL_SHIFTS)
        ]

    @staticmethod
    def get_angular_coefs():
        """
        Get angular coefficients as tensors for AEV computation.

        Returns:
            dict: Angular coefficient parameters
        """
        return {
            'RcA': Config.AEV_ANGULAR_CUTOFF,
            'Zeta': torch.tensor([Config.AEV_ANGULAR_ZETA]),
            'TsA': torch.tensor([Config.AEV_ANGULAR_TS]),
            'EtaA': torch.tensor([Config.AEV_ANGULAR_ETA]),
            'RsA': torch.tensor([Config.AEV_ANGULAR_RS])
        }
