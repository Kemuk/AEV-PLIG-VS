"""
Prediction module for binding affinity prediction.

This module provides classes for validating data, processing graphs,
and making predictions on new protein-ligand complexes.
"""

import json
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from rdkit import Chem
from torch.amp import autocast
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from aev_plig.config import Config
from aev_plig.graphs import create_graph
from aev_plig.loaders import compute_aevs, load_ligand_atoms, load_protein_atoms_biopandas
from aev_plig.models import GATv2NetBayesianMixedPrecision, GATv2NetMixedPrecision, MODEL_REGISTRY


def denormalize(tensor_or_array, scaler):
    """Inverse-transform a 1-D prediction tensor/array using a fitted scaler."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.cpu().detach().numpy()
    else:
        arr = np.asarray(tensor_or_array)
    return scaler.inverse_transform(arr.flatten().reshape(-1, 1)).flatten()


def denormalize_variance(tensor_or_array, scaler):
    """Rescale variance from normalized space to pK² units (var_pK = var_norm × σ²)."""
    if isinstance(tensor_or_array, torch.Tensor):
        arr = tensor_or_array.cpu().detach().numpy()
    else:
        arr = np.asarray(tensor_or_array)
    return arr.flatten() * float(scaler.scale_[0] ** 2)


class Validator:
    """
    Validates protein and ligand structures before processing.

    Checks for:
    - Readability by RDKit (ligands)
    - Readability by BioPandas (proteins)
    - Presence of rare/unsupported atoms
    - Unsupported bond types
    """

    def __init__(self, atom_keys=None, allowed_atoms=None, skip_protein_validation=False):
        """
        Args:
            atom_keys: DataFrame with PDB atom type mappings (required for protein validation)
            allowed_atoms: Set of allowed ligand atom types (default: from Config)
            skip_protein_validation: If True, skip protein structure validation
        """
        self.atom_keys = atom_keys
        self.allowed_atoms = allowed_atoms if allowed_atoms else set(Config.ALLOWED_ATOMS)
        self.skip_protein_validation = skip_protein_validation

    def validate_ligands(self, df):
        """
        Validate ligand structures from dataset.

        Args:
            df: DataFrame with 'sdf_file' and 'unique_id' columns

        Returns:
            pd.DataFrame: Filtered DataFrame with only valid ligands
        """
        print("Checking what molecules are readable by RDKit, and which contain rare atoms\n")

        non_readable = []
        rare_atoms_ids = []

        # Use to_dict('records') for 5-10x faster iteration than iterrows()
        for row in tqdm(df.to_dict('records'), total=len(df)):
            suppl = Chem.SDMolSupplier(row["sdf_file"], removeHs=False)
            assert len(suppl) == 1, f"SDF file should contain exactly 1 molecule: {row['sdf_file']}"
            lig = suppl[0]

            if lig is None:
                non_readable.append(row["unique_id"])
            else:
                mol_df = load_ligand_atoms(lig)
                if not set(mol_df["ATOM_TYPE"].values).issubset(self.allowed_atoms):
                    rare_atoms_ids.append(row["unique_id"])

        print("Number of SDF files not read by RDKit:", len(non_readable))
        print("Number of SDF files with rare elements:", len(rare_atoms_ids))

        df = df[~df["unique_id"].isin(rare_atoms_ids)].reset_index(drop=True)
        df = df[~df["unique_id"].isin(non_readable)].reset_index(drop=True)
        print()

        return df

    def validate_proteins(self, df, num_workers=1):
        """
        Validate protein structures from dataset using BioPandas.

        Args:
            df: DataFrame with 'pdb_file' and 'unique_id' columns
            num_workers: Number of parallel workers for validation

        Returns:
            pd.DataFrame: Filtered DataFrame with only valid proteins
        """
        if self.skip_protein_validation:
            print("Skipping BioPandas validation..\n")
            return df

        if self.atom_keys is None:
            raise ValueError("atom_keys required for protein validation")

        # Add RESIDUE column if not present
        if "RESIDUE" not in self.atom_keys.columns:
            # Use .str accessor for 2-5x faster vectorized string operations
            self.atom_keys["RESIDUE"] = self.atom_keys["PDB_ATOM"].str.split("-").str[0]

        print("Checking what protein structures are readable by BioPandas\n")

        # Convert dataframe rows to list of dictionaries
        rows = df.to_dict("records")

        # Use a partial function to pass atom_keys to the validate_row function
        validate_with_keys = partial(self._validate_protein_row, atom_keys=self.atom_keys)

        # Process the rows in parallel
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            results = list(tqdm(executor.map(validate_with_keys, rows), total=len(rows)))

        # Filter out None values to get non-readable IDs
        non_readable = [r for r in results if r is not None]

        print("Number of PDB files not read by BioPandas:", len(non_readable))
        df = df[~df["unique_id"].isin(non_readable)].reset_index(drop=True)
        print()

        return df

    @staticmethod
    def _validate_protein_row(row, atom_keys):
        """Helper function to validate a single protein structure."""
        try:
            load_protein_atoms_biopandas(row["pdb_file"], atom_keys)
            return None  # Return None if no exception occurs
        except Exception:
            return row["unique_id"]  # Return unique_id on failure

    def analyze_features(self, df):
        """
        Analyse atom features and bond types in the dataset.

        Combines atom and edge analysis into a single pass over the data
        to avoid reading each SDF file twice.

        Args:
            df: DataFrame with 'sdf_file' column

        Returns:
            pd.DataFrame: Filtered DataFrame (removes molecules with unspecified bond types)
        """
        print("Analyse atom features and edges\n")

        features = []
        bond_types = []
        unspecified_bond_mol = []

        # Single pass over all molecules (instead of two separate loops)
        # Use to_dict('records') for 5-10x faster iteration than iterrows()
        for row in tqdm(df.to_dict('records'), total=len(df)):
            suppl = Chem.SDMolSupplier(row["sdf_file"], removeHs=False)
            lig = suppl[0]

            heavy_atom_index = set()  # Use set for O(1) membership test

            # Atom analysis
            for atom in lig.GetAtoms():
                if atom.GetSymbol() != "H":
                    heavy_atom_index.add(atom.GetIdx())

                    neighbors = atom.GetNeighbors()
                    features.append([
                        atom.GetSymbol(),
                        sum(1 for x in neighbors if x.GetSymbol() != "H"),
                        sum(1 for x in neighbors if x.GetSymbol() == "H"),
                        atom.GetExplicitValence(),
                        1 if atom.GetIsAromatic() else 0,
                        1 if atom.IsInRing() else 0,
                    ])

            # Edge analysis (same loop iteration, reuse heavy_atom_index)
            for bond in lig.GetBonds():
                idx1 = bond.GetBeginAtomIdx()
                idx2 = bond.GetEndAtomIdx()
                if idx1 in heavy_atom_index and idx2 in heavy_atom_index:
                    bond_types.append(bond.GetBondType())
                    if bond.GetBondType() == 0:
                        unspecified_bond_mol.append(row["unique_id"])

        # Print atom feature statistics
        features_df = pd.DataFrame(features, columns=[
            "atom_symbol", "num_heavy_atoms", "total_num_Hs",
            "explicit_valence", "is_aromatic", "is_in_ring"
        ])

        print("Atom features:")
        print(features_df["atom_symbol"].value_counts())
        print(features_df["num_heavy_atoms"].value_counts())
        print(features_df["total_num_Hs"].value_counts())
        print(features_df["explicit_valence"].value_counts())
        print(features_df["is_aromatic"].value_counts())
        print(features_df["is_in_ring"].value_counts())
        print()

        # Print edge statistics
        print("Edge analysis:")
        bond_df = pd.DataFrame(data={"bond_type": bond_types})
        print(bond_df["bond_type"].value_counts())
        print("Number of molecules with unspecified bond types:", len(unspecified_bond_mol))

        df = df[~df["unique_id"].isin(unspecified_bond_mol)].reset_index(drop=True)
        print()

        return df


class GraphProcessor:
    """
    Processes protein-ligand complexes into molecular graphs.

    Handles graph generation with optional multiprocessing support.
    """

    def __init__(self, atom_keys, atom_map, radial_coefs):
        """
        Args:
            atom_keys: DataFrame with PDB atom type mappings
            atom_map: DataFrame mapping ATOM_TYPE to ATOM_NR
            radial_coefs: List of [RcR, EtaR, RsR] for AEV computation
        """
        self.atom_keys = atom_keys
        self.atom_map = atom_map
        self.radial_coefs = radial_coefs

    def process_single(self, row_dict):
        """
        Process a single protein-ligand complex into a graph.

        Args:
            row_dict: Dictionary with 'sdf_file', 'pdb_file', 'unique_id'

        Returns:
            tuple: (unique_id, graph_tuple)
        """
        # Set single-threaded for this worker
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

        sdf_file = row_dict["sdf_file"]
        pdb_file = row_dict["pdb_file"]
        unique_id = row_dict["unique_id"]

        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        lig = suppl[0]

        mol_df, aevs = compute_aevs(pdb_file, lig, self.atom_keys, self.radial_coefs, self.atom_map)
        graph = create_graph(lig, mol_df, aevs)

        return unique_id, graph

    def process_batch(self, df, num_workers=1):
        """
        Process multiple complexes in parallel.

        Args:
            df: DataFrame with 'sdf_file', 'pdb_file', 'unique_id' columns
            num_workers: Number of parallel workers

        Returns:
            dict: Dictionary mapping unique_id to graph tuples
        """
        print("Generating graphs\n")
        print(f"Using {num_workers} workers for graph generation.")

        mol_graphs = {}
        # Use to_dict('records') directly - 5-10x faster than iterrows()
        rows = df.to_dict('records')

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(self.process_single, row) for row in rows]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating graphs"):
                try:
                    unique_id, graph = future.result()
                    mol_graphs[unique_id] = graph
                except Exception as e:
                    print("Error processing a graph:", e)

        return mol_graphs


class Predictor:
    """
    Makes binding affinity predictions using ensemble of trained models.

    Loads multiple model checkpoints and averages their predictions.
    """

    def __init__(self, model_class, model_paths, scaler_path, device, config):
        """
        Args:
            model_class: Model class (e.g., GATv2Net)
            model_paths: List of paths to model checkpoints
            scaler_path: Path to scaler pickle file
            device: PyTorch device
            config: Configuration object with model parameters
        """
        self.model_class = model_class
        self.model_paths = model_paths
        self.device = device
        self.config = config
        self.use_amp = (
            issubclass(model_class, (GATv2NetMixedPrecision, GATv2NetBayesianMixedPrecision))
            and device.type == 'cuda'
        )

        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)

        print(f"Loaded {len(model_paths)} model checkpoints")
        if self.use_amp:
            print("Mixed precision inference enabled (AMP)")

    def predict(self, dataset):
        """
        Make predictions on a dataset using ensemble of models.

        Args:
            dataset: list[torch_geometric.data.Data]

        Returns:
            pd.DataFrame: Predictions with columns ['graph_id', 'preds_0', ..., 'preds']
        """
        print("Making predictions\n")

        if len(dataset) == 0:
            print("Warning: Empty dataset, returning empty predictions")
            return pd.DataFrame({"graph_id": [], "preds": []})

        loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)

        # Initialise model architecture
        model = self.model_class(
            node_feature_dim=dataset[0].x.shape[1],
            edge_feature_dim=dataset[0].edge_attr.shape[1],
            config=self.config
        )

        # Collect all predictions first, then build DataFrame once (2-3x faster)
        is_bayesian = self.model_class.is_bayesian
        all_preds: list = []
        all_vars:  list = []
        graph_ids = None

        # Predict with each model in ensemble
        for i, model_path in enumerate(self.model_paths):
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            model.to(self.device)

            total_preds = torch.Tensor().to(self.device)
            total_vars  = torch.Tensor().to(self.device) if is_bayesian else None
            graph_offset = 0  # Track position in dataset using offset, not data.y
            total_graph_ids = torch.LongTensor()

            with torch.no_grad():
                for data in loader:
                    data = data.to(self.device)
                    with autocast('cuda', enabled=self.use_amp):
                        if is_bayesian:
                            mean_out, var_out = model.forward(data)
                        else:
                            mean_out = model.predict(data)
                    total_preds = torch.cat((total_preds, mean_out), 0)
                    if is_bayesian:
                        total_vars = torch.cat((total_vars, var_out), 0)
                    # Use positional counting: data.y contains pK targets, not graph IDs
                    batch_ids = torch.arange(graph_offset, graph_offset + mean_out.shape[0])
                    total_graph_ids = torch.cat((total_graph_ids, batch_ids), 0)
                    graph_offset += mean_out.shape[0]

            # Denormalise mean predictions
            if graph_ids is None:
                graph_ids = total_graph_ids.numpy().flatten()
            preds = denormalize(total_preds, self.scaler)
            all_preds.append(preds)

            # Denormalise variance: var_pK = var_norm * sigma²  (sigma = scaler.scale_[0])
            if is_bayesian:
                all_vars.append(denormalize_variance(total_vars, self.scaler))

        # Build DataFrame once with all data
        pred_data = {'graph_id': graph_ids}
        for i, preds in enumerate(all_preds):
            pred_data[f'preds_{i}'] = preds

        df_preds = pd.DataFrame(pred_data)

        # Compute ensemble average
        pred_cols = [c for c in df_preds.columns if c.startswith('preds_')]
        df_preds['preds'] = df_preds[pred_cols].mean(axis=1)

        # Bayesian aleatoric variance columns (pK² units, mean across checkpoints)
        if is_bayesian and all_vars:
            for i, vars_ in enumerate(all_vars):
                df_preds[f'var_{i}'] = vars_
            var_cols = [f'var_{i}' for i in range(len(all_vars))]
            df_preds['var'] = df_preds[var_cols].mean(axis=1)

        return df_preds


# ==================== Pipeline Functions ====================
# These functions are called by scripts/predict.py (and scripts/train.py for
# post-training prediction). All non-trivial logic lives here; the scripts are
# thin shells.

def validate_and_process_csv(config):
    """Validate data, generate graphs, and create PyTorch dataset from CSV."""
    from aev_plig.datasets import create_dataset
    start_time = time.time()

    print("\n" + "="*60)
    print("STEP 1: VALIDATE DATA")
    print("="*60 + "\n")

    df = pd.read_csv(config.dataset_csv)
    atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=",")
    atom_keys["RESIDUE"] = atom_keys["PDB_ATOM"].str.split("-").str[0]

    validator = Validator(atom_keys=atom_keys, skip_protein_validation=config.skip_validation)
    df = validator.validate_ligands(df)
    df = validator.validate_proteins(df, num_workers=config.num_workers)
    df = validator.analyze_features(df)

    if len(df) == 0:
        raise ValueError("No valid molecules remaining after validation!")

    processed_csv = config.dataset_csv.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv, index=False)
    print(f"Saved processed dataset to {processed_csv}\n")

    print("\n" + "="*60)
    print("STEP 2: GENERATE MOLECULAR GRAPHS")
    print("="*60 + "\n")

    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    radial_coefs = Config.get_radial_coefs()
    processor = GraphProcessor(atom_keys, atom_map, radial_coefs)
    mol_graphs = processor.process_batch(df, num_workers=config.num_workers)

    output_graphs_file = f"data/{config.data_name}_graphs.pickle"
    with open(output_graphs_file, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graph_time = time.time() - start_time
    print(f"\nGraph generation time: {graph_time:.2f} seconds\n")

    print("\n" + "="*60)
    print("STEP 3: CREATE PYTORCH DATASET")
    print("="*60 + "\n")

    df["graph_id"] = range(len(df))
    test_ids = list(df["unique_id"])
    test_graph_ids = list(df["graph_id"])

    test_data, _ = create_dataset(test_ids, test_graph_ids, mol_graphs, scale=False)

    return test_data, df, graph_time


def load_processed_data(config):
    """Load PyTorch dataset from processed data directory."""
    data_name = config.data_name
    split = 'test'

    dataset_dir = Path(Config.PROCESSED_DATA_DIR) / data_name
    split_dir = dataset_dir / split
    manifest_path = split_dir / "manifest.json"

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"No processed data found at {split_dir}\n"
            f"Run create_pytorch_data.py first to generate processed datasets."
        )

    scaler_path = dataset_dir / "scaler.pickle"
    if not scaler_path.exists():
        print(f"Warning: No scaler found at {scaler_path}")
        print("   Predictions may not be properly denormalized.")

    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Loading processed dataset: {data_name}/{split}")
    print(f"  Chunks: {len(manifest['parts'])}")
    print(f"  Graphs: {manifest['num_graphs_written']}")

    all_data = []
    for part_file in tqdm(manifest['parts'], desc="Loading chunks"):
        part_path = split_dir / part_file
        data_chunk = torch.load(part_path, weights_only=False)
        all_data.extend(data_chunk)

    print(f"Loaded {len(all_data)} graphs\n")

    if all_data and hasattr(all_data[0], 'unique_id'):
        df = pd.DataFrame({
            'graph_id': range(len(all_data)),
            'unique_id': [d.unique_id for d in all_data],
            'pK':        [d.pK for d in all_data],
            'sdf_file':  [d.sdf_file for d in all_data],
            'pdb_file':  [d.pdb_file for d in all_data],
        })
    else:
        df = pd.DataFrame({'graph_id': list(range(len(all_data)))})

    return all_data, df


def load_data(config):
    """Dispatcher: load from processed dir or CSV depending on config."""
    if config.use_processed or config.dataset_csv is None:
        data, df = load_processed_data(config)
        return data, df, None
    return validate_and_process_csv(config)  # returns (data, df, graph_time)


def run_predictions(test_data, df, config):
    """Make predictions using ensemble of trained models.

    Loads model architecture from config.json in the model directory if present
    (written by train.py). Falls back to config.model for backward compat with
    models trained before this change.
    """
    print("\n" + "="*60)
    print("STEP: MAKE PREDICTIONS")
    print("="*60 + "\n")

    os.environ["OMP_NUM_THREADS"] = str(config.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(config.num_workers)
    torch.set_num_threads(config.num_workers)

    model_dir = f'{Config.TRAINED_MODELS_DIR}/{config.trained_model_name}'
    print(f"Model directory: {model_dir}")

    # Load arch from config.json if present (train.py saves this)
    config_path = Path(model_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            model_cfg = json.load(f)
        model_class = MODEL_REGISTRY[model_cfg["model"]]
        pred_config = type('Namespace', (), {**model_cfg, "device": config.device})()
    else:
        # Backward compat: use CLI-provided arch flags
        model_class = MODEL_REGISTRY[config.model]
        pred_config = config

    model_paths = sorted(str(p) for p in Path(model_dir).glob("*.model"))
    scaler_path = f'{model_dir}/scaler.pickle'

    predictor = Predictor(
        model_class=model_class,
        model_paths=model_paths,
        scaler_path=scaler_path,
        device=config.device,
        config=pred_config,
    )

    df_preds = predictor.predict(test_data)
    df = df.merge(df_preds, on='graph_id', how='left')

    return df


def save_results(df, config, total_time, graph_time=None):
    """Save predictions and print summary. Prints metrics if pK column present."""
    print("\n" + "="*60)
    print("STEP: SAVE RESULTS")
    print("="*60 + "\n")

    output_file = (
        f"{Config.PREDICTIONS_DIR}/"
        f"{config.trained_model_name}/{config.data_name}_predictions.parquet"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pl.from_pandas(df).write_parquet(output_file)
    print(f"Saved predictions to {output_file}")

    # Metrics when ground truth is available
    if 'pK' in df.columns and 'preds' in df.columns:
        from aev_plig.results import overall_metrics
        pl_df = pl.from_pandas(df)
        print("\n" + "="*60)
        print("PREDICTION METRICS")
        print("="*60)
        for col in sorted(c for c in df.columns if c.startswith('preds_')):
            m = overall_metrics(pl_df, truth_col='pK', pred_col=col)
            print(f"  {col}: R={m['pearson_r']:.4f}  RMSE={m['rmse']:.4f}  τ={m['kendall_tau']:.4f}")
        ens = overall_metrics(pl_df, truth_col='pK', pred_col='preds')
        print(f"  ensemble: R={ens['pearson_r']:.4f}  RMSE={ens['rmse']:.4f}"
              f"  τ={ens['kendall_tau']:.4f}  n={ens['n']:.0f}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total complexes processed: {len(df)}")
    if graph_time is not None:
        print(f"Graph generation time: {graph_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    print("="*60 + "\n")