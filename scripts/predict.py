"""
Make binding affinity predictions on protein-ligand complexes.

This script processes protein-ligand complexes, generates graphs,
and makes predictions using an ensemble of trained models.
"""

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from tqdm import tqdm

from aev_plig.config import Config
from aev_plig.datasets import create_dataset
from aev_plig.models import MODEL_REGISTRY
from aev_plig.prediction import GraphProcessor, Predictor, Validator

# Suppress TorchANI warnings
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions on protein-ligand complexes')
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME,
                        help='Model architecture name')
    parser.add_argument('--trained_model_name', type=str,
                        default='model_GATv2Net_ligsim90_fep_benchmark',
                        help='Trained model name (without extension)')
    parser.add_argument('--dataset_csv', type=str, default=None,
                        help='Path to dataset CSV file (optional - if not provided, loads from processed data)')
    parser.add_argument('--data_name', type=str, default='example',
                        help='Name for output files')
    parser.add_argument('--use_processed', action='store_true',
                        help='Force loading from processed data directory (ignores dataset_csv)')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--head', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--activation_function', type=str, default='leaky_relu',
                        help='Activation function')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for processing (0=all available cores)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto", "cpu", or CUDA device index')
    parser.add_argument('--skip_validation', action='store_true',
                        help='Skip BioPandas validation of protein structures')

    args = parser.parse_args()
    return args


def get_device(device_param):
    """Get PyTorch device from parameter."""
    if device_param.lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif device_param.lower() == "cpu":
        return torch.device("cpu")
    else:
        device_idx = int(device_param)
        if device_idx >= torch.cuda.device_count():
            sys.exit(f"CUDA device {device_param} doesn't exist!")
        return torch.device(f"cuda:{device_param}")


def setup_environment(config):
    """
    Configure environment settings for prediction.

    Args:
        config: Parsed command line arguments

    Returns:
        None (modifies config in-place)
    """
    # Set up multiprocessing
    if config.num_workers <= 0:
        config.num_workers = os.cpu_count()
        print(f"Using all available cores: {config.num_workers} workers")
    else:
        print(f"Using {config.num_workers} worker(s)")

    # Configure device
    config.device = get_device(config.device)
    print(f"Using device: {config.device}\n")


def validate_and_process_csv(config):
    """
    Validate data, generate graphs, and create PyTorch dataset from CSV.

    This function handles Steps 1-3 of the CSV mode pipeline:
    - Validate ligands and proteins
    - Generate molecular graphs
    - Create PyTorch Geometric datasets

    Args:
        config: Configuration object with dataset_csv, num_workers, etc.

    Returns:
        tuple: (test_data, df, graph_time)
            - test_data: List of PyTorch Geometric Data objects
            - df: Pandas DataFrame with processed data
            - graph_time: Time taken for graph generation (seconds)
    """
    start_time = time.time()

    # Step 1: Validate Data
    print("\n" + "="*60)
    print("STEP 1: VALIDATE DATA")
    print("="*60 + "\n")

    df = pd.read_csv(config.dataset_csv)
    atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=",")
    # Use .str accessor for 2-5x faster vectorized string operations
    atom_keys["RESIDUE"] = atom_keys["PDB_ATOM"].str.split("-").str[0]

    validator = Validator(atom_keys=atom_keys, skip_protein_validation=config.skip_validation)
    df = validator.validate_ligands(df)
    df = validator.validate_proteins(df, num_workers=config.num_workers)
    df = validator.analyze_features(df)

    if len(df) == 0:
        raise ValueError("No valid molecules remaining after validation!")

    # Save processed dataset
    processed_csv = config.dataset_csv.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv, index=False)
    print(f"Saved processed dataset to {processed_csv}\n")

    # Step 2: Generate Graphs
    print("\n" + "="*60)
    print("STEP 2: GENERATE MOLECULAR GRAPHS")
    print("="*60 + "\n")

    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    radial_coefs = Config.get_radial_coefs()
    processor = GraphProcessor(atom_keys, atom_map, radial_coefs)
    mol_graphs = processor.process_batch(df, num_workers=config.num_workers)

    # Save graphs
    output_graphs_file = f"data/{config.data_name}_graphs.pickle"
    with open(output_graphs_file, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graph_time = time.time() - start_time
    print(f"\nGraph generation time: {graph_time:.2f} seconds\n")

    # Step 3: Create PyTorch Dataset
    print("\n" + "="*60)
    print("STEP 3: CREATE PYTORCH DATASET")
    print("="*60 + "\n")

    df["graph_id"] = range(len(df))
    test_ids = list(df["unique_id"])
    test_graph_ids = list(df["graph_id"])

    test_data, _ = create_dataset(
        test_ids,
        test_graph_ids,
        mol_graphs,
        scale=False,
    )

    return test_data, df, graph_time


def load_processed_data(config):
    """
    Load PyTorch dataset from processed data directory.

    Args:
        config: Configuration object with data_name

    Returns:
        tuple: (test_data, df)
            - test_data: List of PyTorch Geometric Data objects
            - df: Minimal DataFrame with graph_id column for output
    """
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

    # Check if scaler exists (needed for predictions)
    scaler_path = dataset_dir / "scaler.pickle"
    if not scaler_path.exists():
        print(f"⚠️  Warning: No scaler found at {scaler_path}")
        print("   Predictions may not be properly denormalized.")

    # Load manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    print(f"Loading processed dataset: {data_name}/{split}")
    print(f"  Chunks: {len(manifest['parts'])}")
    print(f"  Graphs: {manifest['num_graphs_written']}")

    # Load all parts
    all_data = []
    for part_file in tqdm(manifest['parts'], desc="Loading chunks"):
        part_path = split_dir / part_file
        data_chunk = torch.load(part_path, weights_only=False)
        all_data.extend(data_chunk)

    print(f"✓ Loaded {len(all_data)} graphs\n")

    # Build df from metadata stored directly on each Data object
    if all_data and hasattr(all_data[0], 'unique_id'):
        df = pd.DataFrame({
            'graph_id': range(len(all_data)),
            'unique_id': [d.unique_id for d in all_data],
            'pK':        [d.pK for d in all_data],
            'sdf_file':  [d.sdf_file for d in all_data],
            'pdb_file':  [d.pdb_file for d in all_data],
        })
    else:
        # Fallback: minimal df for backward compatibility with older .pt files
        df = pd.DataFrame({'graph_id': list(range(len(all_data)))})

    return all_data, df


def run_predictions(test_data, df, config):
    """
    Make predictions using ensemble of trained models.

    Args:
        test_data: List of PyTorch Geometric Data objects
        df: DataFrame to merge predictions with
        config: Configuration object with model settings

    Returns:
        pd.DataFrame: DataFrame with predictions merged
    """
    print("\n" + "="*60)
    print("STEP: MAKE PREDICTIONS")
    print("="*60 + "\n")

    # Restore multi-threading for prediction
    os.environ["OMP_NUM_THREADS"] = str(config.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(config.num_workers)
    torch.set_num_threads(config.num_workers)

    model_dir = f'{Config.TRAINED_MODELS_DIR}/{config.trained_model_name}'
    print(f"Model directory: {model_dir}")

    # Get model paths (ensemble of 10 models)
    model_paths = sorted(str(p) for p in Path(model_dir).glob("*.model"))
    scaler_path = f'{model_dir}/scaler.pickle'

    # Create predictor
    predictor = Predictor(
        model_class=MODEL_REGISTRY[config.model],
        model_paths=model_paths,
        scaler_path=scaler_path,
        device=config.device,
        config=config
    )

    # Make predictions
    df_preds = predictor.predict(test_data)

    # Merge predictions with original data
    df = df.merge(df_preds, on='graph_id', how='left')

    return df


def save_results(df, config, total_time, graph_time=None):
    """
    Save predictions and print summary.

    Args:
        df: DataFrame with predictions
        config: Configuration object with output settings
        total_time: Total pipeline time in seconds
        graph_time: Graph generation time in seconds (None for processed mode)
    """
    print("\n" + "="*60)
    print("STEP: SAVE RESULTS")
    print("="*60 + "\n")

    # Create hierarchical output directory structure for better organization
    output_file = f"{Config.PREDICTIONS_DIR}/{config.model}/{config.trained_model_name}/{config.data_name}_predictions.parquet"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pl.from_pandas(df).write_parquet(output_file)
    print(f"Saved predictions to {output_file}")

    # Print summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total complexes processed: {len(df)}")
    if graph_time is not None:
        print(f"Graph generation time: {graph_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    print("="*60 + "\n")


def main():
    """Main prediction pipeline orchestrator."""
    start_time = time.time()

    # Parse arguments
    config = parse_args()

    # Setup environment
    setup_environment(config)

    # ==================== DETERMINE MODE ====================
    use_processed = config.use_processed or config.dataset_csv is None

    if use_processed:
        # ==================== PROCESSED DATA MODE ====================
        print("\n" + "="*60)
        print("MODE: LOADING FROM PROCESSED DATA")
        print("="*60 + "\n")

        test_data, df = load_processed_data(config)
        graph_time = None  # No graph generation in this mode

    else:
        # ==================== CSV MODE ====================
        print("\n" + "="*60)
        print("MODE: PROCESSING NEW DATA FROM CSV")
        print("="*60 + "\n")

        test_data, df, graph_time = validate_and_process_csv(config)

    # ==================== RUN PREDICTIONS ====================
    df = run_predictions(test_data, df, config)

    # ==================== SAVE RESULTS ====================
    total_time = time.time() - start_time
    save_results(df, config, total_time, graph_time)


if __name__ == "__main__":
    main()