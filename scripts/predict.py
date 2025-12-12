"""
Make binding affinity predictions on protein-ligand complexes.

This script processes protein-ligand complexes, generates graphs,
and makes predictions using an ensemble of trained models.
"""

import pandas as pd
import pickle
import torch
import os
import argparse
import time
import sys
import warnings

from aev_plig.prediction import Validator, GraphProcessor, Predictor
from aev_plig.datasets import GraphDatasetPredict
from aev_plig.models import get_model
from aev_plig.config import Config
import numpy as np

# Suppress TorchANI warnings
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Make predictions on protein-ligand complexes')
    parser.add_argument('--trained_model_name', type=str,
                        default='20231116-181233_model_GATv2Net_pdbbind_core',
                        help='Trained model name (without extension)')
    parser.add_argument('--dataset_csv', type=str, default='data/example_dataset.csv',
                        help='Path to dataset CSV file')
    parser.add_argument('--data_name', type=str, default='example',
                        help='Name for output files')
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


def main():
    """Main prediction pipeline."""
    config = parse_args()

    # Set up multiprocessing
    if config.num_workers <= 0:
        config.num_workers = os.cpu_count()
        print(f"Using all available cores: {config.num_workers} workers")
    else:
        print(f"Using {config.num_workers} worker(s)")

    # Configure device
    config.device = get_device(config.device)
    print(f"Using device: {config.device}")

    # ==================== Step 1: Validate Data ====================
    print("\n" + "="*60)
    print("STEP 1: VALIDATE DATA")
    print("="*60 + "\n")

    df = pd.read_csv(config.dataset_csv)
    atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=",")
    atom_keys["RESIDUE"] = atom_keys["PDB_ATOM"].apply(lambda x: x.split("-")[0])

    validator = Validator(atom_keys=atom_keys, skip_protein_validation=config.skip_validation)

    # Validate ligands
    df = validator.validate_ligands(df)

    # Validate proteins (if not skipped)
    df = validator.validate_proteins(df, num_workers=config.num_workers)

    # Analyze features and remove problematic molecules
    df = validator.analyze_features(df)

    # Save processed dataset
    processed_csv = config.dataset_csv.replace('.csv', '_processed.csv')
    df.to_csv(processed_csv, index=False)
    print(f"Saved processed dataset to {processed_csv}\n")

    # ==================== Step 2: Generate Graphs ====================
    print("\n" + "="*60)
    print("STEP 2: GENERATE MOLECULAR GRAPHS")
    print("="*60 + "\n")

    start_time = time.time()

    # Prepare atom map
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    # Get radial coefficients
    radial_coefs = Config.get_radial_coefs()

    # Create graph processor and process all complexes
    processor = GraphProcessor(atom_keys, atom_map, radial_coefs)
    mol_graphs = processor.process_batch(df, num_workers=config.num_workers)

    # Save graphs
    output_graphs_file = f"data/{config.data_name}_graphs.pickle"
    with open(output_graphs_file, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    graph_time = time.time() - start_time
    print(f"\nGraph generation time: {graph_time:.2f} seconds\n")

    # ==================== Step 3: Create PyTorch Dataset ====================
    print("\n" + "="*60)
    print("STEP 3: CREATE PYTORCH DATASET")
    print("="*60 + "\n")

    df["graph_id"] = range(len(df))
    test_ids = list(df["unique_id"])
    test_graph_ids = list(df["graph_id"])

    # Remove existing .pt file if present
    pt_file = f"data/processed/{config.data_name}.pt"
    if os.path.exists(pt_file):
        os.remove(pt_file)

    test_data = GraphDatasetPredict(
        root='data',
        dataset=config.data_name,
        ids=test_ids,
        graph_ids=test_graph_ids,
        graphs_dict=mol_graphs
    )

    # ==================== Step 4: Make Predictions ====================
    print("\n" + "="*60)
    print("STEP 4: MAKE PREDICTIONS")
    print("="*60 + "\n")

    # Restore multi-threading for prediction
    os.environ["OMP_NUM_THREADS"] = str(config.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(config.num_workers)
    torch.set_num_threads(config.num_workers)

    # Get model paths (ensemble of 10 models)
    model_paths = [
        f'{Config.TRAINED_MODELS_DIR}/{config.trained_model_name}_{i}.model'
        for i in range(Config.ENSEMBLE_SIZE)
    ]

    # Load scaler
    scaler_path = f'{Config.TRAINED_MODELS_DIR}/{config.trained_model_name}.pickle'

    # Create predictor
    predictor = Predictor(
        model_class=get_model('GATv2Net'),
        model_paths=model_paths,
        scaler_path=scaler_path,
        device=config.device,
        config=config
    )

    # Make predictions
    df_preds = predictor.predict(test_data)

    # Merge predictions with original data
    df = df.merge(df_preds, on='graph_id', how='left')

    # ==================== Step 5: Save Results ====================
    print("\n" + "="*60)
    print("STEP 5: SAVE RESULTS")
    print("="*60 + "\n")

    output_file = f"{Config.PREDICTIONS_DIR}/{config.data_name}_predictions.csv"
    df.to_csv(output_file, index=False)
    print(f"Saved predictions to {output_file}")

    # ==================== Summary ====================
    total_time = time.time() - start_time
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total complexes processed: {len(df)}")
    print(f"Graph generation time: {graph_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
