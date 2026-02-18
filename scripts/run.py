"""
Train and/or predict binding affinities with an ensemble of GNN models.

Default mode : train ensemble -> evaluate on test split.
Predict-only : --predict-only  skip training, predict with a pre-trained model.
"""

import argparse
import json
import os
import pickle
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import torch
from torch.utils.data import ConcatDataset
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from aev_plig.config import Config
from aev_plig.datasets import create_dataset, init_weights
from aev_plig.models import MODEL_REGISTRY, get_model
from aev_plig.prediction import GraphProcessor, Predictor, Validator
from aev_plig.training import Trainer, pearson, rmse

# Suppress TorchANI warnings
warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train and/or predict binding affinities with an ensemble of GNN models'
    )

    # Mode
    parser.add_argument('--predict-only', action='store_true',
                        help='Skip training; predict using an existing trained model')

    # Shared (model architecture)
    parser.add_argument('--model', type=str, default=Config.MODEL_NAME,
                        help='Model architecture name')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--head', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--activation_function', type=str, default='leaky_relu',
                        help='Activation function')

    # Shared (device + workers)
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: "auto", "cpu", or CUDA device index')
    parser.add_argument('--num_workers', type=int, default=0,
                        help='Number of workers for processing (0=all available cores)')

    # Training args
    train_grp = parser.add_argument_group('training')
    train_grp.add_argument('--dataset', type=str, default='pdbbind_U_bindingnet_ligsim90',
                           help='Dataset name')
    train_grp.add_argument('--batch_size', type=int, default=128, help='Batch size')
    train_grp.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    train_grp.add_argument('--lr', type=float, default=0.00012291937615434127,
                           help='Learning rate')
    train_grp.add_argument('--seed', type=int, default=None,
                           help='Train single model with this seed (for parallel jobs)')
    train_grp.add_argument('--timestamp', type=str, default=None,
                           help='Timestamp for output directory (shared across parallel jobs)')

    # Prediction args
    pred_grp = parser.add_argument_group('prediction')
    pred_grp.add_argument('--trained_model_name', type=str,
                          default='model_GATv2Net_ligsim90_fep_benchmark',
                          help='Trained model name (required with --predict-only)')
    pred_grp.add_argument('--dataset_csv', type=str, default=None,
                          help='Path to dataset CSV file (optional)')
    pred_grp.add_argument('--data_name', type=str, default='example',
                          help='Name for output files')
    pred_grp.add_argument('--use_processed', action='store_true',
                          help='Force loading from processed data directory')
    pred_grp.add_argument('--skip_validation', action='store_true',
                          help='Skip BioPandas validation of protein structures')

    return parser.parse_args()


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


def _load_split(dataset_name, split):
    """Load split data from chunked manifest format, with flat-file fallback."""
    dataset_root = Path("data/processed") / dataset_name
    split_dir = dataset_root / split
    manifest_path = split_dir / "manifest.json"

    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        parts = [
            torch.load(split_dir / part_name, weights_only=False)
            for part_name in manifest["parts"]
        ]
        return parts[0] if len(parts) == 1 else ConcatDataset(parts)

    legacy_path = Path("data/processed") / f"{dataset_name}_{split}.pt"
    if legacy_path.exists():
        return torch.load(legacy_path, weights_only=False)

    raise FileNotFoundError(
        f"No dataset artifacts found for split '{split}'. "
        f"Checked {manifest_path} and {legacy_path}."
    )


def run_training(args, device, timestamp):
    """Train ensemble of models.

    Returns:
        tuple: (output_dir, test_data)
    """
    print(f'Training {args.model} on {args.dataset} for {args.epochs} epochs')

    Config.validate_ensemble_seeds()

    ensemble_seeds = [args.seed] if args.seed is not None else Config.ENSEMBLE_SEEDS
    if args.seed is not None:
        print(f"\nSingle-seed mode: training seed {args.seed}")
    else:
        print(f"\nEnsemble mode: training {len(ensemble_seeds)} models")

    output_dir = os.path.join('output', 'trained_models', f'{args.model}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    print(f"Timestamp: {timestamp}\n")

    train_data = _load_split(args.dataset, "train")
    valid_data = _load_split(args.dataset, "valid")
    test_data  = _load_split(args.dataset, "test")

    scaler_path = Path("data/processed") / args.dataset / "scaler.pickle"
    legacy_scaler = Path("data/processed") / f"{args.dataset}_scaler.pickle"
    with open(scaler_path if scaler_path.exists() else legacy_scaler, 'rb') as f:
        y_scaler = pickle.load(f)

    num_node_features = train_data[0].x.shape[1]
    num_edge_features = train_data[0].edge_attr.shape[1]
    print(f"Number of node features: {num_node_features}")
    print(f"Number of edge features: {num_edge_features}")
    print(f'Device: {device}')

    df_test = None
    for i, seed in enumerate(ensemble_seeds):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{len(ensemble_seeds)} with seed {seed}")
        print(f"{'='*60}\n")

        random.seed(seed)
        torch.manual_seed(int(seed))

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

        model = get_model(
            args.model,
            node_feature_dim=num_node_features,
            edge_feature_dim=num_edge_features,
            config=args,
        )
        model.apply(init_weights)

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            y_scaler=y_scaler,
            learning_rate=args.lr,
        )

        model_save_path = os.path.join(output_dir, f"model_seed_{seed}.model")
        trainer.fit(n_epochs=args.epochs, model_save_path=model_save_path)
        print(f"Saved: {model_save_path}")

        model.load_state_dict(torch.load(model_save_path))
        pred_out = trainer.predict(test_loader)
        G_test, P_test = pred_out[:2]

        if df_test is None:
            df_test = pd.DataFrame(data=G_test, columns=['truth'])
        df_test[f'preds_{seed}'] = P_test

    with open(os.path.join(output_dir, "scaler.pickle"), 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"Saved scaler: {output_dir}/scaler.pickle")

    if len(ensemble_seeds) > 1:
        pred_cols = [c for c in df_test.columns if c.startswith('preds_')]
        df_test['preds'] = df_test[pred_cols].mean(axis=1)
        test_ens_pc   = pearson(df_test['truth'].values, df_test['preds'].values)
        test_ens_rmse = rmse(df_test['truth'].values, df_test['preds'].values)
        print(f"\n{'='*60}")
        print("ENSEMBLE TEST RESULTS")
        print(f"{'='*60}")
        print(f"Ensemble test Pearson correlation: {test_ens_pc:.4f}")
        print(f"Ensemble test RMSE: {test_ens_rmse:.4f}")
        print(f"{'='*60}\n")
    else:
        print("\nSingle-seed mode: skipping ensemble metrics computation")

    return output_dir, test_data


def validate_and_process_csv(config):
    """Validate data, generate graphs, and create PyTorch dataset from CSV."""
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


def run_predictions(test_data, df, config):
    """Make predictions using ensemble of trained models."""
    print("\n" + "="*60)
    print("STEP: MAKE PREDICTIONS")
    print("="*60 + "\n")

    os.environ["OMP_NUM_THREADS"] = str(config.num_workers)
    os.environ["MKL_NUM_THREADS"] = str(config.num_workers)
    torch.set_num_threads(config.num_workers)

    model_dir = f'{Config.TRAINED_MODELS_DIR}/{config.trained_model_name}'
    print(f"Model directory: {model_dir}")

    model_paths = sorted(str(p) for p in Path(model_dir).glob("*.model"))
    scaler_path = f'{model_dir}/scaler.pickle'

    predictor = Predictor(
        model_class=MODEL_REGISTRY[config.model],
        model_paths=model_paths,
        scaler_path=scaler_path,
        device=config.device,
        config=config,
    )

    df_preds = predictor.predict(test_data)
    df = df.merge(df_preds, on='graph_id', how='left')

    return df


def save_results(df, config, total_time, graph_time=None):
    """Save predictions and print summary."""
    print("\n" + "="*60)
    print("STEP: SAVE RESULTS")
    print("="*60 + "\n")

    output_file = (
        f"{Config.PREDICTIONS_DIR}/{config.model}/"
        f"{config.trained_model_name}/{config.data_name}_predictions.parquet"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    pl.from_pandas(df).write_parquet(output_file)
    print(f"Saved predictions to {output_file}")

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total complexes processed: {len(df)}")
    if graph_time is not None:
        print(f"Graph generation time: {graph_time:.2f} seconds")
    print(f"Total pipeline time: {total_time:.2f} seconds")
    print("="*60 + "\n")


def main():
    """Main orchestrator for train+predict or predict-only pipeline."""
    start_time = time.time()

    args = parse_args()

    # Resolve device
    device = get_device(args.device)
    args.device = device
    print(f"Using device: {device}\n")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Resolve num_workers
    if args.num_workers <= 0:
        args.num_workers = os.cpu_count()
        print(f"Using all available cores: {args.num_workers} workers")
    else:
        print(f"Using {args.num_workers} worker(s)")

    if args.predict_only:
        # Predict-only mode
        print("\n" + "="*60)
        print("MODE: PREDICT ONLY")
        print("="*60 + "\n")

        use_processed = args.use_processed or args.dataset_csv is None

        if use_processed:
            print("\n" + "="*60)
            print("MODE: LOADING FROM PROCESSED DATA")
            print("="*60 + "\n")
            test_data, df = load_processed_data(args)
            graph_time = None
        else:
            print("\n" + "="*60)
            print("MODE: PROCESSING NEW DATA FROM CSV")
            print("="*60 + "\n")
            test_data, df, graph_time = validate_and_process_csv(args)

        df = run_predictions(test_data, df, args)
        total_time = time.time() - start_time
        save_results(df, args, total_time, graph_time)

    else:
        # Train + predict mode
        print("\n" + "="*60)
        print("MODE: TRAIN + PREDICT")
        print("="*60 + "\n")

        timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S")
        output_dir, test_data = run_training(args, device, timestamp)

        # Configure prediction to use the freshly trained models
        args.trained_model_name = f'{args.model}_{timestamp}'
        args.data_name = args.dataset

        df = pd.DataFrame({'graph_id': range(len(test_data))})
        df = run_predictions(test_data, df, args)
        total_time = time.time() - start_time
        save_results(df, args, total_time)


if __name__ == "__main__":
    main()
