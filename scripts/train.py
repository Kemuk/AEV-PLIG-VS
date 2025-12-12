"""
Train GNN models for protein-ligand binding affinity prediction.

This script trains an ensemble of models using the refactored training module.
"""

import torch
import random
import time
import os
import pandas as pd
import argparse
import pickle

from torch_geometric.loader import DataLoader
from aev_plig.datasets import GraphDataset, init_weights
from aev_plig.models import get_model
from aev_plig.training import Trainer, pearson, rmse
from aev_plig.config import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train GNN model for binding affinity prediction')
    parser.add_argument('--model', type=str, default='GATv2Net', help='Model name')
    parser.add_argument('--dataset', type=str, default='pdbbind_U_bindingnet_ligsim90', help='Dataset name')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=200, help='Number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--head', type=int, default=3, help='Number of attention heads')
    parser.add_argument('--lr', type=float, default=0.00012291937615434127, help='Learning rate')
    parser.add_argument('--activation_function', type=str, default='leaky_relu', help='Activation function')
    args = parser.parse_args()
    return args


def train_ensemble(args):
    """Train an ensemble of models."""

    print(f'Training {args.model} on {args.dataset} for {args.epochs} epochs')

    # Setup directories
    timestr = time.strftime("%Y%m%d-%H%M%S")
    model_output_dir = Config.TRAINED_MODELS_DIR

    # Load datasets
    train_data = GraphDataset(root='data', dataset=args.dataset + '_train', y_scaler=None)
    valid_data = GraphDataset(root='data', dataset=args.dataset + '_valid', y_scaler=train_data.y_scaler)
    test_data = GraphDataset(root='data', dataset=args.dataset + '_test', y_scaler=train_data.y_scaler)

    print(f"Number of node features: {train_data.num_node_features}")
    print(f"Number of edge features: {train_data.num_edge_features}")

    # Detect device
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f'Device: {device}')

    # Train ensemble
    ensemble_seeds = Config.ENSEMBLE_SEEDS

    for i, seed in enumerate(ensemble_seeds):
        print(f"\n{'='*60}")
        print(f"Training model {i+1}/{len(ensemble_seeds)} with seed {seed}")
        print(f"{'='*60}\n")

        random.seed(seed)
        torch.manual_seed(int(seed))

        # Create data loaders
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

        # Create model
        model = get_model(
            args.model,
            node_feature_dim=train_data.num_node_features,
            edge_feature_dim=train_data.num_edge_features,
            config=args
        )
        model.apply(init_weights)

        # Create trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=valid_loader,
            device=device,
            y_scaler=train_data.y_scaler,
            learning_rate=args.lr
        )

        # Train model
        model_file_name = f"{timestr}_model_{args.model}_{args.dataset}_{i}.model"
        model_save_path = os.path.join(model_output_dir, model_file_name)

        trainer.fit(n_epochs=args.epochs, model_save_path=model_save_path)

        # Load best model and evaluate on test set
        model.load_state_dict(torch.load(model_save_path))
        G_test, P_test = trainer.predict(test_loader)

        if i == 0:
            df_test = pd.DataFrame(data=G_test, columns=['truth'])

        df_test[f'preds_{i}'] = P_test

    # Compute ensemble predictions
    pred_cols = [c for c in df_test.columns if c.startswith('preds_')]
    df_test['preds'] = df_test[pred_cols].mean(axis=1)

    # Save scaler
    scaler_file = f"{timestr}_model_{args.model}_{args.dataset}.pickle"
    scaler_path = os.path.join(model_output_dir, scaler_file)
    with open(scaler_path, 'wb') as f:
        pickle.dump(train_data.y_scaler, f)

    # Compute ensemble metrics
    test_preds = df_test['preds'].values
    test_truth = df_test['truth'].values
    test_ens_pc = pearson(test_truth, test_preds)
    test_ens_rmse = rmse(test_truth, test_preds)

    print(f"\n{'='*60}")
    print("ENSEMBLE TEST RESULTS")
    print(f"{'='*60}")
    print(f"Ensemble test Pearson correlation: {test_ens_pc:.4f}")
    print(f"Ensemble test RMSE: {test_ens_rmse:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    start_time = time.time()

    args = parse_args()
    train_ensemble(args)

    print(f"Total time: {time.time() - start_time:.2f} seconds")
