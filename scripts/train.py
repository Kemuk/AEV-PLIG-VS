"""Train one seed (or the full ensemble) of AEV-PLIG GNN models."""
import argparse
import json
import os
import pickle
import random
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from aev_plig.config import Config
from aev_plig.datasets import init_weights, load_split
from aev_plig.models import get_model
from aev_plig.training import Trainer, pearson, rmse

warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def parse_args():
    p = argparse.ArgumentParser(
        description='Train GNN models for binding affinity prediction'
    )
    p.add_argument('--model',               type=str,   default=Config.MODEL_NAME)
    p.add_argument('--hidden_dim',          type=int,   default=Config.HIDDEN_DIM)
    p.add_argument('--head',                type=int,   default=Config.NUM_ATTENTION_HEADS)
    p.add_argument('--activation_function', type=str,   default=Config.ACTIVATION_FUNCTION)
    p.add_argument('--device',              type=str,   default='auto')
    p.add_argument('--num_workers',         type=int,   default=0)
    p.add_argument('--dataset',    type=str,   default='pdbbind_U_bindingnet_ligsim90')
    p.add_argument('--batch_size', type=int,   default=Config.BATCH_SIZE)
    p.add_argument('--epochs',     type=int,   default=Config.NUM_EPOCHS)
    p.add_argument('--lr',         type=float, default=Config.LEARNING_RATE)
    p.add_argument('--seed',       type=int,   default=None,
                   help='Train single model with this seed (for parallel jobs)')
    p.add_argument('--timestamp',  type=str,   default=None,
                   help='Shared timestamp for ensemble output dir (parallel jobs)')
    p.add_argument('--wandb',         action='store_true', help='Log to Weights & Biases')
    p.add_argument('--wandb_project', type=str, default='aev-plig-vs')
    p.add_argument('--wandb_entity',  type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    timestamp = args.timestamp or time.strftime("%Y%m%d_%H%M%S")
    device = Config.get_device(args.device)
    args.device = device
    if args.num_workers <= 0:
        args.num_workers = os.cpu_count()
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    Config.validate_ensemble_seeds()
    ensemble_seeds = [args.seed] if args.seed is not None else Config.ENSEMBLE_SEEDS
    mode = "Single-seed" if args.seed is not None else "Ensemble"
    print(f"\n{mode} mode: training {len(ensemble_seeds)} model(s)")

    output_dir = Path('output') / 'trained_models' / f'{args.model}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}\nTimestamp: {timestamp}\n")

    train_data = load_split(args.dataset, "train")
    valid_data = load_split(args.dataset, "valid")
    test_data  = load_split(args.dataset, "test")

    scaler_path = Path("data/processed") / args.dataset / "scaler.pickle"
    legacy_scaler = Path("data/processed") / f"{args.dataset}_scaler.pickle"
    with open(scaler_path if scaler_path.exists() else legacy_scaler, 'rb') as f:
        y_scaler = pickle.load(f)

    num_node_features = train_data[0].x.shape[1]
    num_edge_features = train_data[0].edge_attr.shape[1]
    print(f"Node features: {num_node_features}  Edge features: {num_edge_features}  Device: {device}")

    # Write config.json — arch + provenance so predict.py needs no arch flags
    config_dict = {
        "model":               args.model,
        "hidden_dim":          args.hidden_dim,
        "head":                args.head,
        "activation_function": args.activation_function,
        "node_feature_dim":    num_node_features,
        "edge_feature_dim":    num_edge_features,
        "dataset":             args.dataset,
        "epochs":              args.epochs,
        "lr":                  args.lr,
        "batch_size":          args.batch_size,
        "timestamp":           timestamp,
    }
    with open(output_dir / "config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)

    df_test = None
    for i, seed in enumerate(ensemble_seeds):
        print(f"\n{'='*60}\nTraining model {i+1}/{len(ensemble_seeds)} with seed {seed}\n{'='*60}\n")

        if args.wandb:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                group=f"{args.model}_{timestamp}",
                name=f"seed_{seed}",
                config={**config_dict, "seed": seed},
                tags=[args.model, args.dataset],
            )

        random.seed(seed)
        torch.manual_seed(int(seed))

        model = get_model(
            args.model,
            node_feature_dim=num_node_features,
            edge_feature_dim=num_edge_features,
            config=args,
        )
        model.apply(init_weights)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False)
        test_loader  = DataLoader(test_data,  batch_size=args.batch_size, shuffle=False)

        trainer = Trainer(
            model=model, train_loader=train_loader, valid_loader=valid_loader,
            device=device, y_scaler=y_scaler, learning_rate=args.lr,
        )

        model_save_path = output_dir / f"model_seed_{seed}.model"
        trainer.fit(n_epochs=args.epochs, model_save_path=str(model_save_path))
        print(f"Saved: {model_save_path}")

        # Test evaluation
        model.load_state_dict(torch.load(model_save_path, weights_only=True))
        pred_out = trainer.predict(test_loader)
        G_test, P_test = pred_out[:2]
        test_pc       = pearson(G_test, P_test)
        test_rmse_val = rmse(G_test, P_test)
        print(f"  Test Pearson: {test_pc:.4f}  RMSE: {test_rmse_val:.4f}")

        if args.wandb:
            import wandb
            abs_res = np.abs(G_test - P_test)
            summary = {
                "best_val_pearson":   trainer.best_pc,
                "test_pearson":       test_pc,
                "test_rmse":          test_rmse_val,
                "test_success_0.5pK": float((abs_res <= 0.5).mean()),
                "test_success_1.0pK": float((abs_res <= 1.0).mean()),
                "test_success_1.5pK": float((abs_res <= 1.5).mean()),
                "test_success_2.0pK": float((abs_res <= 2.0).mean()),
            }
            if len(pred_out) == 3:  # Bayesian: aleatoric calibration
                aleatoric_std = np.sqrt(pred_out[2])
                summary["test_aleatoric_calibration_r"] = float(
                    np.corrcoef(aleatoric_std, abs_res)[0, 1]
                )
            wandb.run.summary.update(summary)
            wandb.finish()

        if df_test is None:
            df_test = pd.DataFrame(data=G_test, columns=['truth'])
        df_test[f'preds_{seed}'] = P_test

    with open(output_dir / "scaler.pickle", 'wb') as f:
        pickle.dump(y_scaler, f)
    print(f"Saved scaler: {output_dir}/scaler.pickle")

    if len(ensemble_seeds) > 1:
        pred_cols = [c for c in df_test.columns if c.startswith('preds_')]
        df_test['preds'] = df_test[pred_cols].mean(axis=1)
        ens_pc   = pearson(df_test['truth'].values, df_test['preds'].values)
        ens_rmse = rmse(df_test['truth'].values, df_test['preds'].values)
        print(f"\n{'='*60}\nENSEMBLE TEST RESULTS\n{'='*60}")
        print(f"Pearson: {ens_pc:.4f}  RMSE: {ens_rmse:.4f}\n{'='*60}\n")
    else:
        print("\nSingle-seed mode: skipping ensemble metrics computation")


if __name__ == "__main__":
    main()
