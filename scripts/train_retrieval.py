"""Train a GATv2Net with pairwise margin ranking loss for virtual screening."""
import argparse
import json
import os
import random
import warnings
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader

from aev_plig.config import Config, RetrievalConfig
from aev_plig.datasets import (
    TargetAwareBatchSampler,
    get_target_labels,
    init_weights,
    load_split,
)
from aev_plig.models import get_model
from aev_plig.training import RetrievalTrainer

warnings.filterwarnings("ignore", message="cuaev not installed")


def parse_args():
    p = argparse.ArgumentParser(
        description='Train GATv2Net with pairwise ranking loss for virtual screening'
    )
    p.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (under data/processed/)')
    p.add_argument('--run-name', type=str, required=True,
                   help='Name for this training run')
    p.add_argument('--epochs', type=int, default=RetrievalConfig.NUM_EPOCHS)
    p.add_argument('--batch-size', type=int, default=RetrievalConfig.BATCH_SIZE)
    p.add_argument('--margin', type=float, default=RetrievalConfig.MARGIN)
    p.add_argument('--lr', type=float, default=RetrievalConfig.LEARNING_RATE)
    p.add_argument('--weight-decay', type=float, default=RetrievalConfig.WEIGHT_DECAY)
    p.add_argument('--patience', type=int, default=RetrievalConfig.EARLY_STOPPING_PATIENCE)
    p.add_argument('--complexes-per-target', type=int,
                   default=RetrievalConfig.COMPLEXES_PER_TARGET)
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--model', type=str, default='GATv2Net')
    p.add_argument('--hidden_dim', type=int, default=Config.HIDDEN_DIM)
    p.add_argument('--head', type=int, default=Config.NUM_ATTENTION_HEADS)
    p.add_argument('--num_layers', type=int, default=Config.NUM_GNN_LAYERS)
    p.add_argument('--activation_function', type=str, default=Config.ACTIVATION_FUNCTION)
    p.add_argument('--wandb', action='store_true', help='Log to Weights & Biases')
    p.add_argument('--wandb-project', type=str, default='aev-plig-vs')
    p.add_argument('--wandb-entity', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = Config.get_device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # Load data
    print(f"\nLoading dataset: {args.dataset}")
    train_data = load_split(args.dataset, 'train')
    valid_data = load_split(args.dataset, 'valid')

    # Convert ConcatDataset to list if needed
    if not isinstance(train_data, list):
        train_data = [train_data[i] for i in range(len(train_data))]
    if not isinstance(valid_data, list):
        valid_data = [valid_data[i] for i in range(len(valid_data))]

    print(f"Train: {len(train_data)}  Valid: {len(valid_data)}")

    num_node_features = train_data[0].x.shape[1]
    num_edge_features = train_data[0].edge_attr.shape[1]
    print(f"Node features: {num_node_features}  Edge features: {num_edge_features}")

    # Build target-aware batch sampler
    target_labels = get_target_labels(train_data)
    sampler = TargetAwareBatchSampler(
        target_labels,
        complexes_per_target=args.complexes_per_target,
        batch_size=args.batch_size,
        seed=args.seed,
    )
    train_loader = DataLoader(train_data, batch_sampler=sampler)

    # Create model
    model = get_model(
        args.model,
        node_feature_dim=num_node_features,
        edge_feature_dim=num_edge_features,
        config=args,
    )
    model.apply(init_weights)
    print(f"Model: {args.model}")

    # Output directory
    output_dir = Path('output') / 'trained_models' / args.run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    config_dict = {
        'model': args.model,
        'task': 'retrieval',
        'loss': 'margin_ranking',
        'hidden_dim': args.hidden_dim,
        'head': args.head,
        'activation_function': args.activation_function,
        'num_layers': args.num_layers,
        'node_feature_dim': num_node_features,
        'edge_feature_dim': num_edge_features,
        'dataset': args.dataset,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'margin': args.margin,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'seed': args.seed,
        'complexes_per_target': args.complexes_per_target,
    }
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)

    # WandB
    if args.wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.run_name,
            config=config_dict,
            tags=[args.model, 'retrieval', args.dataset],
        )

    # Train
    trainer = RetrievalTrainer(
        model=model,
        train_loader=train_loader,
        valid_data=valid_data,
        device=device,
        lr=args.lr,
        weight_decay=args.weight_decay,
        margin=args.margin,
    )

    best_bedroc = trainer.fit(
        n_epochs=args.epochs,
        save_path=str(output_dir),
        patience=args.patience,
    )

    print(f"\nTraining complete. Best validation BEDROC: {best_bedroc:.4f}")
    print(f"Model saved to: {output_dir}")

    if args.wandb:
        import wandb
        wandb.run.summary['best_val_bedroc'] = best_bedroc
        wandb.finish()


if __name__ == "__main__":
    main()
