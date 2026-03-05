"""Evaluate a trained model as a virtual screening ranker.

Works with any trained GATv2Net — affinity (MSE-trained) or ranking
(margin-ranking-trained). Scores all complexes, ranks ligands within each
protein target, and computes enrichment metrics via RDKit Scoring.
"""
import argparse
import json
import os
import warnings
from pathlib import Path

import polars as pl
import torch

from aev_plig.config import Config, RetrievalConfig
from aev_plig.datasets import load_split
from aev_plig.models import MODEL_REGISTRY
from aev_plig.prediction import evaluate_retrieval, predict_retrieval
from aev_plig.results import analyze_false_positives, summarize_diagnostics

warnings.filterwarnings("ignore", message="cuaev not installed")


def parse_args():
    p = argparse.ArgumentParser(
        description='Evaluate a trained model for virtual screening / retrieval'
    )
    p.add_argument('--model-dir', type=str, required=True,
                   help='Path to trained model directory (contains model.pt or *.model + config.json)')
    p.add_argument('--dataset', type=str, required=True,
                   help='Dataset name (under data/processed/)')
    p.add_argument('--split', type=str, default='test',
                   choices=['train', 'valid', 'test'],
                   help='Data split to evaluate (default: test)')
    p.add_argument('--output-dir', type=str, default=None,
                   help='Output directory (default: output/retrieval_results/<model-dir-name>/)')
    p.add_argument('--device', type=str, default='auto')
    p.add_argument('--diagnostics', action='store_true',
                   help='Run false-positive diagnostic analysis')
    return p.parse_args()


def _load_model(model_dir, device):
    """Load model from directory, supporting both .model and .pt checkpoints."""
    model_dir = Path(model_dir)
    config_path = model_dir / 'config.json'

    if not config_path.exists():
        raise FileNotFoundError(f"No config.json found in {model_dir}")

    with open(config_path) as f:
        model_cfg = json.load(f)

    model_name = model_cfg.get('model', Config.MODEL_NAME)
    model_class = MODEL_REGISTRY[model_name]

    # Detect feature dims from config or data
    node_dim = model_cfg.get('node_feature_dim')
    edge_dim = model_cfg.get('edge_feature_dim')

    config_ns = type('Namespace', (), model_cfg)()
    model = model_class(
        node_feature_dim=node_dim,
        edge_feature_dim=edge_dim,
        config=config_ns,
    )

    # Try model.pt first, then *.model files (use first/best checkpoint)
    pt_path = model_dir / 'model.pt'
    if pt_path.exists():
        model.load_state_dict(torch.load(pt_path, map_location=device))
    else:
        model_files = sorted(model_dir.glob('*.model'))
        if not model_files:
            raise FileNotFoundError(f"No model checkpoint found in {model_dir}")
        model.load_state_dict(torch.load(model_files[0], map_location=device))

    model.to(device)
    model.eval()
    return model, model_cfg


def main():
    args = parse_args()
    device = Config.get_device(args.device)

    print(f"\nLoading model from {args.model_dir}")
    model, model_cfg = _load_model(args.model_dir, device)

    print(f"Loading {args.split} data from {args.dataset}")
    data = load_split(args.dataset, args.split)
    # Convert ConcatDataset to list if needed
    if not isinstance(data, list):
        data = [data[i] for i in range(len(data))]

    print(f"Loaded {len(data)} complexes")

    # Output directory
    model_dir_name = Path(args.model_dir).name
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path('output') / 'retrieval_results' / model_dir_name
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run predictions
    print("\nScoring complexes and computing per-protein rankings...")
    predictions_df = predict_retrieval(model, data, device)
    pred_path = output_dir / f'{args.dataset}_{args.split}_predictions.parquet'
    predictions_df.write_parquet(pred_path)
    print(f"Saved predictions to {pred_path}")

    # Compute retrieval metrics
    print("\nComputing retrieval metrics...")
    metrics_df = evaluate_retrieval(model, data, device)
    metrics_path = output_dir / f'{args.dataset}_{args.split}_retrieval_metrics.parquet'
    metrics_df.write_parquet(metrics_path)

    # Summary
    summary = {}
    if metrics_df.height > 0:
        for col in ['bedroc', 'rie']:
            if col in metrics_df.columns:
                summary[f'mean_{col}'] = float(metrics_df[col].mean())
        for col in metrics_df.columns:
            if col.startswith('ef_'):
                summary[f'mean_{col}'] = float(metrics_df[col].mean())

    summary_path = output_dir / f'{args.dataset}_{args.split}_retrieval_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nRetrieval Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v:.4f}")
    print(f"\nSaved metrics to {metrics_path}")
    print(f"Saved summary to {summary_path}")

    # Optional diagnostics
    if args.diagnostics:
        print("\nRunning false-positive diagnostics...")
        # Build sdf_paths from data
        sdf_paths = {}
        for d in data:
            if hasattr(d, 'sdf_file') and hasattr(d, 'unique_id'):
                sdf_paths[d.unique_id] = d.sdf_file

        if sdf_paths:
            fp_df = analyze_false_positives(predictions_df, sdf_paths)
            fp_path = output_dir / f'{args.dataset}_{args.split}_false_positives.parquet'
            fp_df.write_parquet(fp_path)

            diag_summary = summarize_diagnostics(fp_df)
            diag_path = output_dir / f'{args.dataset}_{args.split}_diagnostics_summary.json'
            with open(diag_path, 'w') as f:
                json.dump(diag_summary, f, indent=2)

            print(f"Saved false positives to {fp_path}")
            print(f"Saved diagnostics to {diag_path}")
        else:
            print("Warning: No SDF paths available in data, skipping diagnostics")


if __name__ == "__main__":
    main()
