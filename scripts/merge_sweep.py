"""Merge per-member sweep parquets into one ensemble parquet.

After selecting the top-N runs from the W&B dashboard, run predict.py on each
to generate their parquets, then call this script to combine them.

Usage:
    python scripts/merge_sweep.py \\
        --trained_model_names eager-wind-42 gentle-brook-7 ... \\
        --data_name pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark \\
        --output_name GATv2Net_sweep_abc123

Output:
    output/predictions/{model_name}/{output_name}/{data_name}_predictions.parquet
    output/trained_models/{output_name}/config.json  (synthetic, for load_all_predictions)
"""
import argparse
import json
from pathlib import Path

import polars as pl

from aev_plig.config import Config


def parse_args():
    p = argparse.ArgumentParser(description='Merge sweep member parquets into one ensemble')
    p.add_argument('--trained_model_names', nargs='+', required=True,
                   help='W&B run names of sweep members (space-separated)')
    p.add_argument('--data_name', type=str, required=True,
                   help='Prediction dataset label (parquet filename stem)')
    p.add_argument('--output_name', type=str, required=True,
                   help='Name for the merged output directory under trained_models/')
    p.add_argument('--model_name', type=str, default='GATv2Net',
                   help='Architecture name written to synthetic config.json '
                        '(default: GATv2Net)')
    p.add_argument('--predictions_dir', type=str, default=None,
                   help='Override Config.PREDICTIONS_DIR')
    p.add_argument('--trained_models_dir', type=str, default=None,
                   help='Override Config.TRAINED_MODELS_DIR')
    return p.parse_args()


def _parquet_path(pred_root: Path, model_name: str, run_name: str, data_name: str) -> Path:
    return pred_root / model_name / run_name / f'{data_name}_predictions.parquet'


def _read_model_name(models_root: Path, run_name: str, fallback: str) -> str:
    cfg_path = models_root / run_name / 'config.json'
    if cfg_path.exists():
        with open(cfg_path) as f:
            return json.load(f).get('model', fallback)
    return fallback


def main():
    args = parse_args()

    pred_root   = Path(args.predictions_dir   or Config.PREDICTIONS_DIR)
    models_root = Path(args.trained_models_dir or Config.TRAINED_MODELS_DIR)

    names = args.trained_model_names
    if len(names) < 2:
        raise ValueError(f'Need at least 2 members, got {len(names)}')

    # ── Load each member's parquet ─────────────────────────────────────────────
    print(f'Merging {len(names)} members into {args.output_name!r}')
    member_frames = []
    for run_name in names:
        model_name = _read_model_name(models_root, run_name, args.model_name)
        path = _parquet_path(pred_root, model_name, run_name, args.data_name)
        if not path.exists():
            raise FileNotFoundError(
                f'Parquet not found for {run_name!r}: {path}\n'
                f'Run predict.py on this model first.'
            )
        member_frames.append(pl.read_parquet(path))
        print(f'  loaded {path}')

    # ── Build wide DataFrame: unique_id + preds_0, preds_1, … ─────────────────
    # Start with metadata from the first member (everything except preds).
    meta_cols = [c for c in member_frames[0].columns if c != 'preds']
    merged = member_frames[0].select(meta_cols)

    for i, frame in enumerate(member_frames):
        preds_col = frame.select('unique_id', pl.col('preds').alias(f'preds_{i}'))
        merged = merged.join(preds_col, on='unique_id', how='left')

    # ── Ensemble mean ──────────────────────────────────────────────────────────
    pred_cols = [f'preds_{i}' for i in range(len(names))]
    merged = merged.with_columns(
        pl.mean_horizontal(*pred_cols).alias('preds')
    )

    # ── Save parquet ───────────────────────────────────────────────────────────
    out_dir = pred_root / args.model_name / args.output_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'{args.data_name}_predictions.parquet'
    merged.write_parquet(out_path)
    print(f'Saved merged parquet → {out_path}  ({len(merged)} rows, {len(merged.columns)} cols)')

    # ── Synthetic config.json so load_all_predictions works unchanged ──────────
    model_dir = models_root / args.output_name
    model_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {
        'model':             args.model_name,
        'is_sweep_ensemble': True,
        'members':           names,
        'data_name':         args.data_name,
    }
    cfg_path = model_dir / 'config.json'
    with open(cfg_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f'Saved synthetic config.json → {cfg_path}')


if __name__ == '__main__':
    main()
