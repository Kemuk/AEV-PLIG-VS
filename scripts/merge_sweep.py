"""Assemble a sweep ensemble from all trained-model folders matching a stem.

Finds every folder in trained_models_dir whose name starts with the given stem
(excluding the stem folder itself), copies them into a new stem folder, writes
a synthetic config.json, then runs aev-plig-predict on the ensemble.

Usage:
    python scripts/merge_sweep.py \\
        --stem occam_03-03_02-00 \\
        [--data_name pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark] \\
        [--trained_models_dir /path/to/trained_models] \\
        [--predictions_dir /path/to/predictions]

If the stem folder already exists, member discovery and copying are skipped and
aev-plig-predict is run immediately.

# TODO: the aev-plig-predict invocation below is a temporary shim; remove once
#       sweep_agent.py is updated to call predict directly after training.
"""
import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from aev_plig.config import Config

DEFAULT_DATA_NAME = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'


def parse_args():
    p = argparse.ArgumentParser(
        description='Assemble a sweep ensemble from folders matching a stem glob'
    )
    p.add_argument('--stem', type=str, required=True,
                   help='Common name stem, e.g. occam_03-03_02-00')
    p.add_argument('--data_name', type=str, default=DEFAULT_DATA_NAME,
                   help=f'Prediction dataset label (default: {DEFAULT_DATA_NAME})')
    p.add_argument('--trained_models_dir', type=str, default=None,
                   help='Override Config.TRAINED_MODELS_DIR')
    p.add_argument('--predictions_dir', type=str, default=None,
                   help='Override Config.PREDICTIONS_DIR')
    return p.parse_args()


def _read_model_name(folder: Path) -> str:
    cfg_path = folder / 'config.json'
    if not cfg_path.exists():
        raise FileNotFoundError(f'No config.json found in member folder: {folder}')
    with open(cfg_path) as f:
        cfg = json.load(f)
    if 'model' not in cfg:
        raise KeyError(f"'model' key missing from config.json in: {folder}")
    return cfg['model']


def _assemble_stem_folder(models_root: Path, stem: str) -> tuple[Path, str, list[str]]:
    """Discover members, copy them, return (stem_dir, model_name, member_names)."""
    stem_dir = models_root / stem

    # Glob for all folders that start with the stem, excluding the stem folder itself.
    members = sorted(
        p for p in models_root.glob(f'{stem}*')
        if p.is_dir() and p.name != stem
    )
    if not members:
        raise FileNotFoundError(
            f'No member folders found matching {models_root / stem!r}*\n'
            f'Expected folders like {stem}_0, {stem}_1, … in {models_root}'
        )

    # Ensure all members agree on architecture.
    model_names = {m.name: _read_model_name(m) for m in members}
    unique_models = set(model_names.values())
    if len(unique_models) > 1:
        detail = '\n'.join(f'  {k}: {v}' for k, v in model_names.items())
        raise ValueError(f'Members disagree on model architecture:\n{detail}')
    model_name = unique_models.pop()

    # Copy each member folder into the stem folder.
    stem_dir.mkdir(parents=True, exist_ok=True)
    print(f'Assembling {len(members)} members into {stem_dir}')
    for src in members:
        dst = stem_dir / src.name
        if dst.exists():
            print(f'  skipping {src.name} (already copied)')
        else:
            shutil.copytree(src, dst)
            print(f'  copied {src.name}')

    return stem_dir, model_name, [m.name for m in members]


def _write_config(stem_dir: Path, model_name: str, members: list[str], data_name: str):
    config_dict = {
        'model':             model_name,
        'is_sweep_ensemble': True,
        'members':           members,
        'data_name':         data_name,
    }
    cfg_path = stem_dir / 'config.json'
    with open(cfg_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    print(f'Saved config.json → {cfg_path}')


def _run_predict(stem: str, data_name: str):
    # TODO: temporary shim — remove once sweep_agent.py is updated to invoke
    #       predict directly after training.
    cmd = ['aev-plig-predict', '--trained_model_name', stem, '--data_name', data_name]
    print(f'Running: {" ".join(cmd)}')
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main():
    args = parse_args()

    models_root = Path(args.trained_models_dir or Config.TRAINED_MODELS_DIR)
    stem_dir    = models_root / args.stem

    if stem_dir.exists():
        print(f'Stem folder already exists: {stem_dir} — skipping assembly.')
        # Read config to confirm it is a valid ensemble before predicting.
        cfg_path = stem_dir / 'config.json'
        if not cfg_path.exists():
            raise FileNotFoundError(
                f'Stem folder exists but has no config.json: {stem_dir}\n'
                f'Delete the folder and re-run, or write config.json manually.'
            )
    else:
        stem_dir, model_name, members = _assemble_stem_folder(models_root, args.stem)
        _write_config(stem_dir, model_name, members, args.data_name)

    _run_predict(args.stem, args.data_name)


if __name__ == '__main__':
    main()