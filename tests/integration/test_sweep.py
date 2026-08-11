"""
Integration tests for train_model() and merge_sweep.py.

Tests the high-level pipeline function extracted for sweep_agent.py,
using in-memory mock data so no real dataset is required.
"""
import json
import pickle
import tempfile
import shutil
from pathlib import Path

import pytest
import torch
import polars as pl

from torch_geometric.loader import DataLoader

from aev_plig.datasets import GraphDataset
from aev_plig.training import train_model


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def temp_data_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def train_dataset(sample_graphs_dict, sample_labels, temp_data_dir):
    ids = list(sample_graphs_dict.keys())
    return GraphDataset(
        root=str(temp_data_dir / 'train'),
        dataset='sweep_train',
        ids=ids,
        y=sample_labels,
        graphs_dict=sample_graphs_dict,
        y_scaler=None,
    )


@pytest.fixture
def valid_dataset(sample_graphs_dict, sample_labels, train_dataset, temp_data_dir):
    ids = list(sample_graphs_dict.keys())
    return GraphDataset(
        root=str(temp_data_dir / 'valid'),
        dataset='sweep_valid',
        ids=ids,
        y=sample_labels,
        graphs_dict=sample_graphs_dict,
        y_scaler=train_dataset.y_scaler,
    )


# ─────────────────────────────────────────────────────────────────────────────
# train_model() tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestTrainModel:
    """Test the train_model() pipeline function."""

    def _setup_fs(self, tmp_path, dataset_name, y_scaler):
        """Write a real scaler.pickle where train_model() expects it."""
        scaler_dir = tmp_path / 'data' / 'processed' / dataset_name
        scaler_dir.mkdir(parents=True)
        with open(scaler_dir / 'scaler.pickle', 'wb') as f:
            pickle.dump(y_scaler, f)

    def test_writes_config_json_and_checkpoint(
        self, mock_config, train_dataset, valid_dataset, device, tmp_path, monkeypatch
    ):
        """train_model() creates config.json and a model checkpoint."""
        import aev_plig.datasets as ds_module

        dataset_name = 'mock_sweep_dataset'
        self._setup_fs(tmp_path, dataset_name, train_dataset.y_scaler)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ds_module, 'load_split',
            lambda ds, split: train_dataset if split == 'train' else valid_dataset,
        )

        model_dir = train_model(
            mock_config,
            dataset=dataset_name,
            device=device,
            run_name='test_sweep_run',
            epochs=1,
        )

        assert model_dir.exists(), 'output directory was not created'

        cfg_path = model_dir / 'config.json'
        assert cfg_path.exists(), 'config.json was not written'
        cfg = json.loads(cfg_path.read_text())

        assert cfg['model'] == 'GATv2Net'
        assert cfg['hidden_dim'] == 64   # from mock_config
        assert cfg['head'] == 2          # from mock_config
        assert 'num_layers' in cfg
        assert 'lr' in cfg
        assert 'weight_decay' in cfg

        seed = cfg['seed']
        ckpt = model_dir / f'model_seed_{seed}.model'
        assert ckpt.exists(), f'checkpoint not found: {ckpt}'

    def test_namespace_and_wandb_config_compatible(
        self, train_dataset, valid_dataset, device, tmp_path, monkeypatch
    ):
        """Both argparse.Namespace and a plain object work as hp_config."""
        import types
        import aev_plig.datasets as ds_module

        dataset_name = 'mock_compat_dataset'
        self._setup_fs(tmp_path, dataset_name, train_dataset.y_scaler)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ds_module, 'load_split',
            lambda ds, split: train_dataset if split == 'train' else valid_dataset,
        )

        # Simulate wandb.config-like object (supports attribute access, read-only)
        hp = types.SimpleNamespace(
            hidden_dim=64,
            head=2,
            activation_function='leaky_relu',
            num_layers=3,
            lr=0.001,
            weight_decay=0.0,
            seed=7,
        )

        model_dir = train_model(
            hp,
            dataset=dataset_name,
            device=device,
            run_name='test_compat_run',
            epochs=1,
        )

        cfg = json.loads((model_dir / 'config.json').read_text())
        assert cfg['num_layers'] == 3
        assert cfg['seed'] == 7

        ckpt = model_dir / 'model_seed_7.model'
        assert ckpt.exists()

    def test_wandb_run_receives_val_pearson_r(
        self, mock_config, train_dataset, valid_dataset, device, tmp_path, monkeypatch
    ):
        """If wandb_run is provided, val_pearson_r is written to its summary."""
        import types
        import aev_plig.datasets as ds_module

        dataset_name = 'mock_wandb_dataset'
        self._setup_fs(tmp_path, dataset_name, train_dataset.y_scaler)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            ds_module, 'load_split',
            lambda ds, split: train_dataset if split == 'train' else valid_dataset,
        )

        fake_summary = {}

        class FakeRun:
            summary = fake_summary

        train_model(
            mock_config,
            dataset=dataset_name,
            device=device,
            run_name='test_wandb_run',
            epochs=1,
            wandb_run=FakeRun(),
        )

        assert 'val_pearson_r' in fake_summary, 'val_pearson_r not logged to wandb_run.summary'
        assert isinstance(fake_summary['val_pearson_r'], float)


# ─────────────────────────────────────────────────────────────────────────────
# merge_sweep.py tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
class TestMergeSweep:
    """Test the merge_sweep.py logic (imported directly, not as subprocess)."""

    def _write_member_parquet(self, pred_root, model_name, run_name, data_name, n=5):
        """Write a minimal predictions parquet for one sweep member."""
        out_dir = pred_root / model_name / run_name
        out_dir.mkdir(parents=True)
        df = pl.DataFrame({
            'unique_id': [f'mol_{i}' for i in range(n)],
            'pK':        [float(i) for i in range(n)],
            'preds':     [float(i) * 1.1 for i in range(n)],
        })
        df.write_parquet(out_dir / f'{data_name}_predictions.parquet')
        return df

    def _write_member_config(self, models_root, run_name, model_name='GATv2Net'):
        cfg_dir = models_root / run_name
        cfg_dir.mkdir(parents=True)
        (cfg_dir / 'config.json').write_text(
            json.dumps({'model': model_name})
        )

    def test_merge_produces_correct_columns(self, tmp_path):
        """Merged parquet has preds_0, preds_1, preds columns."""
        from scripts.merge_sweep import main as _merge_main
        import sys

        pred_root   = tmp_path / 'predictions'
        models_root = tmp_path / 'trained_models'
        data_name   = 'test_data'
        model_name  = 'GATv2Net'
        runs        = ['run_a', 'run_b']

        for run in runs:
            self._write_member_parquet(pred_root, model_name, run, data_name)
            self._write_member_config(models_root, run, model_name)

        # Invoke main() with patched sys.argv
        sys.argv = [
            'merge_sweep.py',
            '--trained_model_names', *runs,
            '--data_name',           data_name,
            '--output_name',         'merged_test',
            '--model_name',          model_name,
            '--predictions_dir',     str(pred_root),
            '--trained_models_dir',  str(models_root),
        ]
        _merge_main()

        out_path = pred_root / model_name / 'merged_test' / f'{data_name}_predictions.parquet'
        assert out_path.exists()
        result = pl.read_parquet(out_path)

        assert 'preds_0' in result.columns
        assert 'preds_1' in result.columns
        assert 'preds'   in result.columns
        assert 'ensemble_std' not in result.columns, 'ensemble_std should be computed in notebook'

        # preds should be the mean of preds_0 and preds_1
        expected_mean = (result['preds_0'] + result['preds_1']) / 2
        assert (result['preds'] - expected_mean).abs().max() < 1e-5

    def test_merge_writes_synthetic_config_json(self, tmp_path):
        """Merged output_name dir has config.json with model field."""
        from scripts.merge_sweep import main as _merge_main
        import sys

        pred_root   = tmp_path / 'predictions'
        models_root = tmp_path / 'trained_models'
        data_name   = 'test_data'
        model_name  = 'GATv2Net'
        runs        = ['run_x', 'run_y']

        for run in runs:
            self._write_member_parquet(pred_root, model_name, run, data_name)
            self._write_member_config(models_root, run, model_name)

        sys.argv = [
            'merge_sweep.py',
            '--trained_model_names', *runs,
            '--data_name',           data_name,
            '--output_name',         'merged_cfg_test',
            '--model_name',          model_name,
            '--predictions_dir',     str(pred_root),
            '--trained_models_dir',  str(models_root),
        ]
        _merge_main()

        cfg_path = models_root / 'merged_cfg_test' / 'config.json'
        assert cfg_path.exists()
        cfg = json.loads(cfg_path.read_text())
        assert cfg['model'] == model_name
        assert cfg['is_sweep_ensemble'] is True
        assert set(runs) == set(cfg['members'])
