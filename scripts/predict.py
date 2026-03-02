"""Predict binding affinities using a pre-trained GNN ensemble."""
import argparse
import time
import warnings

import torch

from aev_plig.config import Config
from aev_plig.prediction import load_data, run_predictions, save_results

warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase will not be available")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data will not be available")


def parse_args():
    p = argparse.ArgumentParser(
        description='Predict binding affinities using a pre-trained GNN ensemble'
    )
    p.add_argument('--device',      type=str, default='auto')
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--trained_model_name', type=str,
                   default='model_GATv2Net_ligsim90_fep_benchmark')
    p.add_argument('--dataset_csv',        type=str, default=None)
    p.add_argument('--data_name',          type=str, default='example')
    p.add_argument('--use_processed',      action='store_true')
    p.add_argument('--skip_validation',    action='store_true')
    # Backward compat: ignored if {trained_model_name}/config.json exists
    p.add_argument('--model',               type=str, default=Config.MODEL_NAME)
    p.add_argument('--hidden_dim',          type=int, default=256)
    p.add_argument('--head',                type=int, default=3)
    p.add_argument('--activation_function', type=str, default='leaky_relu')
    p.add_argument('--wandb',         action='store_true', help='Log evaluation to Weights & Biases')
    p.add_argument('--wandb_project', type=str, default='aev-plig-vs')
    p.add_argument('--wandb_entity',  type=str, default=None)
    return p.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    args.device = Config.get_device(args.device)
    if args.num_workers <= 0:
        args.num_workers = Config.get_cpu_count()

    # Pin PyTorch internal thread pools to match SLURM allocation
    torch.set_num_threads(args.num_workers)
    torch.set_num_interop_threads(1)

    test_data, df, graph_time = load_data(args)
    df = run_predictions(test_data, df, args)
    save_results(df, args, time.time() - start_time, graph_time)

    if args.wandb:
        import json
        import wandb
        from pathlib import Path
        from aev_plig.models import MODEL_REGISTRY
        from aev_plig.results import log_evaluation_to_wandb

        model_dir = Path(Config.TRAINED_MODELS_DIR) / args.trained_model_name
        config_path = model_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                model_cfg = json.load(f)
        else:
            model_cfg = {"model": args.model, "dataset": "unknown", "timestamp": "unknown"}

        _model_class = MODEL_REGISTRY.get(model_cfg["model"])
        _tags = [model_cfg["model"], model_cfg.get("dataset", "unknown")]
        if _model_class is not None and _model_class.is_bayesian:
            _tags.append("Bayesian")
        if "MixedPrecision" in model_cfg["model"]:
            _tags.append("MixedPrecision")

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            group=f"{model_cfg['model']}_{model_cfg.get('timestamp', 'unknown')}",
            name="evaluation",
            job_type="evaluation",
            tags=_tags,
        )
        log_evaluation_to_wandb(run, df, model_cfg)

        artifact = wandb.Artifact(
            name=f"model-{model_cfg['model']}_{model_cfg.get('timestamp', 'unknown')}",
            type="model",
            metadata=model_cfg,
        )
        artifact.add_dir(str(model_dir))
        run.log_artifact(artifact)
        run.finish()


if __name__ == "__main__":
    main()
