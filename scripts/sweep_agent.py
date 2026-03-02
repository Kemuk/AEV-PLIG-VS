"""W&B sweep agent entry point for Bayesian HP search.

Each SLURM array task runs this script once (--count 1 is set by the job script).
W&B Bayesian controller assigns each agent a unique HP configuration via wandb.config.

Usage (W&B mode, called by wandb agent):
    python scripts/sweep_agent.py --dataset pdbbind_U_bindingnet_ligsim90

Usage (local smoke-test, no W&B):
    python scripts/sweep_agent.py --no_wandb --dataset pdbbind_U_bindingnet_ligsim90 \\
        --hidden_dim 256 --head 3 --num_layers 5 --lr 0.00012291937615434127 \\
        --weight_decay 0.0 --activation_function leaky_relu --seed 42 --epochs 5
"""
import argparse
import time
import warnings

import torch

warnings.filterwarnings("ignore", message="cuaev not installed")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.ase")
warnings.filterwarnings("ignore", message="Dependency not satisfied, torchani.data")

from aev_plig.config import Config
from aev_plig.training import train_model


def parse_args():
    p = argparse.ArgumentParser(description='W&B sweep agent for AEV-PLIG HP search')

    # ── Runtime args (always from CLI, same for all array tasks) ──────────────
    p.add_argument('--dataset',       type=str, default='pdbbind_U_bindingnet_ligsim90')
    p.add_argument('--device',        type=str, default='auto')
    p.add_argument('--num_workers',   type=int, default=0)
    p.add_argument('--epochs',        type=int, default=Config.NUM_EPOCHS)
    p.add_argument('--batch_size',    type=int, default=Config.BATCH_SIZE)
    p.add_argument('--model',         type=str, default=Config.MODEL_NAME)
    p.add_argument('--wandb_project', type=str, default='aev-plig-vs')
    p.add_argument('--wandb_entity',  type=str, default=None)
    p.add_argument('--no_wandb',      action='store_true',
                   help='Skip W&B; use CLI HP args below for local smoke-testing')

    # ── Runtime args from YAML command: section ────────────────────────────────
    p.add_argument('--base_model_dir',     type=str,   default=None)
    p.add_argument('--max_training_hours', type=float, default=None)
    p.add_argument('--archetype',          type=str,   default=None)

    # ── HP fallback args — only used when --no_wandb is set ───────────────────
    p.add_argument('--hidden_dim',          type=int,   default=Config.HIDDEN_DIM)
    p.add_argument('--head',                type=int,   default=Config.NUM_ATTENTION_HEADS)
    p.add_argument('--num_layers',          type=int,   default=Config.NUM_GNN_LAYERS)
    p.add_argument('--lr',                  type=float, default=Config.LEARNING_RATE)
    p.add_argument('--weight_decay',        type=float, default=Config.WEIGHT_DECAY)
    p.add_argument('--activation_function', type=str,   default=Config.ACTIVATION_FUNCTION)
    p.add_argument('--seed',                type=int,   default=42)

    return p.parse_args()


def main():
    args = parse_args()
    device = Config.get_device(args.device)
    if args.num_workers <= 0:
        args.num_workers = Config.get_cpu_count()

    # Pin PyTorch internal thread pools to match SLURM allocation
    torch.set_num_threads(args.num_workers)
    torch.set_num_interop_threads(1)

    if args.no_wandb:
        hp = args
        run_name = f"local_{time.strftime('%Y%m%d_%H%M%S')}"
        wandb_run = None
    else:
        import wandb

        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
        )

        seed = wandb_run.config.get("seed")

        run_name = (
            f"{args.archetype}_{time.strftime('%d-%m_%H-00')}_{seed}"
            if args.archetype else None
        )

        hp = wandb.config

        wandb.run.name = run_name

    train_model(
        hp,
        dataset=args.dataset,
        device=device,
        num_workers=args.num_workers,
        epochs=args.epochs,
        batch_size=args.batch_size,
        model_type=args.model,
        run_name=run_name,
        wandb_run=wandb_run,
        base_model_dir=args.base_model_dir,
        max_training_hours=args.max_training_hours,
        archetype=args.archetype,
    )


if __name__ == '__main__':
    main()
