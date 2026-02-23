"""Predict binding affinities using a pre-trained GNN ensemble."""
import argparse
import os
import time
import warnings

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
    return p.parse_args()


def main():
    start_time = time.time()
    args = parse_args()
    args.device = Config.get_device(args.device)
    if args.num_workers <= 0:
        args.num_workers = os.cpu_count()

    test_data, df, graph_time = load_data(args)
    df = run_predictions(test_data, df, args)
    save_results(df, args, time.time() - start_time, graph_time)


if __name__ == "__main__":
    main()
