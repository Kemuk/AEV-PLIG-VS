"""
Create PyTorch Geometric datasets from graph pickle files.

This script combines graphs from multiple datasets (PDBbind, BindingNet, BindingDB)
and creates train/valid/test splits for model training.

Optimizations:
- Parallel pickle loading (ThreadPoolExecutor)
- Polars instead of pandas (5-10x faster)
- Progress bars (tqdm) for visibility
"""

import polars as pl
import pickle
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from aev_plig.datasets import GraphDataset


def load_pickle(path):
    """Load a pickle file (for parallel execution)."""
    print(f"Loading {path}...")
    with open(path, 'rb') as handle:
        return pickle.load(handle)


def main():
    """
    Load graphs and create PyTorch datasets.
    """

    # =========================================================================
    # Load graphs in parallel (Phase 1)
    # =========================================================================
    print("="*60)
    print("Loading graph pickle files in parallel...")
    print("="*60)

    pickle_files = [
        "data/pdbbind.pickle",
        "data/bindingnet.pickle",
        "data/bindingdb.pickle"
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(tqdm(
            executor.map(load_pickle, pickle_files),
            total=len(pickle_files),
            desc="Loading pickles"
        ))

    pdbbind_graphs, bindingnet_graphs, bindingdb_graphs = results

    # Merge all graphs into single dictionary
    print(f"\nMerging {len(pdbbind_graphs)} + {len(bindingnet_graphs)} + {len(bindingdb_graphs)} graphs...")
    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}
    print(f"✓ Total graphs: {len(graphs_dict)}\n")

    # =========================================================================
    # Process CSV files with Polars (Phase 2)
    # =========================================================================
    print("="*60)
    print("Processing CSV files with Polars...")
    print("="*60)

    # Load and filter PDBbind (lazy evaluation for speed)
    print("Processing pdbbind_processed.csv...")
    pdbbind = (
        pl.scan_csv("data/pdbbind_processed.csv")
        .select([
            pl.col("PDB_code").alias("unique_id"),
            pl.col("-logKd/Ki").alias("pK"),
            pl.col("split_core").alias("split"),
            pl.col("max_tanimoto_fep_benchmark")
        ])
        .filter(pl.col("max_tanimoto_fep_benchmark") < 0.9)
        .select(["unique_id", "pK", "split"])
        .collect()
    )
    print(f"  → {len(pdbbind)} entries after filtering")

    # Load and filter BindingNet
    print("Processing bindingnet_processed.csv...")
    bindingnet = (
        pl.scan_csv("data/bindingnet_processed.csv")
        .select([
            pl.col("unique_identify").alias("unique_id"),
            pl.col("-logAffi").alias("pK"),
            pl.col("max_tanimoto_fep_benchmark")
        ])
        .filter(pl.col("max_tanimoto_fep_benchmark") < 0.9)
        .with_columns(pl.lit("train").alias("split"))
        .select(["unique_id", "pK", "split"])
        .collect()
    )
    print(f"  → {len(bindingnet)} entries after filtering")

    # Load and filter BindingDB
    print("Processing bindingdb_processed.csv...")
    bindingdb = (
        pl.scan_csv("data/bindingdb_processed.csv")
        .select([
            pl.col("unique_id"),
            pl.col("pK"),
            pl.col("max_tanimoto_fep_benchmark")
        ])
        .filter(pl.col("max_tanimoto_fep_benchmark") < 0.9)
        .with_columns(pl.lit("train").alias("split"))
        .select(["unique_id", "pK", "split"])
        .collect()
    )
    print(f"  → {len(bindingdb)} entries after filtering")

    # Combine all datasets
    data = pl.concat([pdbbind, bindingnet, bindingdb])
    print(f"\n✓ Total combined entries: {len(data)}")
    print("\nSplit distribution:")
    print(data.group_by("split").agg(pl.count()).sort("split"))

    dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'

    # =========================================================================
    # Create PyTorch Geometric datasets (Phase 3 optimizations in datasets.py)
    # =========================================================================
    print("\n" + "="*60)
    print("Creating PyTorch Geometric datasets...")
    print("="*60)

    # Extract train/valid/test splits
    train_df = data.filter(pl.col("split") == "train")
    train_ids = train_df["unique_id"].to_list()
    train_y = train_df["pK"].to_list()
    print(f"\nTrain set: {len(train_ids)} samples")

    valid_df = data.filter(pl.col("split") == "valid")
    valid_ids = valid_df["unique_id"].to_list()
    valid_y = valid_df["pK"].to_list()
    print(f"Valid set: {len(valid_ids)} samples")

    test_df = data.filter(pl.col("split") == "test")
    test_ids = test_df["unique_id"].to_list()
    test_y = test_df["pK"].to_list()
    print(f"Test set:  {len(test_ids)} samples")

    # Create PyTorch Geometric datasets (with progress tracking)
    print(f"\n{'='*60}")
    print('Creating train dataset...')
    print(f"{'='*60}")
    train_data = GraphDataset(
        root='data',
        dataset=dataset + '_train',
        ids=train_ids,
        y=train_y,
        graphs_dict=graphs_dict
    )

    print(f"\n{'='*60}")
    print('Creating validation dataset...')
    print(f"{'='*60}")
    valid_data = GraphDataset(
        root='data',
        dataset=dataset + '_valid',
        ids=valid_ids,
        y=valid_y,
        graphs_dict=graphs_dict
    )

    print(f"\n{'='*60}")
    print('Creating test dataset...')
    print(f"{'='*60}")
    test_data = GraphDataset(
        root='data',
        dataset=dataset + '_test',
        ids=test_ids,
        y=test_y,
        graphs_dict=graphs_dict
    )

    print("\n" + "="*60)
    print("✓ All datasets created successfully!")
    print("="*60)
    print(f"Train: {len(train_data)} graphs")
    print(f"Valid: {len(valid_data)} graphs")
    print(f"Test:  {len(test_data)} graphs")
    print("="*60)


if __name__ == "__main__":
    main()
