"""
Create PyTorch Geometric datasets from graph pickle files.

This script combines graphs from multiple datasets (PDBbind, BindingNet, BindingDB)
and creates train/valid/test splits for model training.

Optimizations:
- Parallel pickle loading (ThreadPoolExecutor)
- Polars instead of pandas (5-10x faster)
- Progress bars (tqdm) for visibility

Quick Test Mode:
Set environment variable QUICK_TEST=1 to run in dry-run mode with test split only.
Usage: QUICK_TEST=1 python create_pytorch_data.py
"""

import polars as pl
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from aev_plig.datasets import GraphDataset

# Check for quick test mode (dry run with test split only)
QUICK_TEST = os.getenv('QUICK_TEST', '0') == '1'


def load_pickle(path):
    """
    Load a pickle file with progress tracking.

    Shows file size and progress during loading.
    """
    # Get file size
    file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
    filename = os.path.basename(path)

    print(f"Loading {filename} ({file_size:.1f} MB)...")

    with open(path, 'rb') as handle:
        # For very large files, we could implement chunked reading here
        # For now, just load and show completion
        data = pickle.load(handle)
        print(f"  ✓ Loaded {filename}: {len(data):,} graphs")
        return data


def main():
    """
    Load graphs and create PyTorch datasets.
    """

    if QUICK_TEST:
        print("="*70)
        print("🚀 QUICK TEST MODE: Using test split only for dry run")
        print("="*70)
        print()

    # =========================================================================
    # Load graphs in parallel (Phase 1)
    # =========================================================================
    print("="*70)
    print("PHASE 1: Loading graph pickle files in parallel (3 files)...")
    print("="*70)

    pickle_files = [
        "data/pdbbind.pickle",
        "data/bindingnet.pickle",
        "data/bindingdb.pickle"
    ]

    # Calculate total size
    total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in pickle_files)
    print(f"Total pickle size: {total_size_mb:.1f} MB\n")

    with ThreadPoolExecutor(max_workers=3) as executor:
        # Submit all tasks
        futures = [executor.submit(load_pickle, f) for f in pickle_files]

        # Wait for completion with overall progress bar
        results = []
        with tqdm(total=len(futures), desc="Overall progress", unit="file") as pbar:
            for future in futures:
                results.append(future.result())
                pbar.update(1)

    pdbbind_graphs, bindingnet_graphs, bindingdb_graphs = results

    # Merge all graphs into single dictionary
    print(f"\nMerging graphs...")
    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}
    print(f"✓ Total graphs loaded: {len(graphs_dict):,}\n")

    # =========================================================================
    # Process CSV files with Polars (Phase 2)
    # =========================================================================
    print("="*70)
    if QUICK_TEST:
        print("PHASE 2: Processing CSV files with Polars (TEST SPLIT ONLY)...")
    else:
        print("PHASE 2: Processing CSV files with Polars...")
    print("="*70)

    if QUICK_TEST:
        # QUICK TEST MODE: Only test split from pdbbind + small sample from others
        print("Processing pdbbind_processed.csv (test split only)...")
        pdbbind = (
            pl.scan_csv("data/pdbbind_processed.csv")
            .select([
                pl.col("PDB_code").alias("unique_id"),
                pl.col("-logKd/Ki").alias("pK"),
                pl.col("split_core").alias("split"),
                pl.col("max_tanimoto_fep_benchmark")
            ])
            .filter(pl.col("max_tanimoto_fep_benchmark") < 0.9)
            .filter(pl.col("split_core") == "test")  # TEST ONLY
            .select(["unique_id", "pK", "split"])
            .collect()
        )
        print(f"  → {len(pdbbind)} test entries")

        # Take small sample from other datasets for quick test
        print("Processing bindingnet_processed.csv (small sample)...")
        bindingnet = (
            pl.scan_csv("data/bindingnet_processed.csv")
            .select([
                pl.col("unique_identify").alias("unique_id"),
                pl.col("-logAffi").alias("pK"),
                pl.col("max_tanimoto_fep_benchmark")
            ])
            .filter(pl.col("max_tanimoto_fep_benchmark") < 0.9)
            .with_columns(pl.lit("test").alias("split"))
            .select(["unique_id", "pK", "split"])
            .head(50)  # Small sample
            .collect()
        )
        print(f"  → {len(bindingnet)} test entries (limited for quick test)")

        # Combine test data only
        data = pl.concat([pdbbind, bindingnet])
        print(f"\n✓ Total test entries: {len(data)}")
        dataset = 'quick_test'

    else:
        # NORMAL MODE: Full dataset
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
    print("\n" + "="*70)
    print("PHASE 3: Creating PyTorch Geometric datasets (parallel processing)...")
    print("="*70)

    if QUICK_TEST:
        # QUICK TEST MODE: Only create test dataset
        test_ids = data["unique_id"].to_list()
        test_y = data["pK"].to_list()
        print(f"\nTest set: {len(test_ids)} samples")

        print(f"\n{'─'*70}")
        print('Creating TEST dataset...')
        print(f"{'─'*70}")
        test_data = GraphDataset(
            root='data',
            dataset=dataset + '_test',
            ids=test_ids,
            y=test_y,
            graphs_dict=graphs_dict
        )

        print("\n" + "="*70)
        print("✓ QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"  Test:  {len(test_data):,} graphs")
        print(f"  Output: data/processed/{dataset}_test.pt")
        print("="*70)

    else:
        # NORMAL MODE: Create all three datasets
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
        print(f"\n{'─'*70}")
        print('Creating TRAIN dataset...')
        print(f"{'─'*70}")
        train_data = GraphDataset(
            root='data',
            dataset=dataset + '_train',
            ids=train_ids,
            y=train_y,
            graphs_dict=graphs_dict
        )

        print(f"\n{'─'*70}")
        print('Creating VALIDATION dataset...')
        print(f"{'─'*70}")
        valid_data = GraphDataset(
            root='data',
            dataset=dataset + '_valid',
            ids=valid_ids,
            y=valid_y,
            graphs_dict=graphs_dict
        )

        print(f"\n{'─'*70}")
        print('Creating TEST dataset...')
        print(f"{'─'*70}")
        test_data = GraphDataset(
            root='data',
            dataset=dataset + '_test',
            ids=test_ids,
            y=test_y,
            graphs_dict=graphs_dict
        )

        print("\n" + "="*70)
        print("✓ ALL DATASETS CREATED SUCCESSFULLY!")
        print("="*70)
        print(f"  Train: {len(train_data):,} graphs")
        print(f"  Valid: {len(valid_data):,} graphs")
        print(f"  Test:  {len(test_data):,} graphs")
        print(f"  Total: {len(train_data) + len(valid_data) + len(test_data):,} graphs")
        print("="*70)


if __name__ == "__main__":
    main()
