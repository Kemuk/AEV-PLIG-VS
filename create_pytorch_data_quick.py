"""
DRY RUN: Quick test using pre-existing TEST split subset.

Runs all 3 optimization phases on the smallest pre-existing subset (test split).
Tests entire pipeline end-to-end with ~5-10K graphs instead of ~95K.

Expected runtime: ~2-5 minutes
"""

import polars as pl
import pickle
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from aev_plig.datasets import GraphDataset


def load_pickle(path):
    """Load a pickle file with progress tracking."""
    file_size = os.path.getsize(path) / (1024 * 1024)  # Convert to MB
    filename = os.path.basename(path)
    print(f"Loading {filename} ({file_size:.1f} MB)...")
    with open(path, 'rb') as handle:
        data = pickle.load(handle)
        print(f"  ✓ Loaded {filename}: {len(data):,} graphs")
        return data


def main():
    """Quick dry run using test split only."""

    print("="*70)
    print("DRY RUN: Testing pipeline with TEST SPLIT ONLY")
    print("="*70)

    # =========================================================================
    # PHASE 1: Load all pickle files (needed for graph lookup)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 1: Loading graph pickle files in parallel (3 files)...")
    print("="*70)

    pickle_files = [
        "data/pdbbind.pickle",
        "data/bindingnet.pickle",
        "data/bindingdb.pickle"
    ]

    total_size_mb = sum(os.path.getsize(f) / (1024 * 1024) for f in pickle_files)
    print(f"Total pickle size: {total_size_mb:.1f} MB\n")

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(load_pickle, f) for f in pickle_files]
        results = []
        with tqdm(total=len(futures), desc="Overall progress", unit="file") as pbar:
            for future in futures:
                results.append(future.result())
                pbar.update(1)

    pdbbind_graphs, bindingnet_graphs, bindingdb_graphs = results
    graphs_dict = {**pdbbind_graphs, **bindingnet_graphs, **bindingdb_graphs}
    print(f"\n✓ Total graphs loaded: {len(graphs_dict):,}\n")

    # =========================================================================
    # PHASE 2: Process CSV files with Polars (TEST SPLIT ONLY)
    # =========================================================================
    print("="*70)
    print("PHASE 2: Processing CSV files with Polars (TEST SPLIT ONLY)...")
    print("="*70)

    # Load and filter PDBbind - TEST SPLIT ONLY
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

    # BindingNet - TEST SPLIT ONLY (assign as test)
    print("Processing bindingnet_processed.csv (treated as test)...")
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
        .head(100)  # Take first 100 for quick test
        .collect()
    )
    print(f"  → {len(bindingnet)} test entries (limited to 100)")

    # Combine test data only
    data = pl.concat([pdbbind, bindingnet])
    print(f"\n✓ Total test entries: {len(data)}")

    dataset = 'quick_test'

    # =========================================================================
    # PHASE 3: Create PyTorch dataset (TEST SPLIT ONLY)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: Creating PyTorch dataset (parallel processing)...")
    print("="*70)

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
    print("✓ DRY RUN COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"  Test: {len(test_data):,} graphs")
    print(f"  Output: data/processed/{dataset}_test.pt")
    print("="*70)
    print("\nAll 3 phases tested successfully with test subset!")
    print("="*70)


if __name__ == "__main__":
    main()
