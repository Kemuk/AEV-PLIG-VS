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
import json
import torch
from pathlib import Path
from collections import ChainMap
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from aev_plig.datasets import create_dataset

# Check for quick test mode (dry run with test split only)
QUICK_TEST = os.getenv('QUICK_TEST', '0') == '1'
CHUNK_SIZE = int(os.getenv('PHASE3_CHUNK_SIZE', '10000'))


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


def write_split_chunks(split_dir, ids, targets, graphs_lookup, chunk_size, scale=False, y_scaler=None):
    """
    Create a split in chunked .pt files and write a JSON manifest.

    Returns:
        tuple[int, StandardScaler | None]: number of graphs written and scaler.
    """
    split_dir.mkdir(parents=True, exist_ok=True)

    total_graphs = 0
    part_files = []
    scaler = y_scaler

    for offset in range(0, len(ids), chunk_size):
        ids_chunk = ids[offset:offset + chunk_size]
        targets_chunk = targets[offset:offset + chunk_size]

        data_chunk, scaler = create_dataset(
            ids_chunk,
            targets_chunk,
            graphs_lookup,
            scale=scale,
            y_scaler=scaler,
        )

        part_name = f"part-{offset // chunk_size:05d}.pt"
        torch.save(data_chunk, split_dir / part_name)
        part_files.append(part_name)
        total_graphs += len(data_chunk)

    manifest = {
        "split": split_dir.name,
        "chunk_size": chunk_size,
        "num_input_rows": len(ids),
        "num_graphs_written": total_graphs,
        "parts": part_files,
    }

    with open(split_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    return total_graphs, scaler


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

    # Use layered lookup to avoid duplicating ~9GB dict with a merge copy
    print(f"\nPreparing graph lookup...")
    graphs_lookup = ChainMap(bindingdb_graphs, bindingnet_graphs, pdbbind_graphs)
    print(f"✓ Total graphs available: {len(graphs_lookup):,}\n")

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
            .filter(pl.col("split") == "test")  # TEST ONLY (use 'split' after aliasing)
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
        print(data.group_by("split").agg(pl.len().alias("count")).sort("split"))

        dataset = 'pdbbind_U_bindingnet_U_bindingdb_ligsim90_fep_benchmark'

    # =========================================================================
    # Create PyTorch Geometric datasets (Phase 3 optimizations in datasets.py)
    # =========================================================================
    print("\n" + "="*70)
    print("PHASE 3: Creating PyTorch Geometric datasets...")
    print("="*70)
    processed_root = Path("data/processed")
    processed_root.mkdir(parents=True, exist_ok=True)
    dataset_root = processed_root / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)

    if QUICK_TEST:
        # QUICK TEST MODE: Only create test dataset
        test_ids = data["unique_id"].to_list()
        test_y = data["pK"].to_list()
        print(f"\nTest set: {len(test_ids)} samples")

        print(f"\n{'─'*70}")
        print('Creating TEST dataset...')
        print(f"{'─'*70}")
        test_dir = dataset_root / "test"
        written_test, test_scaler = write_split_chunks(
            test_dir,
            test_ids,
            test_y,
            graphs_lookup,
            CHUNK_SIZE,
            scale=True,
        )

        scaler_output = dataset_root / "scaler.pickle"
        with open(scaler_output, "wb") as handle:
            pickle.dump(test_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\n" + "="*70)
        print("✓ QUICK TEST COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"  Test:  {written_test:,} graphs")
        print(f"  Output directory: {dataset_root}")
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
        train_dir = dataset_root / "train"
        written_train, y_scaler = write_split_chunks(
            train_dir,
            train_ids,
            train_y,
            graphs_lookup,
            CHUNK_SIZE,
            scale=True,
        )

        print(f"\n{'─'*70}")
        print('Creating VALIDATION dataset...')
        print(f"{'─'*70}")
        valid_dir = dataset_root / "valid"
        written_valid, _ = write_split_chunks(
            valid_dir,
            valid_ids,
            valid_y,
            graphs_lookup,
            CHUNK_SIZE,
            scale=True,
            y_scaler=y_scaler,
        )

        print(f"\n{'─'*70}")
        print('Creating TEST dataset...')
        print(f"{'─'*70}")
        test_dir = dataset_root / "test"
        written_test, _ = write_split_chunks(
            test_dir,
            test_ids,
            test_y,
            graphs_lookup,
            CHUNK_SIZE,
            scale=True,
            y_scaler=y_scaler,
        )
        with open(dataset_root / "scaler.pickle", "wb") as handle:
            pickle.dump(y_scaler, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print("\n" + "="*70)
        print("✓ ALL DATASETS CREATED SUCCESSFULLY!")
        print("="*70)
        print(f"  Train: {written_train:,} graphs")
        print(f"  Valid: {written_valid:,} graphs")
        print(f"  Test:  {written_test:,} graphs")
        print(f"  Total: {written_train + written_valid + written_test:,} graphs")
        print(f"  Output directory: {dataset_root}")
        print("="*70)


if __name__ == "__main__":
    main()
