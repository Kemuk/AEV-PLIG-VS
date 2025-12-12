#!/usr/bin/env python3
"""
Create SDF files from vina_boxes.csv
This is a stub - adapt your existing make_sdf_files.py script

Key changes needed:
1. Remove Slurm-specific code (SLURM_ARRAY_TASK_ID, etc.)
2. Use snakemake.wildcards.shard instead of task_id
3. Use snakemake.params and snakemake.input/output
"""

import sys
import pandas as pd
from pathlib import Path


def make_sdf_files(csv_file, manifest_out, shard_id, shard_total, obabel_bin='obabel'):
    """
    Create SDF files for a specific shard
    
    Args:
        csv_file: Path to vina_boxes.csv
        manifest_out: Path to output manifest CSV for this shard
        shard_id: Current shard index (0-based)
        shard_total: Total number of shards
        obabel_bin: Path to obabel executable
    """
    # Read input CSV
    df = pd.read_csv(csv_file)
    
    # Calculate rows for this shard
    total_rows = len(df)
    rows_per_shard = (total_rows + shard_total - 1) // shard_total
    start_idx = shard_id * rows_per_shard
    end_idx = min(start_idx + rows_per_shard, total_rows)
    
    # Get shard data
    shard_df = df.iloc[start_idx:end_idx]
    
    if len(shard_df) == 0:
        print(f"Shard {shard_id}: no rows to process")
        # Create empty manifest
        pd.DataFrame(columns=['protein', 'ligand_id', 'sdf_path']).to_csv(
            manifest_out, index=False
        )
        return
    
    print(f"Shard {shard_id}: processing rows {start_idx} to {end_idx}")
    
    # TODO: Add your SDF creation logic here
    # This should:
    # 1. Iterate through shard_df
    # 2. Convert each entry to SDF using obabel
    # 3. Save SDF files
    # 4. Create manifest with paths
    
    # Example manifest structure:
    manifest_data = []
    for idx, row in shard_df.iterrows():
        protein = row['protein']
        ligand_id = row['ligand_id']
        sdf_path = f"LIT_PCBA/{protein}/sdfs/{ligand_id}.sdf"
        
        # TODO: Actually create the SDF file here using obabel
        
        manifest_data.append({
            'protein': protein,
            'ligand_id': ligand_id,
            'sdf_path': sdf_path
        })
    
    # Save manifest
    pd.DataFrame(manifest_data).to_csv(manifest_out, index=False)
    print(f"Created manifest: {manifest_out} with {len(manifest_data)} entries")


if __name__ == '__main__':
    # Snakemake provides these variables
    csv_file = snakemake.input.csv
    manifest_out = snakemake.output.manifest_shard
    shard_id = int(snakemake.wildcards.shard)
    shard_total = snakemake.params.shard_total
    obabel_bin = snakemake.params.obabel_bin
    
    try:
        make_sdf_files(csv_file, manifest_out, shard_id, shard_total, obabel_bin)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
