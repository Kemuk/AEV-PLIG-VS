#!/usr/bin/env python3
"""
Convert SMILES ligands to PDBQT format using OpenBabel
Parallelized using joblib
"""

import subprocess
import sys
from pathlib import Path
from joblib import Parallel, delayed
import re


def sanitize_id(mol_id, tag, line_num):
    """
    Sanitize molecule ID to create safe filename
    """
    if not mol_id:
        mol_id = f"{tag}{line_num}"
    
    # Remove unsafe characters
    mol_id = re.sub(r'[^A-Za-z0-9._-]', '_', mol_id)
    mol_id = re.sub(r'^[-.]+|_+', '_', mol_id)
    
    if not mol_id:
        mol_id = "X"
    
    return mol_id


def convert_smiles_to_pdbqt(smiles, mol_id, out_path, ph=7.4):
    """
    Convert a single SMILES string to PDBQT format
    """
    # Skip if already exists
    if out_path.exists():
        return f"exists: {out_path}"
    
    # Convert SMILES to PDBQT
    cmd = [
        'obabel',
        f'-:{smiles}',
        '-opdbqt',
        '--gen3d',
        '-p', str(ph),
        '--partialcharge', 'gasteiger',
        '-O', str(out_path)
    ]
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode == 0 and out_path.exists():
            return f"converted: {out_path}"
        else:
            return f"failed: {mol_id} - {result.stderr[:100]}"
    
    except subprocess.TimeoutExpired:
        return f"timeout: {mol_id}"
    except Exception as e:
        return f"error: {mol_id} - {str(e)}"


def process_smiles_file(smi_file, out_dir, tag, ph, n_jobs):
    """
    Process all SMILES in a file
    """
    smi_path = Path(smi_file)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    if not smi_path.exists() or smi_path.stat().st_size == 0:
        print(f"NOTE: Missing or empty {tag}.smi - skipping")
        return []
    
    print(f"Converting {smi_file} -> {out_dir}")
    
    # Parse SMILES file
    tasks = []
    with open(smi_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Parse line: SMILES ID [rest]
            parts = line.split(None, 1)
            if len(parts) == 0:
                continue
            
            smiles = parts[0]
            mol_id = parts[1] if len(parts) > 1 else ""
            
            # Sanitize ID
            clean_id = sanitize_id(mol_id, tag, line_num)
            output_file = out_path / f"{clean_id}.pdbqt"
            
            tasks.append((smiles, clean_id, output_file))
    
    # Process in parallel
    results = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(convert_smiles_to_pdbqt)(smiles, mol_id, out_file, ph)
        for smiles, mol_id, out_file in tasks
    )
    
    # Print summary
    success = sum(1 for r in results if r.startswith('converted'))
    exists = sum(1 for r in results if r.startswith('exists'))
    failed = sum(1 for r in results if r.startswith(('failed', 'error', 'timeout')))
    
    print(f"\nSummary for {tag}:")
    print(f"  Converted: {success}")
    print(f"  Already existed: {exists}")
    print(f"  Failed: {failed}")
    
    return results


if __name__ == '__main__':
    # Snakemake provides these variables
    actives_smi = snakemake.input.actives_smi
    inactives_smi = snakemake.input.inactives_smi
    out_dir = snakemake.params.out_dir
    ph = snakemake.params.ph
    n_jobs = snakemake.threads
    
    try:
        # Process actives
        actives_out = Path(out_dir) / "actives"
        results_actives = process_smiles_file(
            actives_smi, actives_out, "actives", ph, n_jobs
        )
        
        # Process inactives
        inactives_out = Path(out_dir) / "inactives"
        results_inactives = process_smiles_file(
            inactives_smi, inactives_out, "inactives", ph, n_jobs
        )
        
        # Create completion flags
        Path(snakemake.output.actives_flag).touch()
        Path(snakemake.output.inactives_flag).touch()
        
        print(f"\nCompleted conversion for {out_dir}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
