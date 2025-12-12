#!/usr/bin/env python3
"""
Convert MOL2 files to PDBQT format and remove water molecules
"""

import subprocess
import sys
from pathlib import Path


def strip_waters_pdbqt(in_pdbqt, out_pdbqt):
    """
    Remove water molecules from PDBQT file
    """
    with open(in_pdbqt, 'r') as f_in, open(out_pdbqt, 'w') as f_out:
        for line in f_in:
            if line.startswith(('ATOM', 'HETATM')):
                resn = line[17:20].strip()
                if resn not in ('HOH', 'WAT', 'H2O'):
                    f_out.write(line)
            elif line.startswith(('REMARK', 'TER', 'END', 'MODEL', 'ENDMDL', 'CONECT')):
                f_out.write(line)
    
    # Check for ligand-only tags that shouldn't be in receptor
    with open(out_pdbqt, 'r') as f:
        content = f.read()
        if any(tag in content for tag in ['ROOT', 'BRANCH', 'ENDBRANCH', 'ENDROOT', 'TORSDOF']):
            raise ValueError(f"Error: ligand-only tags remain in receptor PDBQT ({out_pdbqt})")


def convert_mol2_to_pdbqt(mol2_file, pdbqt_file):
    """
    Convert MOL2 to PDBQT using OpenBabel
    """
    mol2_path = Path(mol2_file)
    pdbqt_path = Path(pdbqt_file)
    tmp_path = pdbqt_path.with_suffix('.tmp.pdbqt')
    
    if not mol2_path.exists():
        raise FileNotFoundError(f"Input MOL2 file not found: {mol2_file}")
    
    # Convert MOL2 to PDBQT
    cmd = [
        'obabel',
        '-i', 'mol2', str(mol2_path),
        '-o', 'pdbqt',
        '-O', str(tmp_path)
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"OpenBabel conversion failed: {result.stderr}")
    
    # Strip waters
    strip_waters_pdbqt(tmp_path, pdbqt_path)
    
    # Clean up temp file
    tmp_path.unlink()
    
    print(f"Successfully converted {mol2_file} -> {pdbqt_file}")


if __name__ == '__main__':
    # Snakemake provides these variables
    mol2_file = snakemake.input.mol2
    pdbqt_file = snakemake.output.pdbqt
    
    try:
        convert_mol2_to_pdbqt(mol2_file, pdbqt_file)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
