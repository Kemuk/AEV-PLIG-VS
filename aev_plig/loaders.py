"""
Data loading functions for protein and ligand structures.

This module provides functions to parse PDB (protein) and SDF/MOL2 (ligand) files
and compute Atomic Environment Vectors (AEVs) for protein-ligand complexes.
"""

import pandas as pd
import torch
import torchani
import torchani_mod
import qcelemental as qcel
import numpy as np
from biopandas.pdb import PandasPdb


def load_ligand_atoms(mol):
    """
    Load ligand atoms from RDKit molecule object as DataFrame.

    Extracts atom information (index, type, coordinates) for heavy atoms only.

    Args:
        mol: RDKit molecule object with 3D coordinates

    Returns:
        pd.DataFrame: Ligand atoms with columns [ATOM_INDEX, ATOM_TYPE, X, Y, Z]
    """
    atoms = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "H":  # Include only non-hydrogen atoms
            entry = [int(atom.GetIdx())]
            entry.append(str(atom.GetSymbol()))
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            entry.append(float("{0:.4f}".format(pos.x)))
            entry.append(float("{0:.4f}".format(pos.y)))
            entry.append(float("{0:.4f}".format(pos.z)))
            atoms.append(entry)

    df = pd.DataFrame(atoms)
    df.columns = ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]

    return df


def load_protein_atoms(pdb_path, atom_keys):
    """
    Load protein atoms from PDB file as DataFrame (original string parsing version).

    Parses PDB file using string operations and filters by atom keys.
    This is the original implementation used in generate_pdbbind_graphs.py.

    Args:
        pdb_path: Path to PDB file
        atom_keys: DataFrame with PDB atom type mappings

    Returns:
        pd.DataFrame: Protein atoms with columns [ATOM_INDEX, ATOM_TYPE, X, Y, Z]
    """
    prot_atoms = []

    with open(pdb_path) as f:
        for line in f:
            if line[:4] == "ATOM":
                # Include only non-hydrogen atoms
                atom_name = line[12:16].replace(" ", "")
                if (len(atom_name) < 4 and atom_name[0] != "H") or \
                   (len(atom_name) == 4 and atom_name[1] != "H" and atom_name[0] != "H"):
                    prot_atoms.append([
                        int(line[6:11]),
                        line[17:20] + "-" + atom_name,
                        float(line[30:38]),
                        float(line[38:46]),
                        float(line[46:54])
                    ])

    df = pd.DataFrame(prot_atoms, columns=["ATOM_INDEX", "PDB_ATOM", "X", "Y", "Z"])
    df = df.merge(atom_keys, left_on='PDB_ATOM', right_on='PDB_ATOM')[
        ["ATOM_INDEX", "ATOM_TYPE", "X", "Y", "Z"]
    ].sort_values(by="ATOM_INDEX").reset_index(drop=True)

    if list(df["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types. Only supported atom-type pairs are counted.")

    return df


def load_protein_atoms_biopandas(pdb_path, atom_keys):
    """
    Load protein atoms from PDB file using BioPandas (enhanced version).

    Uses BioPandas for more robust PDB parsing with residue validation.
    This is the enhanced implementation used in process_and_predict.py.

    Args:
        pdb_path: Path to PDB file
        atom_keys: DataFrame with PDB atom type mappings (must include RESIDUE column)

    Returns:
        pd.DataFrame: Protein atoms with columns [ATOM_INDEX, ATOM_TYPE, X, Y, Z]

    Raises:
        Exception: If PDB file cannot be parsed
    """
    allowed_residues = atom_keys["RESIDUE"].unique()
    ppdb = PandasPdb().read_pdb(pdb_path)
    protein = ppdb.df['ATOM']

    # Filter out hydrogen atoms and atoms starting with numbers
    protein = protein[~protein["atom_name"].str.startswith("H")]
    protein = protein[~protein["atom_name"].str.startswith(tuple(map(str, range(10))))]

    # Check for unsupported residues
    discard = protein[~protein["residue_name"].isin(allowed_residues)]
    if len(discard) > 0:
        print("WARNING: Protein contains unsupported residues.", pdb_path)
        print("Ignoring following residues:")
        print(discard["residue_name"].unique())

    # Filter to allowed residues only
    protein = protein[protein["residue_name"].isin(allowed_residues)]
    protein["PDB_ATOM"] = protein["residue_name"] + "-" + protein["atom_name"]
    protein = protein[['atom_number', 'PDB_ATOM', 'x_coord', 'y_coord', 'z_coord']].rename(
        columns={"atom_number": "ATOM_INDEX", "x_coord": "X", "y_coord": "Y", "z_coord": "Z"}
    )
    protein = protein.merge(atom_keys, how='left', on='PDB_ATOM').sort_values(
        by="ATOM_INDEX"
    ).reset_index(drop=True)

    # Check for unsupported atom types
    if list(protein["ATOM_TYPE"].isna()).count(True) > 0:
        print("WARNING: Protein contains unsupported atom types.", pdb_path)
        print("Ignoring following atom types:")
        print(protein[protein["ATOM_TYPE"].isna()]["PDB_ATOM"].unique())

    return protein


def compute_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map, use_biopandas=False):
    """
    Compute Atomic Environment Vectors (AEVs) for ligand atoms.

    Uses modified TorchANI to compute AEVs that capture protein-ligand interactions.
    Only radial terms are used (no angular terms).

    Args:
        protein_path: Path to protein PDB file
        mol: RDKit molecule object for ligand
        atom_keys: DataFrame with PDB atom type mappings
        radial_coefs: List of [RcR, EtaR, RsR] radial coefficients
        atom_map: DataFrame mapping ATOM_TYPE to ATOM_NR
        use_biopandas: If True, use BioPandas for PDB parsing (default: False)

    Returns:
        tuple: (ligand_df, aevs_tensor)
            - ligand_df: DataFrame of ligand atoms
            - aevs_tensor: Tensor of AEVs for each ligand atom (shape: [num_atoms, 352])
    """
    # Load protein and ligand structures
    if use_biopandas:
        Target = load_protein_atoms_biopandas(protein_path, atom_keys)
    else:
        Target = load_protein_atoms(protein_path, atom_keys)

    Ligand = load_ligand_atoms(mol)

    # Extract radial coefficients
    RcR = radial_coefs[0]
    EtaR = radial_coefs[1]
    RsR = radial_coefs[2]

    # Angular coefficients (not used in final features, but needed for AEVComputer)
    RcA = 2.0
    Zeta = torch.tensor([1.0])
    TsA = torch.tensor([1.0])
    EtaA = torch.tensor([1.0])
    RsA = torch.tensor([1.0])

    # Reduce size of Target df based on radial cutoff (optimization)
    distance_cutoff = RcR + 0.1
    for coord in ["X", "Y", "Z"]:
        Target = Target[Target[coord] < float(Ligand[coord].max()) + distance_cutoff]
        Target = Target[Target[coord] > float(Ligand[coord].min()) - distance_cutoff]

    Target = Target.merge(atom_map, on='ATOM_TYPE', how='left')

    # Create tensors of atomic numbers and coordinates
    # Ligand atoms are encoded as carbon (atomic number 6)
    mol_len = torch.tensor(len(Ligand))
    atomicnums = np.append(np.ones(mol_len) * 6, Target["ATOM_NR"])
    atomicnums = torch.tensor(atomicnums, dtype=torch.int64).unsqueeze(0)

    coordinates = pd.concat([Ligand[['X', 'Y', 'Z']], Target[['X', 'Y', 'Z']]])
    coordinates = torch.tensor(coordinates.values).unsqueeze(0)

    # Use torchani_mod to calculate AEVs
    atom_symbols = []
    for i in range(1, 23):  # 22 atom types
        atom_symbols.append(qcel.periodictable.to_symbol(i))

    AEVC = torchani_mod.AEVComputer(RcR, RcA, EtaR, RsR, EtaA, Zeta, RsA, TsA, len(atom_symbols))

    SC = torchani.SpeciesConverter(atom_symbols)
    sc = SC((atomicnums, coordinates))

    # Call modified forward method with mol_len to separate ligand from protein
    aev = AEVC.forward((sc.species, sc.coordinates), mol_len)

    # Extract only radial terms (no angular terms)
    n = len(atom_symbols)
    n_rad_sub = len(EtaR) * len(RsR)
    indices = list(np.arange(n * n_rad_sub))

    return Ligand, aev.aevs.squeeze(0)[:mol_len, indices]


def elements_to_atomicnums(elements):
    """
    Convert element symbols to atomic numbers.

    Args:
        elements: List of element symbols (e.g., ['C', 'N', 'O'])

    Returns:
        np.ndarray: Array of atomic numbers
    """
    atomicnums = np.zeros(len(elements), dtype=int)

    for idx, e in enumerate(elements):
        atomicnums[idx] = qcel.periodictable.to_Z(e)

    return atomicnums
