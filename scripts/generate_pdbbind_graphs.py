"""
Generate molecular graphs for PDBbind dataset.

This script processes PDBbind protein-ligand complexes and generates
graph representations for use in GNN training.
"""

import pandas as pd
import pickle
import torch
from tqdm import tqdm
import os
from rdkit import Chem
import numpy as np

from aev_plig.loaders import compute_aevs
from aev_plig.graphs import create_graph
from aev_plig.config import Config


def main():
    """Generate graphs for PDBbind dataset."""

    # Load data
    data = pd.read_csv("data/pdbbind_processed.csv", index_col=0)
    print("The number of data points is", len(data))

    # Load atom keys and create atom map
    atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=",")
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    # Get radial coefficients from config
    radial_coefs = Config.get_radial_coefs()

    mol_graphs = {}
    failed_list = []
    failed_after_reading = []

    # Process each complex
    for i, pdb in tqdm(enumerate(data["PDB_code"])):
        if data["refined"][i]:
            folder = "data/pdbbind/refined-set/"
        else:
            folder = "data/pdbbind/general-set/"

        mol_path = os.path.join(folder, pdb, f'{pdb}_ligand.mol2')
        mol = Chem.MolFromMol2File(mol_path)

        if mol is None:
            print("Can't read molecule structure:", pdb)
            failed_list.append(pdb)
            continue
        else:
            mol = Chem.AddHs(mol, addCoords=True)

        try:
            protein_path = os.path.join(folder, pdb, f'{pdb}_protein.pdb')

            # Use package functions
            mol_df, aevs = compute_aevs(protein_path, mol, atom_keys, radial_coefs, atom_map)
            graph = create_graph(mol, mol_df, aevs)
            mol_graphs[pdb] = graph

        except ValueError as e:
            print(e)
            failed_after_reading.append(pdb)
            continue

    print(f"Failed to read: {len(failed_list)}, Failed after reading: {len(failed_after_reading)}")

    # Save the graphs
    output_file_graphs = "data/pdbbind.pickle"
    with open(output_file_graphs, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(mol_graphs)} graphs to {output_file_graphs}")


if __name__ == "__main__":
    main()
