"""
Generate molecular graphs for BindingDB dataset.

This script processes BindingDB protein-ligand complexes and generates
graph representations for use in GNN training.
"""

import pandas as pd
import pickle
from tqdm import tqdm
from rdkit import Chem
import numpy as np

from aev_plig.loaders import compute_aevs
from aev_plig.graphs import create_graph
from aev_plig.config import Config


def main():
    """Generate graphs for BindingDB dataset."""

    # Load data
    df = pd.read_csv("data/bindingdb_processed.csv", index_col=0)
    folder = "data/bindingdb/surflex/"

    print(f"Processing {len(df)} BindingDB complexes")

    # Load atom keys and create atom map
    atom_keys = pd.read_csv(Config.ATOM_KEYS_FILE, sep=",")
    atom_map = pd.DataFrame(pd.unique(atom_keys["ATOM_TYPE"]))
    atom_map[1] = list(np.arange(len(atom_map)) + 1)
    atom_map = atom_map.rename(columns={0: "ATOM_TYPE", 1: "ATOM_NR"})

    # Get radial coefficients from config
    radial_coefs = Config.get_radial_coefs()

    mol_graphs = {}

    # Process each complex
    for index, row in tqdm(df.iterrows(), total=len(df)):
        mol2_file = folder + row["folder"] + "/" + row["mol2_file"]
        lig = Chem.MolFromMol2File(mol2_file)
        lig = Chem.AddHs(lig, addCoords=True)

        protein_path = folder + row["folder"] + "/" + row["pdb_file"]

        # Use package functions
        mol_df, aevs = compute_aevs(protein_path, lig, atom_keys, radial_coefs, atom_map)
        graph = create_graph(lig, mol_df, aevs)
        mol_graphs[row["unique_id"]] = graph

    # Save the graphs
    output_file_graphs = "data/bindingdb.pickle"
    with open(output_file_graphs, 'wb') as handle:
        pickle.dump(mol_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Saved {len(mol_graphs)} graphs to {output_file_graphs}")


if __name__ == "__main__":
    main()
