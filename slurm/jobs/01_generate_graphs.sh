#!/bin/bash
#SBATCH --job-name=aev-graphs
#SBATCH --cluster=htc
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/graphs_%j.out
#SBATCH --error=logs/graphs_%j.err
#SBATCH --chdir=$DATA/AEV-PLIG-VS
# =============================================================================
# Step 1: Generate molecular graphs for all three datasets sequentially.
# Each script reads structures from data/ and writes a .pickle file.
# =============================================================================

source slurm/config.sh

echo "Starting graph generation for 3 datasets sequentially..."

python scripts/generate_pdbbind_graphs.py || exit 1
echo "✓ PDBbind graphs complete"

python scripts/generate_bindingnet_graphs.py || exit 1
echo "✓ BindingNet graphs complete"

python scripts/generate_bindingdb_graphs.py || exit 1
echo "✓ BindingDB graphs complete"

echo "All graph generation completed successfully."
