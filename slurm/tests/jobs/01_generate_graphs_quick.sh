#!/bin/bash
#SBATCH --job-name=aev-graphs-quick
#SBATCH --cluster=htc
#SBATCH --partition=short
#SBATCH --time=00:30:00
#SBATCH --mem=20GB
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/graphs_quick_%j.out
#SBATCH --error=logs/graphs_quick_%j.err
# =============================================================================
# Quick test: generate graphs for PDBbind only (smallest dataset).
# Use to verify environment setup and code changes without full pipeline.
# =============================================================================

source "$(dirname "$0")/../../config.sh"

echo "Quick test: generating PDBbind graphs only..."

python scripts/generate_pdbbind_graphs.py

echo "Quick graph generation completed."
