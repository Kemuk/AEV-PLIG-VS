#!/bin/bash
#SBATCH --job-name=aev-graphs
#SBATCH --cluster=arc
#SBATCH --partition=short
#SBATCH --time=08:00:00
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/graphs_%j.out
#SBATCH --error=logs/graphs_%j.err
# =============================================================================
# Step 1: Generate molecular graphs for all three datasets in parallel.
# Each script reads structures from data/ and writes a .pickle file.
# =============================================================================

source "$(dirname "${BASH_SOURCE[0]}")/../config.sh"

echo "Starting graph generation for 3 datasets in parallel..."

python scripts/generate_pdbbind_graphs.py &
PID_PDBBIND=$!

python scripts/generate_bindingnet_graphs.py &
PID_BINDINGNET=$!

python scripts/generate_bindingdb_graphs.py &
PID_BINDINGDB=$!

echo "PIDs: pdbbind=$PID_PDBBIND bindingnet=$PID_BINDINGNET bindingdb=$PID_BINDINGDB"

# Wait for all three and capture exit codes
FAILED=0

wait $PID_PDBBIND || { echo "FAILED: generate_pdbbind_graphs.py"; FAILED=1; }
wait $PID_BINDINGNET || { echo "FAILED: generate_bindingnet_graphs.py"; FAILED=1; }
wait $PID_BINDINGDB || { echo "FAILED: generate_bindingdb_graphs.py"; FAILED=1; }

if [ $FAILED -ne 0 ]; then
    echo "One or more graph generation jobs failed."
    exit 1
fi

echo "All graph generation completed successfully."
