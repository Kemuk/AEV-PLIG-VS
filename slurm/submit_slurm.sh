#!/bin/bash
# =============================================================================
# Submit full pipeline: graphs → data → train → predict
# Usage: TRAINED_MODEL_NAME will be unknown until training completes.
#        Prediction uses example dataset as a smoke test.
#
#   ./slurm/submit_slurm.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Function to wait for a job to complete on a specific cluster
wait_for_job() {
    local cluster=$1
    local job_id=$2
    local timeout=${3:-36000}  # Default 10 hours
    local elapsed=0
    local interval=30
    
    echo "Waiting for job $job_id on $cluster cluster to complete..."
    
    while [[ $elapsed -lt $timeout ]]; do
        # Query job state (suppress errors if job no longer exists)
        STATE=$(squeue --cluster="$cluster" -j "$job_id" -h -o "%T" 2>/dev/null || echo "NOT_FOUND")
        
        if [[ "$STATE" == "NOT_FOUND" ]] || [[ -z "$STATE" ]]; then
            # Job not in queue = completed successfully (or check sacct for final state)
            FINAL_STATE=$(sacct --cluster="$cluster" -j "$job_id" -n -o State -P 2>/dev/null | head -1 || echo "COMPLETED")
            if [[ "$FINAL_STATE" =~ ^(COMPLETED|COMPLETING)$ ]]; then
                echo "✓ Job $job_id completed successfully"
                return 0
            elif [[ "$FINAL_STATE" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$ ]]; then
                echo "✗ ERROR: Job $job_id failed with state: $FINAL_STATE"
                return 1
            else
                # Assume completed if not in queue and no error state found
                echo "✓ Job $job_id completed (state: $FINAL_STATE)"
                return 0
            fi
        elif [[ "$STATE" =~ ^(FAILED|CANCELLED|TIMEOUT|NODE_FAIL)$ ]]; then
            echo "✗ ERROR: Job $job_id failed with state: $STATE"
            return 1
        fi
        
        # Job still running
        sleep "$interval"
        elapsed=$((elapsed + interval))
        
        # Progress indicator every 5 minutes
        if [[ $((elapsed % 300)) -eq 0 ]]; then
            echo "  Still waiting... (${elapsed}s elapsed, state: $STATE)"
        fi
    done
    
    echo "✗ ERROR: Timeout waiting for job $job_id (${timeout}s)"
    return 1
}

echo "Submitting full pipeline (arc/htc clusters)..."

# Submit job 1 to arc cluster
J1=$(sbatch --cluster=arc --parsable "$SCRIPT_DIR/jobs/01_generate_graphs.sh" | cut -d';' -f1)
echo "  01_generate_graphs: $J1 (arc)"

# Wait for job 1 to complete before submitting job 2 to different cluster
wait_for_job arc "$J1" || exit 1

# Submit job 2 to htc cluster (no dependency needed, J1 already completed)
J2=$(sbatch --cluster=htc --parsable "$SCRIPT_DIR/jobs/02_create_data.sh" | cut -d';' -f1)
echo "  02_create_data:     $J2 (htc)"

# Submit job 3 to htc cluster (native dependency works - same cluster)
J3=$(sbatch --cluster=htc --parsable --dependency=afterok:"$J2" "$SCRIPT_DIR/jobs/03_train.sh" | cut -d';' -f1)
echo "  03_train:           $J3 (htc, after $J2)"

echo ""
echo "Training pipeline submitted (jobs 01-03)."
echo ""
echo "NOTE: Prediction (job 04) must be submitted separately after training"
echo "completes, because the model name includes a timestamp."
echo "Once training finishes, check the model name:"
echo "  ls output/trained_models/"
echo "Then submit prediction:"
echo "  TRAINED_MODEL_NAME=\"<name>\" ./slurm/submit_prediction.sh"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER --cluster=htc  # for data creation and training"
