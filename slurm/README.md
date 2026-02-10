# AEV-PLIG SLURM Workflow Guide

This directory contains scripts for running AEV-PLIG on an HPC cluster using SLURM.

## Quick Start

### Running the Full Training Pipeline

To run the complete training workflow (graph generation → data creation → model training):

```bash
./slurm/submit_training.sh
```

This will submit three jobs that run sequentially:
1. **Job 01**: Generate molecular graphs for all datasets
2. **Job 02**: Create PyTorch datasets from graphs
3. **Job 03**: Train an ensemble of models

### Running Predictions

After training completes, run predictions on new data:

```bash
# First, check the trained model name
ls output/trained_models/

# Then submit prediction job with the model name
TRAINED_MODEL_NAME="model_GATv2Net_ligsim90_fep_benchmark" \
  ./slurm/submit_prediction.sh data/example_dataset.csv my_predictions
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER --cluster=htc

# View output logs (replace JOB_ID with actual job ID)
tail -f logs/graphs_JOB_ID.out
tail -f logs/train_JOB_ID.out
```

## Pipeline Architecture

### Job Flow

```
01_generate_graphs.sh  (8h, 32GB, 16 CPUs)  [htc cluster]
         ↓ (afterok)
02_create_data.sh      (2h, 20GB, 8 CPUs)   [htc cluster]
         ↓ (afterok)
03_train.sh            (24h, 20GB, 8 CPUs, 1 GPU)  [htc cluster]
         ↓ (manual)
04_predict.sh          (4h, 20GB, 8 CPUs, 1 GPU)   [htc cluster]
```

All jobs run on the HTC cluster and use native SLURM `--dependency=afterok:$JOB_ID` to ensure sequential execution. If any job fails, subsequent jobs are automatically cancelled.

### Individual Jobs

#### 01_generate_graphs.sh - Parallel Graph Generation

**What it does:**
- Generates molecular graphs for three datasets: PDBbind, BindingNet, and BindingDB
- **Runs 3 Python scripts in PARALLEL** within a single SLURM job
- Each script processes one dataset independently
- Outputs: `pdbbind.pickle`, `bindingnet.pickle`, `bindingdb.pickle`

**Parallel Processing Details:**
```bash
python scripts/generate_pdbbind_graphs.py &     # Background process 1
python scripts/generate_bindingnet_graphs.py &  # Background process 2
python scripts/generate_bindingdb_graphs.py &   # Background process 3
wait  # Wait for all three to complete
```

This is **NOT** multiple SLURM jobs - it's parallel execution within a single job using shell background processes. This approach:
- Maximizes CPU utilization (uses all 16 allocated CPUs)
- Provides isolated memory per process (automatic cleanup)
- Reduces queue time (one job submission instead of three)
- Simplified dependency management

**Resources:** 32GB RAM (to accommodate 3 parallel processes), 16 CPUs, 8 hours

#### 02_create_data.sh - Dataset Creation

**What it does:**
- Loads all three pickle files
- Creates train/valid/test splits
- Saves PyTorch Geometric datasets to `data/processed/`

**Resources:** 20GB RAM, 8 CPUs, 2 hours

#### 03_train.sh - Model Training

**What it does:**
- Trains an ensemble of models (default: 10 models with different seeds)
- Uses GPU acceleration
- Saves models to `output/trained_models/`
- Includes timestamp in model names

**Resources:** 20GB RAM, 8 CPUs, 1 GPU, 24 hours

#### 04_predict.sh - Inference

**What it does:**
- Loads trained model ensemble
- Makes predictions on new protein-ligand complexes
- Saves results to `output/predictions/`

**Note:** Must be submitted separately after training because the trained model name includes a timestamp that's unknown before training completes.

**Resources:** 20GB RAM, 8 CPUs, 1 GPU, 4 hours

## Submission Scripts

### submit_training.sh

Submit the training pipeline (jobs 01-03):

```bash
./slurm/submit_training.sh
```

**Use when:** You want to train new models from scratch.

### submit_prediction.sh

Submit prediction job only:

```bash
TRAINED_MODEL_NAME="model_name_here" \
  ./slurm/submit_prediction.sh [input.csv] [output_name]
```

**Arguments:**
- `input.csv` - Path to CSV with protein-ligand complexes (default: `data/example_dataset.csv`)
- `output_name` - Name for output files (default: `predictions`)

**Use when:** You have trained models and want to make predictions on new data.

### submit_slurm.sh

Submit complete pipeline (jobs 01-03, prediction must be done separately):

```bash
./slurm/submit_slurm.sh
```

**Use when:** You want to run the full workflow. Note that prediction (job 04) must still be submitted manually after training completes.

## Configuration

All jobs source `config.sh` which sets up:
- Conda environment activation
- CUDA module loading
- Cluster settings (partition names, resource presets)
- Runtime choices (model name, dataset name)

### Customizing Settings

Edit `config.sh` to change:
- `MODEL_NAME` - Model architecture (default: `GATv2NetBayesian`)
- `DATASET_NAME` - Training dataset name

**Note:** Hyperparameters (learning rate, batch size, epochs, etc.) are defined in `aev_plig/config.py` and should NOT be duplicated in `config.sh`. Override them using CLI flags in job scripts if needed.

## Testing

### Environment Validation

Before submitting jobs, validate your setup:

```bash
./slurm/tests/test_environment.sh
```

Checks:
- Conda environment exists
- Required Python packages installed
- CUDA availability
- Data files present

### Interactive Testing

For rapid development, test in an interactive session:

```bash
# Request interactive session
srun --cluster=htc --partition=interactive \
  --mem=20GB --cpus-per-task=4 --gres=gpu:1 \
  --time=04:00:00 --pty bash

# Run local tests
./slurm/tests/test_local.sh
```

This runs quick tests without SLURM submission overhead.

### SLURM Submission Testing

Test job submission to the development queue:

```bash
./slurm/tests/test_slurm.sh
```

Submits jobs to the `devel` partition (10-minute limit) to verify:
- Job submission works
- Scripts have correct syntax
- Dependencies are properly configured

Jobs will likely timeout - this is expected. The purpose is to test submission, not completion.

### Quick Test Jobs

Test individual components with reduced runtime:

```bash
# Test graph generation only (30 minutes)
sbatch --cluster=htc slurm/tests/jobs/01_generate_graphs_quick.sh

# Test training with 5 epochs (2 hours)
sbatch --cluster=htc slurm/tests/jobs/03_train_quick.sh
```

## Monitoring and Troubleshooting

### Check Job Status

```bash
# View all your jobs
squeue -u $USER --cluster=htc

# View specific job
squeue -j JOB_ID --cluster=htc
```

### View Logs

Logs are written to the `logs/` directory:

```bash
# Real-time log viewing
tail -f logs/graphs_JOB_ID.out
tail -f logs/train_JOB_ID.out

# View completed logs
less logs/graphs_JOB_ID.out
```

Error logs have `.err` extension:

```bash
cat logs/graphs_JOB_ID.err
```

### Cancel Jobs

```bash
# Cancel specific job
scancel JOB_ID

# Cancel all your jobs on the cluster
scancel -u $USER --cluster=htc
```

### Common Issues

#### Job Stays in Queue

**Problem:** Job shows status `PD` (pending) for a long time.

**Solutions:**
- Check partition limits: Some partitions may be full
- Verify resource requests are reasonable
- Use `squeue -j JOB_ID --start` to see estimated start time

#### Job Fails Immediately

**Problem:** Job status changes to `FAILED` or `CANCELLED` right after starting.

**Solutions:**
1. Check error log: `cat logs/jobname_JOB_ID.err`
2. Common causes:
   - Module not available: Check `module avail`
   - Conda environment doesn't exist: Run from correct directory
   - Out of memory: Increase `--mem` in job script
   - File not found: Verify data paths

#### GPU Not Available

**Problem:** Training fails with CUDA errors.

**Solutions:**
- Verify `--gres=gpu:1` is set in the job script
- Check GPU is allocated: `nvidia-smi` in the job
- Ensure CUDA module is loaded in `config.sh`

#### Dependency Chain Breaks

**Problem:** Job 02 or 03 doesn't start after Job 01 completes.

**Solutions:**
- Check if previous job completed successfully (status `COMPLETED`, not `FAILED`)
- Verify logs show successful completion
- If a job failed, fix the issue and resubmit from that point

## Advanced Usage

### Resubmitting from a Specific Step

If a job fails, you can restart the pipeline from that step:

```bash
# If graph generation completed but data creation failed:
J2=$(sbatch --cluster=htc --parsable slurm/jobs/02_create_data.sh)
J3=$(sbatch --cluster=htc --parsable --dependency=afterok:$J2 slurm/jobs/03_train.sh)

# If only training failed:
sbatch --cluster=htc slurm/jobs/03_train.sh
```

### Customizing Job Parameters

Override parameters by editing the job script or setting environment variables:

```bash
# Example: Train with different model
MODEL_NAME="GATv2Net" sbatch --cluster=htc slurm/jobs/03_train.sh
```

### Parallel Prediction on Multiple Datasets

Submit multiple prediction jobs in parallel:

```bash
for dataset in dataset1.csv dataset2.csv dataset3.csv; do
  TRAINED_MODEL_NAME="model_name" \
    sbatch --cluster=htc \
    --export=ALL,PREDICT_CSV="$dataset" \
    slurm/jobs/04_predict.sh
done
```

## Resource Usage Guidelines

### Memory Recommendations

- **Graph generation:** 32GB (for 3 parallel processes)
- **Data creation:** 20GB (loads all pickle files)
- **Training:** 20GB (sufficient for batch_size=128)
- **Prediction:** 20GB

### Time Estimates

Actual runtime depends on dataset size and GPU speed. Default time limits:

- **Graph generation:** 8 hours (usually completes in 2-4 hours)
- **Data creation:** 2 hours (usually completes in 10-30 minutes)
- **Training:** 24 hours for 200 epochs (may finish earlier with early stopping)
- **Prediction:** 4 hours (depends on dataset size)

### Partition Selection

- **short** (12h max): Graph generation, data creation, prediction
- **long** (unlimited): Training (needs 24h for 200 epochs)
- **devel** (10 min max): Testing only
- **interactive** (4h max): Development and debugging

## FAQ

**Q: Do I need to submit jobs one at a time?**

No. The submission scripts automatically handle dependencies. When you run `submit_training.sh`, all three jobs are submitted at once with proper dependencies.

**Q: Can I run graph generation for datasets in separate SLURM jobs?**

Yes, but the current setup is optimized for a single job with 3 parallel processes. This reduces queue time and simplifies management. If you need separate jobs, you can modify the script.

**Q: How do I know when training is complete?**

Check job status with `squeue`. When the training job disappears from the queue, check the logs and `output/trained_models/` directory for the saved models.

**Q: Can I use a different model architecture?**

Yes. Set `MODEL_NAME` in `config.sh` or pass it as a command-line argument to the training script. Options include `GATv2Net` and `GATv2NetBayesian`.

**Q: How do I train on a different dataset?**

Modify `DATASET_NAME` in `config.sh` or pass `--dataset` flag to the training script. Ensure the corresponding processed data files exist in `data/processed/`.

## Additional Resources

- **Main documentation:** See `PLAN.md` for detailed SLURM workflow design
- **Python configuration:** See `aev_plig/config.py` for model hyperparameters
- **Package documentation:** See `README_REFACTORED.md` for package usage

## Contact

For issues or questions about the SLURM workflow:
1. Check logs in `logs/` directory
2. Review this README and `PLAN.md`
3. Ensure environment validation passes
