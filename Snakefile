"""
Molecular Docking and Prediction Pipeline
Snakemake workflow replacing Slurm-based pipeline
"""

import os
import pandas as pd
from pathlib import Path

# Configuration
configfile: "config.yaml"

# Global variables
WORK_DIR = config.get("work_dir", ".")
DATA_DIR = f"{WORK_DIR}/LIT_PCBA"
VINA_BOXES = f"{DATA_DIR}/vina_boxes.csv"

# Load protein directories
PROTEIN_DIRS = []
if os.path.exists(DATA_DIR):
    PROTEIN_DIRS = [d for d in os.listdir(DATA_DIR) 
                    if os.path.isdir(os.path.join(DATA_DIR, d))]

# Number of shards for parallel processing
N_SHARDS = config.get("n_shards", 100)
SHARD_IDS = list(range(N_SHARDS))


# ============================================================================
# MAIN WORKFLOW RULES
# ============================================================================

rule all:
    """
    Default target: run complete pipeline
    """
    input:
        # Data preparation outputs
        expand("{data_dir}/{protein}/receptor.pdbqt", 
               data_dir=DATA_DIR, protein=PROTEIN_DIRS),
        expand("{data_dir}/{protein}/pdbqt/actives/done.flag",
               data_dir=DATA_DIR, protein=PROTEIN_DIRS),
        expand("{data_dir}/{protein}/pdbqt/inactives/done.flag",
               data_dir=DATA_DIR, protein=PROTEIN_DIRS),
        
        # Manifest files
        f"{DATA_DIR}/sdf_manifest.csv",
        f"{DATA_DIR}/aev_plig.csv",
        
        # Docking outputs (optional - uncomment if running docking)
        # f"{DATA_DIR}/docking_results/cpu_done.flag",
        # f"{DATA_DIR}/docking_results/gpu_done.flag",
        
        # Prediction outputs
        "output/predictions/LIT_PCBA_predictions.csv",
        "output/LIT_PCBA_predictions.csv"


# ============================================================================
# DATA PREPARATION RULES
# ============================================================================

rule mol2_to_pdbqt:
    """
    Convert MOL2 receptor files to PDBQT format and remove waters
    """
    input:
        mol2 = "{data_dir}/{protein}/receptor.mol2"
    output:
        pdbqt = "{data_dir}/{protein}/receptor.pdbqt"
    conda:
        "envs/vscreen.yaml"
    threads: 1
    resources:
        mem_mb = 4000
    log:
        "logs/mol2_to_pdbqt/{protein}.log"
    script:
        "scripts/mol2_to_pdbqt.py"


rule smiles_to_pdbqt:
    """
    Convert SMILES ligands to PDBQT format for one protein
    """
    input:
        actives_smi = "{data_dir}/{protein}/actives.smi",
        inactives_smi = "{data_dir}/{protein}/inactives.smi"
    output:
        actives_flag = "{data_dir}/{protein}/pdbqt/actives/done.flag",
        inactives_flag = "{data_dir}/{protein}/pdbqt/inactives/done.flag"
    params:
        out_dir = "{data_dir}/{protein}/pdbqt",
        ph = config.get("ph", 7.4)
    conda:
        "envs/vscreen.yaml"
    threads: config.get("cpus_per_conversion", 16)
    resources:
        mem_mb = 16000
    log:
        "logs/smiles_to_pdbqt/{protein}.log"
    script:
        "scripts/smiles_to_pdbqt.py"


rule create_sdf_files:
    """
    Create SDF files from vina_boxes.csv (parallelized by shard)
    """
    input:
        csv = VINA_BOXES
    output:
        manifest_shard = temp(f"{DATA_DIR}/sdf_manifest_shard_{{shard}}.csv")
    params:
        shard_total = N_SHARDS,
        obabel_bin = "obabel"
    conda:
        "envs/vscreen.yaml"
    threads: config.get("cpus_per_task", 4)
    resources:
        mem_mb = 8000
    log:
        "logs/create_sdf/{shard}.log"
    script:
        "scripts/make_sdf_files.py"


rule merge_sdf_manifests:
    """
    Merge all SDF manifest shards into master manifest
    """
    input:
        shards = expand(f"{DATA_DIR}/sdf_manifest_shard_{{shard}}.csv", 
                       shard=SHARD_IDS)
    output:
        manifest = f"{DATA_DIR}/sdf_manifest.csv"
    log:
        "logs/merge_sdf_manifests.log"
    run:
        import pandas as pd
        
        dfs = []
        for shard_file in input.shards:
            if os.path.exists(shard_file) and os.path.getsize(shard_file) > 0:
                df = pd.read_csv(shard_file)
                dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result.to_csv(output.manifest, index=False)
        else:
            # Create empty manifest with proper columns
            pd.DataFrame(columns=['protein', 'ligand_id', 'sdf_path']).to_csv(
                output.manifest, index=False)


# ============================================================================
# MANIFEST CREATION RULES
# ============================================================================

rule create_plig_manifest:
    """
    Create AEV-PLIG manifest for predictions
    """
    input:
        vina_boxes = VINA_BOXES
    output:
        manifest = f"{DATA_DIR}/aev_plig.csv"
    params:
        shard_total = N_SHARDS
    conda:
        "envs/vscreen.yaml"
    threads: 16
    resources:
        mem_mb = 32000
    log:
        "logs/create_plig_manifest.log"
    shell:
        """
        python scripts/make_aev_plig.py \
            --csv {input.vina_boxes} \
            --out {output.manifest} \
            --shard_total {params.shard_total} \
            2>&1 | tee {log}
        """


# ============================================================================
# DOCKING RULES (Optional)
# ============================================================================

rule docking_cpu:
    """
    Run AutoDock Vina docking on CPU (parallelized by shard)
    """
    input:
        csv = VINA_BOXES
    output:
        flag = temp(f"{DATA_DIR}/docking_results/cpu_shard_{{shard}}.flag")
    params:
        cpu_per_dock = config.get("cpu_per_dock", 8),
        shard_total = N_SHARDS
    conda:
        "envs/vscreen.yaml"
    threads: config.get("cpus_per_dock_task", 32)
    resources:
        mem_mb = 64000
    log:
        "logs/docking_cpu/{shard}.log"
    shell:
        """
        python scripts/dock.py \
            --csv {input.csv} \
            --vina_bin vina \
            --mode cpu \
            --cpu {params.cpu_per_dock} \
            --workers 0 \
            --shard_total {params.shard_total} \
            --shard_idx {wildcards.shard} \
            --skip_existing \
            2>&1 | tee {log}
        
        touch {output.flag}
        """


rule docking_gpu:
    """
    Run QuickVina2-GPU docking (parallelized by shard)
    """
    input:
        master_manifest = f"{DATA_DIR}/master_manifest.csv"
    output:
        flag = temp(f"{DATA_DIR}/docking_results/gpu_shard_{{shard}}.flag")
    params:
        vina_bin = config.get("vina_gpu_bin", "vina-gpu-dev/QuickVina2-GPU-2-1")
    conda:
        "envs/vscreen.yaml"
    threads: 2
    resources:
        mem_mb = 20000,
        gpu = 1
    log:
        "logs/docking_gpu/{shard}.log"
    shell:
        """
        python scripts/dock.py \
            --master_manifest {input.master_manifest} \
            --task_id {wildcards.shard} \
            --mode gpu \
            --vina_bin {params.vina_bin} \
            --gpu_id 0 \
            --runtime_workers {threads} \
            2>&1 | tee {log}
        
        touch {output.flag}
        """


rule merge_docking_results_cpu:
    """
    Wait for all CPU docking shards to complete
    """
    input:
        flags = expand(f"{DATA_DIR}/docking_results/cpu_shard_{{shard}}.flag",
                      shard=SHARD_IDS)
    output:
        flag = f"{DATA_DIR}/docking_results/cpu_done.flag"
    shell:
        "touch {output.flag}"


rule merge_docking_results_gpu:
    """
    Wait for all GPU docking shards to complete
    """
    input:
        flags = expand(f"{DATA_DIR}/docking_results/gpu_shard_{{shard}}.flag",
                      shard=SHARD_IDS)
    output:
        flag = f"{DATA_DIR}/docking_results/gpu_done.flag"
    shell:
        "touch {output.flag}"


# ============================================================================
# PREDICTION RULES
# ============================================================================

rule plig_predictions:
    """
    Run AEV-PLIG predictions (parallelized by shard)
    """
    input:
        dataset_csv = "../dataset.csv"
    output:
        predictions = temp("output/predictions/LIT_PCBA_{shard}_predictions.csv")
    params:
        data_name = "LIT_PCBA_{shard}",
        model_name = config.get("model_name", "model_GATv2Net_ligsim90_fep_benchmark")
    conda:
        "envs/aev_plig.yaml"
    threads: config.get("cpus_per_task", 8)
    resources:
        mem_mb = 20000,
        gpu = 1
    log:
        "logs/plig_predictions/{shard}.log"
    script:
        "scripts/process_and_predict.py"


rule merge_plig_predictions:
    """
    Merge all PLIG prediction shards
    """
    input:
        predictions = expand("output/predictions/LIT_PCBA_{shard}_predictions.csv",
                           shard=SHARD_IDS)
    output:
        merged = "output/predictions/LIT_PCBA_predictions.csv"
    log:
        "logs/merge_plig_predictions.log"
    run:
        import pandas as pd
        
        dfs = []
        for pred_file in input.predictions:
            if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
                df = pd.read_csv(pred_file)
                dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result.to_csv(output.merged, index=False)
            print(f"Merged {len(dfs)} prediction files into {output.merged}")
        else:
            print("No prediction files to merge")


rule ligand_based_predictions:
    """
    Run ligand-based predictions (parallelized by shard)
    """
    input:
        manifest = f"{DATA_DIR}/manifest.csv"
    output:
        predictions = temp("output/LIT_PCBA_{shard}_predictions.csv")
    params:
        n_jobs = config.get("cpus_per_task", 4)
    conda:
        "envs/vscreen.yaml"
    threads: config.get("cpus_per_task", 8)
    resources:
        mem_mb = 20000
    log:
        "logs/ligand_based/{shard}.log"
    script:
        "scripts/worker_stream.py"


rule merge_ligand_based_predictions:
    """
    Merge all ligand-based prediction shards
    """
    input:
        predictions = expand("output/LIT_PCBA_{shard}_predictions.csv",
                           shard=SHARD_IDS)
    output:
        merged = "output/LIT_PCBA_predictions.csv"
    log:
        "logs/merge_ligand_based_predictions.log"
    run:
        import pandas as pd
        
        dfs = []
        for pred_file in input.predictions:
            if os.path.exists(pred_file) and os.path.getsize(pred_file) > 0:
                df = pd.read_csv(pred_file)
                dfs.append(df)
        
        if dfs:
            result = pd.concat(dfs, ignore_index=True)
            result.to_csv(output.merged, index=False)
            print(f"Merged {len(dfs)} prediction files into {output.merged}")
        else:
            print("No prediction files to merge")


# ============================================================================
# UTILITY RULES
# ============================================================================

rule clean:
    """
    Clean up intermediate files
    """
    shell:
        """
        rm -rf logs/
        rm -rf output/predictions/LIT_PCBA_*_predictions.csv
        rm -rf {DATA_DIR}/sdf_manifest_shard_*.csv
        rm -rf {DATA_DIR}/docking_results/*_shard_*.flag
        """


rule clean_all:
    """
    Clean up all generated files
    """
    shell:
        """
        rm -rf logs/
        rm -rf output/
        rm -rf {DATA_DIR}/*/pdbqt/
        rm -rf {DATA_DIR}/*.pdbqt
        rm -rf {DATA_DIR}/*_manifest.csv
        rm -rf {DATA_DIR}/docking_results/
        """
