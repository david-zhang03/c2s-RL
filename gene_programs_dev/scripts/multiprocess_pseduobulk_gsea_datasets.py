import os
import subprocess
import multiprocessing as mp
import re

# ---------------- CONFIG ----------------
LOCAL_BASE_DIR = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"
REMOTE_H5AD_DIR = "/SAY/standard/HCA-CC1022-InternalMedicine/datasets/HCA_CxG_Processing/hca_cellxgene_step1_colunified_h5ad"
LOCAL_DEST_DIR = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models"
PREP_SCRIPT = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/run_pseduobulk_gsea.py"
GENESET_PATH = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data/msigdb_v2024.1.Hs_GMTs"
PYTHON_ENV = "/gpfs/radev/home/ddz5/.conda/envs/c2gsea_clean/bin/python"
SSH_ALIAS = "transfer"

# ---------------- PROCESS FUNCTION ----------------
def prepare_file_from_transfer(dataset_name):
    local_file = os.path.join(LOCAL_DEST_DIR, f"{dataset_name}_colunified.h5ad")
    remote_file = f"{REMOTE_H5AD_DIR}/{dataset_name}_colunified.h5ad"

    if not os.path.exists(local_file):
        print(f"Copying {remote_file} via ssh on transfer node")
        ssh_cmd = (
            f'ssh {SSH_ALIAS} "cp \\"{remote_file}\\" \\"{local_file}\\""'
        )
        result = subprocess.run(ssh_cmd, shell=True)
        if result.returncode != 0:
            print(f"Failed to copy {dataset_name} from transfer node")
            return None
    else:
        print(f"File already exists: {local_file}")
    return local_file

def process_dataset(dataset_name):
    print(f"\nStarting: {dataset_name}")

    # Step 1: Copy .h5ad file from remote
    local_file = prepare_file_from_transfer(dataset_name)
    if local_file is None:
        return

    # Step 2: Run GSEA preparation script
    output_path = os.path.join(LOCAL_BASE_DIR, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    run_cmd = f"""
    {PYTHON_ENV} {PREP_SCRIPT} \\
        --h5ad_path "{local_file}" \\
        --genesets_path "{GENESET_PATH}" \\
        --dataset_name "{dataset_name}" \\
        --output_prefix "{output_path}"
    """

    print(f"Running GSEA for {dataset_name}")
    exit_code = os.system(run_cmd)
    if exit_code != 0:
        print(f"GSEA failed for {dataset_name}")
        return

    # Step 3: Delete the copied file
    try:
        os.remove(local_file)
        print(f"Deleted {local_file}")
    except Exception as e:
        print(f"Could not delete {local_file}: {e}")

    print(f"Finished {dataset_name}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Find all dataset directories like local(*)
    dataset_names = []
    for d in os.listdir(LOCAL_BASE_DIR):
        match = re.match(r"local\((\d+)\)", d)
        if match:
            dataset_names.append(f"local({match.group(1)})")

    print(f"Found {len(dataset_names)} datasets to process.")

    # Parallel processing
    NUM_CORES = 12 # requesting 12 cpus in slurm script
    print(f"Using {NUM_CORES} cores for parallel processing...\n")

    with mp.Pool(NUM_CORES) as pool:
        pool.map(process_dataset, dataset_names)

    print("\nAll datasets processed.")
