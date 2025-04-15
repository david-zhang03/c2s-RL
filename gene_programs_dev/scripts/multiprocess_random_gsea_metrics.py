import os
import re
import pandas as pd
import multiprocessing as mp
from functools import partial

# local imports
from calculate_gsea_metrics import run_random_meta_gsea_all

def process_random_gsea_dataset(
    dataset_name,
    scgsea_gps_df_path,
    pseudobulk_base_dir,
    scgsea_base_dir,
    n_perm,
    sort_by,
    n_random_repeats
):
    dataset_dir_path = os.path.join(scgsea_gps_df_path, dataset_name)
    top_gene_programs_path = os.path.join(dataset_dir_path, "top_gene_programs.csv")
    if not os.path.exists(top_gene_programs_path):
        print(f"[SKIP] Missing top_gene_programs.csv for {dataset_name}\n")
        return

    try:
        # recall: this is our scGSEA outputs (top programs / cell grouping)
        scgsea_gps_df_dataset = pd.read_csv(top_gene_programs_path)
    except Exception as e:
        print(f"[SKIP] Failed to load top_gene_programs.csv for {dataset_name}: {e}\n")
        return

    required_cols = {"Cell Type", "Disease", "Gene Program", "Rank Type"}
    if not required_cols.issubset(set(scgsea_gps_df_dataset.columns)):
        print(f"[SKIP] Missing required columns in {top_gene_programs_path}\n")
        return

    output_path = os.path.join(dataset_dir_path, f"{dataset_name}_random_comparison.csv")
    
    try:
        print(f"[START] Running random meta-GSEA for {dataset_name}\n")
        run_random_meta_gsea_all(
            dataset_name=dataset_name,
            scgsea_gps_df_dataset=scgsea_gps_df_dataset,
            pseudobulk_base_dir=os.path.join(pseudobulk_base_dir, dataset_name),
            scgsea_base_dir=os.path.join(scgsea_base_dir, dataset_name),
            output_path=output_path,
            sort_by=sort_by,
            n_perm=n_perm,
            n_random_repeats=n_random_repeats
        )
        print(f"[DONE] Saved random GSEA summary to {output_path}\n")
    except Exception as e:
        print(f"[FAIL] Random GSEA for {dataset_name}: {e}\n")


# ---------------- CONFIG ----------------
# PATHS
SCGSEA_BASE_DIR = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/finished_datasets"
PSEUDOBULK_DIR = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"
GPS_PATH = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"
# OUTPUT_DIR = "/home/ddz5/Desktop/c2s-RL/random_control_outputs"

# Hyperparams
NUM_CORES = 12
N_PERM = 1000
SORT_BY = "NES"
N_RANDOM_REPEATS = 10

# os.makedirs(OUTPUT_DIR, exist_ok=True)

if __name__ == "__main__":
    dataset_names = [
        d for d in os.listdir(SCGSEA_BASE_DIR)
        if re.match(r"local\(\d+\)", d)
    ]

    print(f"Found {len(dataset_names)} datasets. Using {NUM_CORES} cores.\n")

    with mp.Pool(NUM_CORES) as pool:
        pool.map(partial(
            process_random_gsea_dataset,
            scgsea_gps_df_path=GPS_PATH,
            pseudobulk_base_dir=PSEUDOBULK_DIR,
            scgsea_base_dir=SCGSEA_BASE_DIR,
            n_perm=N_PERM,
            sort_by=SORT_BY,
            n_random_repeats=N_RANDOM_REPEATS
        ), dataset_names)

    print("\nAll datasets processed.")
