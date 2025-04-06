import os
import pandas as pd
import multiprocessing as mp
import re
import sys

from calculate_gsea_metrics import run_meta_gsea_all

# ---------------- CONFIG ----------------
BASE_DIR = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data"
# OUTPUT_DIR = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/"
NUM_CORES = 12
N_PERM = 1000
SORT_BY = "NES"

# os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------- PROCESS FUNCTION ----------------
def process_dataset(dataset_name):
    dataset_path = os.path.join(BASE_DIR, dataset_name)
    top_gene_programs_path = os.path.join(dataset_path, "top_gene_programs.csv")
    # output_dir = os.path.join(OUTPUT_DIR, dataset_name)
    output_path = os.path.join(dataset_path, f"{dataset_name}_comparison_results.csv")

    if not os.path.exists(top_gene_programs_path):
        print(f"[SKIP] No top_gene_programs.csv found in {dataset_name}")
        return

    try:
        gps_df_dataset = pd.read_csv(top_gene_programs_path)
    except Exception as e:
        print(f"[SKIP] Failed to load top_gene_programs.csv for {dataset_name}: {e}")
        return

    # Check required columns exist
    required_cols = {"Cell Type", "Disease", "Gene Program", "Rank Type"}
    if not required_cols.issubset(set(gps_df_dataset.columns)):
        print(f"[SKIP] Missing required columns in {top_gene_programs_path}")
        return

    print(f"[START] Meta-GSEA on {dataset_name} ...")
    try:
        _ = run_meta_gsea_all(
            gps_df_dataset=gps_df_dataset,
            base_dir=BASE_DIR,
            datasets=[dataset_name],
            output_path=output_path,
            sort_by=SORT_BY,
            n_perm=N_PERM
        )
        print(f"[DONE] Saved summary to {output_path}")
    except Exception as e:
        print(f"[FAIL] {dataset_name}: {e}")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    dataset_names = [
        d for d in os.listdir(BASE_DIR)
        if re.match(r"local\(\d+\)", d)
    ]

    print(f"Found {len(dataset_names)} datasets. Using {NUM_CORES} cores.\n")

    with mp.Pool(NUM_CORES) as pool:
        pool.map(process_dataset, dataset_names)

    print("\nAll datasets processed.")
