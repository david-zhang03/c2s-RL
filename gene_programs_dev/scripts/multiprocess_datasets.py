import os
import multiprocessing as mp
from functools import partial

def process_dataset(dataset_name, data_prefix, geneset_path, output_prefix):
    # Function to process a single dataset
    print(f"Processing {dataset_name}...")
    try:
        os.system(f"""
        python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/prepare_training_input.py \
            --h5ad_path "{data_prefix}/{dataset_name}_cleaned.h5ad" \
            --genesets_path "{geneset_path}" \
            --dataset_name "{dataset_name}" \
            --output_prefix "{output_prefix}"
        """)
        print(f"Completed processing {dataset_name}")
    except Exception as e:
        print(f"Failed to process {dataset_name}: {e}")

if __name__ == "__main__":
    # Dataset and paths configuration
    DATASET_NAMES = [
        "local(706)",
        "local(474)",
        "local(462)",
        "local(486)",
        "local(738)",
        "local(603)",
        "local(698)",
        "local(513)",
        "local(52)",
        "local(471)",
        # "local(23)"
        "local(50)",
        "local(546)",
        "local(728)",
        "local(778)",
        # "local(267)",
        "local(635)",
        "local(632)",
        "local(777)",
        "local(633)",
        "local(473)",
        "local(636)",
        "local(739)"
    ]

    DATA_PREFIX = "/home/sz568/scratch/C2S_RL/all_datasets/hca_cellxgene_cleaned_h5ad/"
    GENESET_PATH = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/data/msigdb.tsv"
    OUTPUT_PREFIX = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/"

    # Number of CPU cores to use
    NUM_CORES = mp.cpu_count()

    # Multiprocessing pool
    with mp.Pool(NUM_CORES) as pool:
        pool.map(partial(process_dataset, data_prefix=DATA_PREFIX, geneset_path=GENESET_PATH, output_prefix=OUTPUT_PREFIX), DATASET_NAMES)
