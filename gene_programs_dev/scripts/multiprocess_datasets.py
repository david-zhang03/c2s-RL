import os
import multiprocessing as mp
from functools import partial

def process_dataset(dataset_name, data_prefix, geneset_path, output_prefix):
    # Function to process a single dataset
    print(f"Processing {dataset_name}...\n")
    try:
        os.system(f"""
        python /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/prepare_training_input.py \
            --h5ad_path "{data_prefix}/{dataset_name}_cleaned.h5ad" \
            --genesets_path "{geneset_path}" \
            --dataset_name "{dataset_name}" \
            --output_prefix "{output_prefix}"
        """)
        print(f"Completed processing {dataset_name}\n")
    except Exception as e:
        print(f"Failed to process {dataset_name}: {e}\n")

if __name__ == "__main__":
    # Dataset and paths configuration
    DATASET_NAMES = [
        "local(474)",
        "local(468)",
        "local(486)",
        "local(198)",
        "local(611)",
        "local(698)",
        "local(465)",
        "local(637)",
        "local(769)",
        "local(608)",
        "local(304)",
        "local(285)",
        "local(605)",
        "local(728)",
        "local(280)",
        "local(282)",
        "local(701)",
        "local(706)",
        "local(303)",
        "local(699)",
        "local(487)",
        "local(277)",
        "local(50)",
        "local(609)",
        "local(607)",
        "local(606)",
        "local(191)",
        "local(777)",
        "local(268)",
        "local(273)",
        "local(473)",
        "local(638)",
        "local(768)",
        "local(703)",
        "local(513)",
        "local(477)",
        "local(289)",
        "local(771)",
        "local(274)",
        "local(700)",
        "local(610)",
        "local(762)",
        "local(467)",
        "local(478)",
        "local(51)",
        "local(462)",
        "local(613)",
        "local(635)",
        "local(482)",
        "local(772)",
        "local(464)",
        "local(763)",
        "local(290)",
        "local(726)",
        "local(286)",
        "local(603)",
        "local(192)",
        "local(269)",
        "local(199)",
        "local(284)",
        "local(770)",
        "local(475)",
        "local(481)",
        "local(275)",
        "local(305)",
        "local(604)",
        "local(612)",
        "local(707)",
        "local(480)",
        "local(702)",
        "local(739)",
        "local(281)",
        "local(636)",
        "local(738)",
        "local(287)",
        "local(633)",
        "local(471)",
        "local(466)",
        "local(272)",
        "local(614)",
        "local(778)",
        "local(632)",
        "local(546)",
        "local(276)",
        "local(52)",
        "local(463)",
        "local(476)",
        "local(634)",
        "local(479)"
    ]

    # DATASET_NAMES = [
    #     "local(23)",
    #     "local(267)"
    # ]

    DATA_PREFIX = "/home/sz568/scratch/C2S_RL/all_datasets/hca_cellxgene_cleaned_h5ad/"
    GENESET_PATH = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/data/msigdb.tsv"
    OUTPUT_PREFIX = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/"

    # Number of CPU cores to use
    NUM_CORES = mp.cpu_count()

    print(f"Number of cores: {NUM_CORES}")

    # Multiprocessing pool
    with mp.Pool(NUM_CORES) as pool:
        pool.map(partial(process_dataset, data_prefix=DATA_PREFIX, geneset_path=GENESET_PATH, output_prefix=OUTPUT_PREFIX), DATASET_NAMES)
