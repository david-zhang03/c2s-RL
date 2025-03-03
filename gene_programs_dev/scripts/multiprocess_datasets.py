import os
import shutil
import multiprocessing as mp
from functools import partial
import math
import uuid

def process_dataset_group(group_index, dataset_group, data_prefix, geneset_path, output_prefix, datasets_per_group=3):
    """Process a group of datasets, storing them in a common subdirectory"""
    # Generate a subdirectory name based on the group index
    subdir_name = f"set_{group_index + 1}"
    subdir_path = os.path.join(output_prefix, subdir_name)
    
    # Create the subdirectory if it doesn't exist
    os.makedirs(subdir_path, exist_ok=True)
    
    print(f"Processing group {group_index + 1} with datasets: {dataset_group} in directory {subdir_path}")
    
    processed_count = 0
    skipped_count = 0

    # Process each dataset in the group
    for dataset_name in dataset_group:
        # Check if this dataset has already been processed
        dataset_output_path = os.path.join(subdir_path, dataset_name)
        pickle_file = os.path.join(dataset_output_path, "training_inputs.pickle")
        metadata_file = os.path.join(dataset_output_path, "metadata.txt")
        
        if os.path.exists(pickle_file) and os.path.exists(metadata_file):
            print(f"Skipping {dataset_name} - already processed (found both training_inputs.pickle and metadata.txt)")
            skipped_count += 1
            continue

        process_single_dataset(dataset_name, data_prefix, geneset_path, subdir_path)
        processed_count += 1
    
    print(f"Group {group_index + 1}: Processed {processed_count} datasets, skipped {skipped_count} datasets")
    return subdir_path, processed_count, skipped_count


def process_single_dataset(dataset_name, data_prefix, geneset_path, output_path):
    """Process a single dataset within a group"""
    print(f"Processing {dataset_name}...")
    
    # Note that os.system creates a subshell which does not inherit conda env's python interpreter
    python_env = "/gpfs/radev/home/sr2464/.conda/envs/llamp/bin/python"
    full_path = f"{data_prefix}/{dataset_name}_colunified.h5ad"
    
    exit_code = os.system(f"""
    {python_env} /home/ddz5/Desktop/c2s-RL/gene_programs_dev/scripts/prepare_training_input.py \
        --h5ad_path "{full_path}" \
        --genesets_path "{geneset_path}" \
        --dataset_name "{dataset_name}" \
        --output_prefix "{output_path}"
    """)
    
    if exit_code == 0:
        print(f"Completed processing {dataset_name}")
    else:
        print(f"Failed to process {dataset_name}: Exit code: {exit_code}")


if __name__ == "__main__":
    # Dataset and paths configuration --> paths refer to TRANSFER PARTITION
    DATA_PREFIX = "/SAY/standard/HCA-CC1022-InternalMedicine/datasets/HCA_CxG_Processing/hca_cellxgene_step1_colunified_h5ad"
    GENESET_PATH = "/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data/msigdb.tsv"
    OUTPUT_PREFIX = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models_bk/Cell2GSEA_QA_dataset_models/"
    
    # Number of datasets to process in each group
    DATASETS_PER_GROUP = 3
    
    DATASET_NAMES = []
    
    DATASET_PATHS = "/home/ddz5/scratch/Cell2GSEA_QA_dataset_models_bk/Cell2GSEA_QA_dataset_models/to_train_filepaths.txt"
    
    with open(DATASET_PATHS, 'r') as f:
        for line in f:
            if line.strip():
                parts = line.strip().split(',')
                if len(parts) > 0:
                    filepath = parts[0].strip()
                    filename = os.path.basename(filepath)
                    if filename.endswith('_colunified.h5ad'):
                        dataset_name = filename.split('_colunified.h5ad')[0]
                    else:
                        dataset_name = os.path.splitext(filename)[0]
                    DATASET_NAMES.append(dataset_name)
    
    print(f"Datasets to process: {DATASET_NAMES}")
    print(f"Total number of datasets: {len(DATASET_NAMES)}")
    
    # Group datasets into chunks of DATASETS_PER_GROUP
    dataset_groups = []
    n_groups = math.ceil(len(DATASET_NAMES) / DATASETS_PER_GROUP)
    
    # this should be deterministic, so running again on unprocessed datasets should
    # not overwrite processed datasets
    for i in range(n_groups):
        start_idx = i * DATASETS_PER_GROUP
        end_idx = min((i + 1) * DATASETS_PER_GROUP, len(DATASET_NAMES))
        dataset_groups.append(DATASET_NAMES[start_idx:end_idx])
    
    print(f"Number of dataset groups: {len(dataset_groups)}")
    
    # Create a list of group indices to pass to the multiprocessing function
    group_indices = list(range(len(dataset_groups)))
    
    # Number of CPU cores to use
    NUM_CORES = mp.cpu_count()
    print(f"Number of cores available: {NUM_CORES}")
    
    # Set up a record of which datasets are in which group
    group_record_path = os.path.join(OUTPUT_PREFIX, "dataset_groups.txt")
    with open(group_record_path, 'w') as f:
        for i, group in enumerate(dataset_groups):
            f.write(f"set_{i+1}: {', '.join(group)}\n")
    
    print(f"Dataset group assignments saved to {group_record_path}")
    
    # Multiprocessing pool to process groups in parallel
    with mp.Pool(NUM_CORES) as pool:
        results = pool.starmap(
            process_dataset_group,
            [(i, dataset_groups[i], DATA_PREFIX, GENESET_PATH, OUTPUT_PREFIX, DATASETS_PER_GROUP) for i in group_indices]
        )
    
    # Print summary
    total_processed = 0
    total_skipped = 0
    for subdir_path, processed, skipped in results:
        total_processed += processed
        total_skipped += skipped
        if processed > 0:  # Only print groups that processed something
            print(f"Processed {processed} datasets in directory {subdir_path}")
    
    print(f"Summary: Processed {total_processed} datasets, skipped {total_skipped} datasets")
    print("All processing complete!")
    
    # DATASET_NAMES = [
    #     "local(474)",
    #     "local(468)",
    #     "local(486)",
    #     "local(198)",
    #     "local(611)",
    #     "local(698)",
    #     "local(465)",
    #     "local(637)",
    #     "local(769)",
    #     "local(608)",
    #     "local(304)",
    #     "local(285)",
    #     "local(605)",
    #     "local(728)",
    #     "local(280)",
    #     "local(282)",
    #     "local(701)",
    #     "local(706)",
    #     "local(303)",
    #     "local(699)",
    #     "local(487)",
    #     "local(277)",
    #     "local(50)",
    #     "local(609)",
    #     "local(607)",
    #     "local(606)",
    #     "local(191)",
    #     "local(777)",
    #     "local(268)",
    #     "local(273)",
    #     "local(473)",
    #     "local(638)",
    #     "local(768)",
    #     "local(703)",
    #     "local(513)",
    #     "local(477)",
    #     "local(289)",
    #     "local(771)",
    #     "local(274)",
    #     "local(700)",
    #     "local(610)",
    #     "local(762)",
    #     "local(467)",
    #     "local(478)",
    #     "local(51)",
    #     "local(462)",
    #     "local(613)",
    #     "local(635)",
    #     "local(482)",
    #     "local(772)",
    #     "local(464)",
    #     "local(763)",
    #     "local(290)",
    #     "local(726)",
    #     "local(286)",
    #     "local(603)",
    #     "local(192)",
    #     "local(269)",
    #     "local(199)",
    #     "local(284)",
    #     "local(770)",
    #     "local(475)",
    #     "local(481)",
    #     "local(275)",
    #     "local(305)",
    #     "local(604)",
    #     "local(612)",
    #     "local(707)",
    #     "local(480)",
    #     "local(702)",
    #     "local(739)",
    #     "local(281)",
    #     "local(636)",
    #     "local(738)",
    #     "local(287)",
    #     "local(633)",
    #     "local(471)",
    #     "local(466)",
    #     "local(272)",
    #     "local(614)",
    #     "local(778)",
    #     "local(632)",
    #     "local(546)",
    #     "local(276)",
    #     "local(52)",
    #     "local(463)",
    #     "local(476)",
    #     "local(634)",
    #     "local(479)"
    # ]

    # DATASET_NAMES = [
    #     "local(23)",
    #     "local(267)"
    # ]