import os
import sys
import shutil
import multiprocessing as mp
from functools import partial
import math
import uuid
import subprocess
import glob
import argparse
from datetime import datetime
import logging

def setup_logging(output_dir):
    """Set up logging for the multiprocessing script"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    log_path = os.path.join(output_dir, f"multiprocess_de_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger()

def run_de_analysis(h5ad_path, dataset_name, output_prefix, cell_type_column="cell_type", 
                    disease_column="disease", reference_disease="normal", logger=None):
    """Run the differential expression analysis script on a single h5ad file"""
    if logger is None:
        logger = logging.getLogger()
    
    # Generate a unique ID for this run
    run_id = str(uuid.uuid4())[:8]
    logger.info(f"Starting DE analysis for {dataset_name} (ID: {run_id})")
    
    # Construct the command to run the DE script
    script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_diff_express.py")
    
    command = [
        "python", script_path,
        "--h5ad_path", h5ad_path,
        "--dataset_name", dataset_name,
        "--output_prefix", output_prefix,
        "--cell_type_column", cell_type_column,
        "--disease_column", disease_column,
        "--reference_disease", reference_disease
    ]
    
    try:
        # Run the command and capture output
        logger.info(f"Running command: {' '.join(command)}")
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate()
        
        # Log the output
        if stdout:
            logger.info(f"[{run_id}] stdout: {stdout}")
        if stderr:
            logger.error(f"[{run_id}] stderr: {stderr}")
        
        # Check return code
        if process.returncode != 0:
            logger.error(f"[{run_id}] Process failed with return code {process.returncode}")
            return {
                "dataset": dataset_name,
                "success": False,
                "error": stderr,
                "return_code": process.returncode
            }
        
        logger.info(f"[{run_id}] DE analysis completed successfully for {dataset_name}")
        return {
            "dataset": dataset_name,
            "success": True,
            "output_dir": os.path.join(output_prefix, dataset_name)
        }
    
    except Exception as e:
        logger.error(f"[{run_id}] Exception during processing {dataset_name}: {str(e)}")
        return {
            "dataset": dataset_name,
            "success": False,
            "error": str(e)
        }

def get_datasets_to_process(data_prefix, output_prefix):
    """
    Compare datasets between output and data prefix to obtain datasets to process.
    Returns datasets that exist in data_prefix but not in output_prefix.
    """
    # Get all h5ad files in the data directory
    h5ad_files = glob.glob(os.path.join(data_prefix, "**", "*.h5ad"), recursive=True)
    
    # Extract dataset names from h5ad files
    all_datasets = []
    for h5ad_path in h5ad_files:
        dataset_name = os.path.basename(h5ad_path).replace(".h5ad", "")
        all_datasets.append((dataset_name, h5ad_path))
    
    # Get all processed datasets (directories in output_prefix)
    processed_datasets = set()
    if os.path.exists(output_prefix):
        processed_datasets = set(os.listdir(output_prefix))
    
    # Filter datasets that haven't been processed yet
    datasets_to_process = []
    for dataset_name, h5ad_path in all_datasets:
        # Check if this dataset has been processed or is in progress
        if dataset_name not in processed_datasets:
            datasets_to_process.append((dataset_name, h5ad_path))
    
    return datasets_to_process

def process_datasets(data_prefix, output_prefix, max_processes=None, cell_type_column="cell_type", 
                     disease_column="disease", reference_disease="normal"):
    """
    Process multiple datasets in parallel using multiprocessing.
    
    Parameters:
    -----------
    data_prefix : str
        Path to directory containing h5ad files
    output_prefix : str
        Path to directory for output results
    max_processes : int, optional
        Maximum number of parallel processes. If None, uses CPU count.
    cell_type_column : str, optional
        Column name for cell type annotations
    disease_column : str, optional
        Column name for disease annotations
    reference_disease : str, optional
        Reference disease state for comparisons
    """
    # Set up logging
    logger = setup_logging(output_prefix)
    
    # Get datasets to process
    datasets_to_process = get_datasets_to_process(data_prefix, output_prefix)
    
    if not datasets_to_process:
        logger.info("No new datasets to process. All datasets have been processed.")
        return
    
    logger.info(f"Found {len(datasets_to_process)} datasets to process")
    
    # Determine number of processes
    if max_processes is None:
        max_processes = mp.cpu_count() - 1  # Leave one CPU free
    max_processes = max(1, min(max_processes, mp.cpu_count()))  # Ensure valid range
    
    logger.info(f"Using {max_processes} processes for parallel execution")
    
    # Create partial function with fixed arguments
    run_de_partial = partial(
        run_de_analysis,
        output_prefix=output_prefix,
        cell_type_column=cell_type_column,
        disease_column=disease_column,
        reference_disease=reference_disease,
        logger=logger
    )
    
    # Process datasets in parallel
    results = []
    with mp.Pool(processes=max_processes) as pool:
        # Start processing
        for dataset_name, h5ad_path in datasets_to_process:
            logger.info(f"Queueing dataset: {dataset_name}")
            result = pool.apply_async(run_de_partial, args=(h5ad_path, dataset_name))
            results.append((dataset_name, result))
        
        # Wait for all processes to complete and collect results
        for dataset_name, result in results:
            try:
                process_result = result.get()  # This will wait for the process to complete
                if process_result["success"]:
                    logger.info(f"Successfully processed {dataset_name}")
                else:
                    logger.error(f"Failed to process {dataset_name}: {process_result.get('error', 'Unknown error')}")
            except Exception as e:
                logger.error(f"Exception while getting result for {dataset_name}: {str(e)}")
    
    logger.info("All datasets have been processed")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run differential expression analysis on multiple datasets in parallel')
    parser.add_argument('--data_prefix', type=str, default="/gpfs/radev/home/sz568/scratch/C2S_RL/hca_cellxgene_cleaned_h5ad",
                        help='Path to directory containing h5ad files')
    parser.add_argument('--output_prefix', type=str, default="/home/ddz5/scratch/QA_dataset_DEGs/",
                        help='Path to directory for output results')
    parser.add_argument('--max_processes', type=int, default=None,
                        help='Maximum number of parallel processes. If not specified, uses CPU count - 1')
    parser.add_argument('--cell_type_column', type=str, default="cell_type",
                        help='Column name for cell type annotations')
    parser.add_argument('--disease_column', type=str, default="disease",
                        help='Column name for disease annotations')
    parser.add_argument('--reference_disease', type=str, default="normal",
                        help='Reference disease state for comparisons')
    
    args = parser.parse_args()
    
    # Make sure output directory exists
    os.makedirs(args.output_prefix, exist_ok=True)
    
    # Run the multiprocessing function
    process_datasets(
        data_prefix=args.data_prefix,
        output_prefix=args.output_prefix,
        max_processes=args.max_processes,
        cell_type_column=args.cell_type_column,
        disease_column=args.disease_column,
        reference_disease=args.reference_disease
    )