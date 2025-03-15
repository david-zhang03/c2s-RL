import os
import shutil
import multiprocessing as mp
from functools import partial
import math
import uuid

# parser.add_argument('--h5ad_path', type=str,
#                     help='Path to h5ad file')

# parser.add_argument('--dataset_name', type=str,
#                     help='Name of .h5ad dataset')

# parser.add_argument('--output_prefix', type=str, default=None,
#                     help='Path to the output file')
    
# parser.add_argument('--cell_type_column', type=str, required=False, default='cell_type',
#                     help='column name in adata.obs that contains cell type information')

# parser.add_argument('--disease_column', type=str, required=False, default='disease',
#                     help='column name in adata.obs that contains disease information')

# parser.add_argument('--norm_method', type=str, required=False, default='auto',
#                     help='normalization method to use')

# parser.add_argument('--target_sum', type=float, required=False, default=1e4,
#                     help='target sum for library size normalization before log1p (default: 10000)')

def process_datasets():
    pass





if __name__ == "__main__":
    DATA_PREFIX = "/SAY/standard/HCA-CC1022-InternalMedicine/datasets/HCA_CxG_Processing/hca_cellxgene_step1_colunified_h5ad"
    OUTPUT_PREFIX = "/home/ddz5/scratch/QA_dataset_DEGs/"

    # Compare datasets between output and data prefix to obtain datasets to process

