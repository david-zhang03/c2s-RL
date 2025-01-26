import random
import os
# import yaml
from tqdm import tqdm
import sys

import wandb
import argparse
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import anndata as ad

from sklearn.model_selection import train_test_split

# local code imports
sys.path.append('..')
from cell2gsea import train_gnn, gnn_config
from cell2gsea.utils import *


parser = argparse.ArgumentParser()

parser.add_argument('--input_path', type=str, 
                    help='Usage: /home/ddz5/scratch/Cell2GSEA_QA_dataset_models/local_23/training_inputs.pickle')

parser.add_argument('--output_prefix', type=str, 
                    default="/home/ddz5/scratch/Cell2GSEA_QA_dataset_models/")

parser.add_argument('--dataset_name', type=str,
                    help='Name of .h5ad dataset')

parser.add_argument('--seed', type=int,
                    default=42)

# .yml file
parser.add_argument('--config_file', type=str, 
                    default=None)

parser.add_argument('--run_name', type=str, default=None)


args = parser.parse_args()

# set all seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# default config
conf = gnn_config(SEED=args.seed)
if args.config_file is not None:
    conf.from_yaml(args.config_file)

if args.output_prefix is None:
    args.output_prefix = os.path.dirname(args.input_path) + '/'

output_dir = os.path.join(args.output_prefix, args.dataset_name)

if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)


# read prepared input from pickle file
print ("__________________________________",flush=True)
script_path = os.path.abspath(__file__)
log_str(f"Started running {script_path}")


log_str(f"Reading prepared input at {os.path.abspath(args.input_path)}")
with open(args.input_path, 'rb') as f:
    prepared_input = pickle.load(f)

n_cell, n_gene = prepared_input['train_x_sparse'].shape
input_train_x = prepared_input['train_x_sparse'].toarray() 

progs =  prepared_input['progs_sparse']

log_str("Preparing train and val set")

dataset = ad.read_h5ad(prepared_input['input_h5ad_file_path'])

train_indices, val_indices = train_test_split(
    np.arange(n_cell),
    test_size = conf.VAL_SPLIT,
    random_state = conf.SEED
)

train_x = input_train_x[train_indices]
val_x = input_train_x[val_indices]

train_edges = get_knn_edges(train_x, conf.KNN_GRAPH_K, conf.KNN_GRAPH_N_PCA)

val_edges = get_knn_edges(val_x, conf.KNN_GRAPH_K, conf.KNN_GRAPH_N_PCA)

# cell types
train_cell_types = None
val_cell_types = None
if 'cell_type' in dataset.obs.columns:
    train_cell_types = dataset.obs['cell_type'][train_indices].values
    val_cell_types = dataset.obs['cell_type'][val_indices].values
else:
    log_str("Dataset has no cell types, exiting...")
    sys.exit(1)

# only train datasets with more than one cell type
if len(set(val_cell_types)) <= 1:
    log_str("Require datasets with more than one cell type, exiting...")
    sys.exit(1)

log_str("Converting programs from sparse to dense")

if conf.PROGRAM_NORMALIZE:
    log_str("Normalizing programs first")

    prog_norm_factor = np.array(prepared_input['gene_set_sizes'])
    norm_factor_bcast = prog_norm_factor[:, np.newaxis]
    progs_norm = progs / norm_factor_bcast # numpy matrix

    input_progs = np.asarray(progs_norm)
else:
    input_progs = progs.toarray()


curr_run  = wandb.init(entity = "dl452")
cluster_labels = None
clusters = "cluster_" + str(conf.CLUSTER_RESOL)
if (clusters != ""):
    cluster_labels = prepared_input["prog_clusters"][clusters]
    n_clusters = len(set(cluster_labels))
    log_str(f"Using program clusters in resolution column {clusters} of gene-set libray (tsv) file. Number of clusters: {n_clusters}")
conf.PROG_CLUSTER = cluster_labels

# set wandb run name
if args.run_name is None:
    run_name = args.dataset_name
else:
    run_name = args.run_name

curr_run.name = run_name

training_output = train_gnn(
    train_x = train_x,
    train_edges = train_edges,
    validation_x = val_x,
    validation_edges = val_edges,    
    progs = input_progs,
    prog_cluster_labels = cluster_labels,
    train_cell_types = train_cell_types,
    val_cell_types = val_cell_types,
    training_config = conf,
    wandb_run = curr_run,
    OUTPUT_PREFIX = output_dir,
    RUN_NAME = run_name,
    ENABLE_OUTPUT_SCORES= True,
    REGRESSION=True,
)

curr_run.finish()

output_dict = dict()
output_dict['input_path'] = args.input_path
output_dict['original_h5ad_path'] = prepared_input['input_h5ad_file_path']
output_dict['gene_set_names'] = prepared_input['gene_set_names'] 
output_dict['gene_set_sizes'] = prepared_input['gene_set_sizes'] 
output_dict['gene_names'] = prepared_input['gene_names']
output_dict['cell_ids'] = prepared_input['cell_ids']
output_dict['config'] = str(conf.__dict__)
output_dict['output'] = training_output


## save the ouptut as a pickle file
log_str("Training done. Saving outputs")
output_path = os.path.join(output_dir,'training_output.pickle')

with open(output_path, 'wb') as f:
    pickle.dump(output_dict, f)

log_str(f"Saved outputs at {os.path.abspath(output_path)}")
print ("__________________________________",flush=True)

# generate training done flag (for interacting with the slurm training)
training_flag_path = os.path.join(output_dir, 'training_done.flag')
with open(training_flag_path, 'w') as f:
    pass