import os
import argparse
from datetime import datetime

import torch
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def format_size(size_in_bytes):
    """Formats a bytes size into a more human-readable string."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} bytes"
    elif size_in_bytes < 1024**2:
        return f"{size_in_bytes / 1024:.2f} KB"
    elif size_in_bytes < 1024**3:
        return f"{size_in_bytes / 1024**2:.2f} MB"
    else:
        return f"{size_in_bytes / 1024**3:.2f} GB"



def get_knn_edges(X,knn_k,n_pca):
    # Get the KNN graph edges without self-loops
    # reduce X to n_pca dimensions
    pca = PCA(n_components=n_pca)
    X = pca.fit_transform(X)


    knn = NearestNeighbors(n_neighbors=knn_k+1).fit(X)
    _, indices = knn.kneighbors(X)
    edge_src = []
    edge_des = []
    for i in range(indices.shape[0]):
        for j in range(1, indices.shape[1]):  # Start from 1 to skip the point itself
            edge_src.append(i)
            edge_des.append(indices[i, j])
    return np.array([edge_src,edge_des])


def check_valid_path(args):
    for arg in args:
        if arg and not os.path.exists(arg):
            raise argparse.ArgumentTypeError(f"The file {arg} does not exist")
    return


def log_memory_usage(stage, device):
    allocated = torch.cuda.memory_allocated(device) / 1e6  # Convert to MB
    reserved = torch.cuda.memory_reserved(device) / 1e6    # Convert to MB
    print(f"[{stage}] Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB")


def log_string(log_fout, out_str):
    log_fout.write(out_str + '\n')
    log_fout.flush()
    print(out_str)



def get_timestamp():
    return datetime.now().strftime(r'%Y_%m_%d__%H_%M_%S')

def log_str(msg):
    print(f"[{get_timestamp()}] {msg}", flush=True)


def generate_gene_row(num_cols, col_indices, gene_list):
    # Obtain the indices 
    indices = []
    for gene in gene_list:
        try:
            indices.append(col_indices[gene])
        except:
            pass
    row = np.zeros(num_cols, dtype=int)
    row[indices] = 1
    return row

def generate_program_dict(filepath):
    filetype = filepath.split('.')[-1]
    if filetype == 'csv':
        programs = pd.read_csv(filepath)
    else:
        programs = pd.read_csv(filepath, sep='\t')
    # First column is gene set/program, second column is genes
    programs.columns = ['gene_set', 'genes']
    program_dict = {program: genes.split(',') for program, genes in zip(programs['gene_set'], programs['genes'])}
    return program_dict

# Generate program matrix from dict of programs and gene associations
def generate_program_matrix(program_dict, all_genes, normalize):
    col_indices = {gene: i for i, gene in enumerate(all_genes)}
    rows = []
    num_cols = len(all_genes)
    for program, genes in program_dict.items():
        if isinstance(genes, str):
            genes = genes.split(',')
        row = generate_gene_row(num_cols, col_indices, genes)
        rows.append(row)
    progs = np.array(rows)
    if normalize:
        for i, genes in enumerate(program_dict.values()):
            progs[i] = progs[i] / np.sqrt(len(genes))
    return progs