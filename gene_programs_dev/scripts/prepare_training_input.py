import pickle
import os
from datetime import datetime
import argparse
import sys

import scanpy as sc
import pandas as pd
import numpy as np

from scipy.sparse import csr_matrix
from sklearn.metrics import pairwise_distances
from sklearn.cluster import AgglomerativeClustering

sys.path.append('..')
from cell2gsea.utils import *

## this script will take 
# 1. an h5ad file : in the h5ad file, the logp10k normalized expressions should be in the adata.X 
# 2. a gene set in tsv file format 
# 3. additional parameters
#
## and produces as output:
# a dictionary which provide consistent inputs for a training run (stored as pickle file) ## we may change this to other format later in case there are better ways


parser = argparse.ArgumentParser()

parser.add_argument('--h5ad_path', type=str,
                    help='Path to h5ad file')

parser.add_argument('--genesets_path', type=str,
                    help='Path to input geneset library in tab-separated (tsv) format ')

parser.add_argument('--dataset_name', type=str,
                    help='Name of .h5ad dataset')

parser.add_argument('--output_prefix', type=str, default=None,
                    help='Path to the output file')

parser.add_argument('--max_missing', type=float, required=False, default=0.2,
                    help='maximum fraction of genes in a gene-set allowed to be missing from single-cell dataset, otherwise the gene-set will not be included')

parser.add_argument('--clust_sim_threshold', type=float, required=False, default=0.99,
                    help='maximum distance between gene sets to be in the same cluster')

parser.add_argument('--knn_k', type=int, required=False, default=5,
                    help='number of neighbors in the knn graph')

parser.add_argument('--knn_n_pca', type=int, required=False, default=50,
                    help='number of neighbors in the knn graph')
    
parser.add_argument('--cell_type_column', type=str, required=False, default='cell_type',
                    help='column name in adata.obs that contains cell type information')

parser.add_argument('--disease_column', type=str, required=False, default='disease',
                    help='column name in adata.obs that contains disease information')

args = parser.parse_args()

output_prefix = os.path.abspath(args.output_prefix)

output_path = os.path.join(output_prefix, args.dataset_name)

if not os.path.exists(output_path):
    os.makedirs(output_path, exist_ok=True)

print ("__________________________________",flush=True)
script_path = os.path.abspath(__file__)
log_str(f"Started running {script_path}")

# geneset_lib_df  = pd.read_csv(args.genesets_csv_path,sep='\t')
geneset_lib_df = pd.read_csv(args.genesets_path,sep='\t')
num_genesets_in_lib = len(geneset_lib_df)
log_str(f"Loaded gene-set library file: {num_genesets_in_lib:,} gene-sets in the library")

## read the h5ad file
log_str(f"Loading h5ad file  {os.path.abspath((args.h5ad_path))}")
adata = sc.read_h5ad(args.h5ad_path)
log_str(f"Loaded h5ad file: n_cells = {adata.shape[0]:,} , n_genes = {adata.shape[1]:,}")

# Check for cell type information
cell_type_column = args.cell_type_column
if cell_type_column in adata.obs.columns:
    log_str(f"Found cell type information in column '{cell_type_column}'")
    cell_types = adata.obs[cell_type_column].values
    unique_cell_types = set(cell_types)
    log_str(f"Dataset has {len(unique_cell_types)} unique cell types")
    if len(unique_cell_types) <= 1:
        log_str("Dataset has one or less cell types. Exiting...")
        sys.exit(1)
else:
    log_str(f"Cell type column '{cell_type_column}' not found in dataset. Exiting...")
    sys.exit(1)

# Check for disease information
disease_column = args.disease_column
if disease_column in adata.obs.columns:
    log_str(f"Found disease information in column '{disease_column}'")
    diseases = adata.obs[disease_column].values
    unique_diseases = set(diseases)
    log_str(f"Dataset has {len(unique_diseases)} unique disease states")
    diseases = diseases.tolist()  # array of shape (n_cells) to store the disease for each cell
else:
    log_str(f"Disease column '{disease_column}' not found in dataset. Using 'Unknown' as default")
    diseases = ['Unknown'] * adata.shape[0]

## build knn graph 
log_str(f"Building KNN graph with K={args.knn_k}")
sc.pp.pca(adata,n_comps =args.knn_n_pca)
sc.pp.neighbors(adata,n_neighbors=5)
edge_list = adata.obsp['connectivities'].nonzero()
log_str(f"Built KNN graph: n_edges = {len(edge_list[0]):,}")


log_str(f"Intersecting genes in library and genes in single-cell dataset")
terms = geneset_lib_df['gene_set'].values
member_genes = geneset_lib_df['genes'].values
# all_geneset_name_list=terms.tolist()

# dict of gene set to genes
gene_set_gene_map = {terms[i]: member_genes[i].split(',') for i in range(len(terms))}
# size_of_gene_set = {terms[i]: len(gene_set_gene_map[terms[i]]) for i in range(len(terms))}
genes_in_lib = set()
for term in gene_set_gene_map:
    genes_in_lib |= set(gene_set_gene_map[term])

## filter the gene sets

column_name = "index"
if 'gene_name' in adata.var.columns:
    column_name = "gene_name"
elif 'gene_symbols' in adata.var.columns:
    column_name = "gene_symbols"

# Filter out versioning and duplicates
sc_dataset_genes = set()
column_map = {}

if column_name == "index":
    unfiltered_genes = list(adata.var.index)
else:
    unfiltered_genes = list(adata.var[column_name].values)

for column, gene in enumerate(unfiltered_genes):
    if '-' in gene:
        gene = gene.split('-')[0]
    if gene not in sc_dataset_genes:
        sc_dataset_genes.add(gene)
        column_map[gene] = column

filtered_gene_set_names = []
filtered_gene_set_gene_map = {}
filtered_gene_set_size = []

# filter out gene sets with > args.max_missing proportion of genes
for term in gene_set_gene_map:
    intersection = set(gene_set_gene_map[term]) & sc_dataset_genes
    if (len(intersection)/len(gene_set_gene_map[term])) < (1-args.max_missing):
        pass
    else:
        filtered_gene_set_gene_map[term] = list(intersection)
        filtered_gene_set_names.append(term)
        filtered_gene_set_size.append(len(intersection))

# gene_set_size = [size_of_gene_set[term] for term in filtered_gene_set_names]
n_progs = len(filtered_gene_set_names)

# new_gene_set_index = {filtered_gene_set_names[i]: i for i in range(len(filtered_gene_set_names))}

common_genes = sc_dataset_genes & genes_in_lib
# gene_name_list = list(common_genes)
n_genes = len(common_genes)
index_of_gene = {gene: i for i, gene in enumerate(common_genes)}

# Filter adata
if column_name != "index":
    adata.var.index = adata.var[column_name]
# column index --> make sure this is filtering out non intersected genes
columns = []
for gene in common_genes:
    columns.append(column_map[gene])

adata_relevant= adata[:, columns]
log_str(f"Filtered genes and gene-sets: {n_genes:,} genes were in the intersection. {n_progs:,} gene-sets passed the (maximum missing ratio) threshold of {args.max_missing}")


## add the program clusters
progs = csr_matrix((n_progs, n_genes), dtype=int)
# Form the filtered gene set by gene binary matrix
for i in range(n_progs):
    gene_indices = [index_of_gene[gene] for gene in filtered_gene_set_gene_map[filtered_gene_set_names[i]]]
    progs[i,gene_indices] = 1


# append the cluster labels for each gene set at varying resolution
cluster_cols = [c for c in geneset_lib_df.columns if str(c).startswith("cluster_")]
cluster_dict = dict()
terms = geneset_lib_df['gene_set'].values
for cl in cluster_cols:    
    clusters = geneset_lib_df[cl].astype(int).values
    cluster_of  = {terms[i]:clusters[i] for i in range(len(terms))}
    cluster_dict[cl] = pd.Series(filtered_gene_set_names).map(cluster_of).values

## store the prepared inputs in a dictionary 
log_str(f"Saving the prepared training inputs...")
prepared_input = dict()
prepared_input['input_gene_set_csv_path'] = args.genesets_path  # the path to csv file that contains the gene sets using which this training input was built
prepared_input['input_h5ad_file_path'] = args.h5ad_path # input path to h5ad file from which this training input was prepared
prepared_input['gene_set_names'] = filtered_gene_set_names # list of length (n_programs) to store the name of gene programs
prepared_input['gene_set_sizes'] = filtered_gene_set_size # list of length (n_programs) which stores the number of genes in the gene set before intersecting with dataset's genes
prepared_input['gene_names'] = list(common_genes) # array of shape (n_genes) to store the name of genes that correspond to columns of the matrix X and columns of the matrix progs. This will be the intersection of genes in the dataset and genes in all gene sets combined

# adata_relevant is the gene filtered .X matrix
prepared_input['cell_ids'] = adata_relevant.obs.index.tolist() # array of shape (n_cells) to store the cell ids that correpond rows in matrix x when it would be inputed to the training script
prepared_input['extra'] = None # other string-formated notes that will be attached to this input

if cell_types is not None:
    prepared_input['cell_types'] = cell_types.tolist() # array of shape (n_cells) to store the cell type for each cell

if diseases is not None:
    prepared_input['disease'] = diseases # should already be in a list

## data: for input to the train_gnn function
prepared_input['train_x_sparse'] = csr_matrix(adata_relevant.X) # sparse matrix of shape (n_cells, n_genes) to store the gene expression data
prepared_input['train_edges'] = edge_list # list of length 2, each of shape (n_edges) to store the edges of the knn graph
prepared_input['progs_sparse'] = progs # sparse matrix of shape (n_programs, n_genes) to store the binary gene programs
prepared_input['prog_clusters'] = cluster_dict # array of shape (n_programs) to store the cluster number for each of the programs

## save the prepared input as a pickle file
pickle_output_path = os.path.join(output_path,"new_training_inputs.pickle")
with open(pickle_output_path, 'wb') as f:
    pickle.dump(prepared_input, f)

log_str(f"Saved the prepared training inputs at {os.path.abspath(output_path)}")

metadata_output_path = os.path.join(output_path, "metadata.txt")
log_str(f"Writing metadata to {os.path.abspath(metadata_output_path)}")

with open(metadata_output_path, 'w') as f:
    f.write(f"dataset_name: {args.dataset_name}\n")
    f.write(f"num_cells: {adata.shape[0]}\n")
    f.write(f"num_genes: {adata.shape[1]}\n")
    f.write(f"num_genes_after_filtering: {n_genes}\n")
    f.write(f"num_gene_sets: {n_progs}\n")
    f.write(f"num_cell_types: {len(unique_cell_types)}\n")
    f.write(f"cell_types: {', '.join(sorted(map(str, unique_cell_types)))}\n")
    f.write(f"num_edges: {len(edge_list[0])}\n")
    f.write(f"num_diseases: {len(unique_diseases)}\n")
    f.write(f"diseases: {', '.join(sorted(map(str, unique_diseases)))}\n")
    # reference timestamp
    f.write(f"processed_time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

log_str(f"Metadata file created successfully")

print ("__________________________________",flush=True)
