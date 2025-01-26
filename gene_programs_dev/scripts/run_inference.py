import os
from tqdm import tqdm
import argparse
import yaml
import random
import pickle
import sys

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import wandb

import torch
from torch_geometric.data import Data
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

from cell2gsea import models, gnn_config
from cell2gsea.utils import *
import temp_import


def infer_gnn(
        input_x,
        input_edges,  
        trained_model,
        train_config,
    ):
    
    inference_X = torch.tensor(input_x, dtype=torch.float32)
    inference_edge_list = torch.tensor(input_edges, dtype=torch.long)
    inference_data = Data(x=inference_X , edge_index=inference_edge_list)

    inference_neighbor_sampler = NeighborSampler(
        inference_data,
        num_neighbors=[train_config.GSAMP_NUM_NBR, train_config.GSAMP_NUM_NBR]
    )

    inference_loader = NodeLoader(
        inference_data,
        node_sampler = inference_neighbor_sampler,
        batch_size = train_config.GSAMP_BATCH_SIZE,  # number of seed nodes
        num_workers = train_config.GSAMP_NUM_WORKER,
    )
    
    n_cell, n_genes = input_x.shape
    

    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    log_str(f"Using device: {device}")

    model = trained_model.to(device)
    model.eval()

    dummy_x = torch.zeros(1,n_genes,device=device)
    dummy_edges = torch.tensor([range(1),range(1)], dtype=torch.long,device=device)
    dummy_out = model(dummy_x,dummy_edges)
    _, n_progs = dummy_out.shape
    output_scores = np.zeros((n_cell,n_progs))

    log_str(f"Running inference: {n_cell} cells - {n_genes} genes - {n_progs} programs")
    with torch.no_grad():
        for idx, subgraph in tqdm(enumerate(inference_loader)):
            n_nodes, _ = subgraph.x.shape
            original_indices = subgraph.input_id
            subgraph = subgraph.to(device)
            batch_identity_adj = torch.tensor([range(n_nodes),range(n_nodes)], dtype = torch.long, device=device)
            edges = torch.cat([batch_identity_adj,subgraph.edge_index],dim = 1)
            batch_program_scores = model(subgraph.x,edges)
            seed_nodes_prog_scores = batch_program_scores[:len(original_indices),:].detach().cpu().numpy()
            output_scores[original_indices,:] = seed_nodes_prog_scores
            
    return output_scores



parser = argparse.ArgumentParser()

parser.add_argument('--model_path', type=str)
parser.add_argument('--prepared_train_inputs', type=str)
parser.add_argument('--output_save_path', type=str)

args = parser.parse_args()


conf = gnn_config()


log_str(f"Reading prepared input at {os.path.abspath(args.prepared_train_inputs)}")


with open(args.prepared_train_inputs, 'rb') as f:
    inputs = pickle.load(f)


progs = inputs['progs_sparse']
progs = progs.toarray()
num_prog, num_genes = progs.shape


n_cell, _ = inputs['train_x_sparse'].shape
input_train_x = inputs['train_x_sparse'].toarray() 


edge_list = inputs['train_edges']
edge_list = np.array(edge_list)


device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)


trained_model = models.get_gnn_model(num_genes, num_prog, conf).to(device)

try:
    checkpoint = torch.load(args.model_path)
    trained_model.load_state_dict(checkpoint['model_state_dict'])
except Exception as e:
    log_str(f"Error loading model: {e}")
    sys.exit(1)


output_scores = infer_gnn(input_train_x, edge_list, trained_model, conf)

# Save outputs

os.makedirs(args.output_save_path, exist_ok=True)

output_scores_path = os.path.join(args.output_save_path, "output_scores.pickle")
with open(output_scores_path, 'wb') as f:
    pickle.dump(output_scores, f)

log_str(f"Inference scores saved to {output_scores_path}")

# After loading inputs
adata = ad.read_h5ad(inputs['input_h5ad_file_path'])

if 'cell_type' not in adata.obs.columns:
    log_str("Error: Required columns 'cell_type' or 'disease' not found in data")
    sys.exit(1)

if 'disease' not in adata.obs.columns:
    log_str("Error: Required columns 'disease' not found in data")
    sys.exit(1)

# Assuming output_scores is a numpy array with shape (n_cells, n_gene_programs)
# inputs['gene_set_names'] is a list of gene program names matching the columns of output_scores
# adata.obs['cell_type'] and adata.obs['disease'] contain cell types and disease states, respectively

# Convert output_scores to a DataFrame for easier manipulation
gene_program_names = inputs['gene_set_names']
cell_types = adata.obs['cell_type']
diseases = adata.obs['disease']

# Analysis of output scores
try:
    # Create DataFrame with scores and metadata
    output_scores_df = pd.DataFrame(output_scores, columns=inputs['gene_set_names'])
    output_scores_df['cell_type'] = adata.obs['cell_type'].values
    output_scores_df['disease'] = adata.obs['disease'].values

    # Create a list to store results
    results_list = []
    
    # Open a text file for writing the analysis
    txt_path = os.path.join(args.output_save_path, "analysis_results.txt")
    with open(txt_path, 'w') as txt_file:
        # Write header
        txt_file.write("Analysis of Gene Program Activity by Disease and Cell Type\n")
        txt_file.write("="*50 + "\n\n")
        
        # Analyze each disease-cell type group
        for (disease, cell_type), group in output_scores_df.groupby(['disease', 'cell_type']):
            # Get gene program columns only
            gene_program_cols = inputs['gene_set_names']
            mean_scores = group[gene_program_cols].mean()
            
            # Get top 5 programs
            top_programs = mean_scores.nlargest(5)
            
            # Write to both console and file
            header = f"\nDisease: {disease}, Cell type: {cell_type}"
            log_str(header)
            txt_file.write(header + "\n")
            
            subheader = "Top 5 gene programs:"
            log_str(subheader)
            txt_file.write(subheader + "\n")
            
            for program, score in top_programs.items():
                result_line = f"  {program}: {score:.3f}"
                log_str(result_line)
                txt_file.write(result_line + "\n")
                
                results_list.append({
                    'Disease': disease,
                    'Cell Type': cell_type,
                    'Gene Program': program,
                    'Mean Activity Score': score
                })

    # Save results to CSV first
    results_df = pd.DataFrame(results_list)
    csv_path = os.path.join(args.output_save_path, "top_gene_programs.csv")
    results_df.to_csv(csv_path, index=False)
    
    # Append the summary to the text file
    with open(txt_path, 'a') as txt_file:
        txt_file.write("\n" + "="*50 + "\n")
        summary = f"\nAnalysis completed. Results saved to {csv_path}"
        txt_file.write(summary)
    
    # Log final messages
    log_str(f"Results saved to:")
    log_str(f"  CSV: {csv_path}")
    log_str(f"  Text analysis: {txt_path}")

except Exception as e:
    log_str(f"Error during analysis: {e}")