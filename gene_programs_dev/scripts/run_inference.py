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
    inference_data = Data(x=inference_X, edge_index=inference_edge_list)

    inference_neighbor_sampler = NeighborSampler(
        inference_data,
        num_neighbors=[train_config.GSAMP_NUM_NBR, train_config.GSAMP_NUM_NBR]
    )

    inference_loader = NodeLoader(
        inference_data,
        node_sampler=inference_neighbor_sampler,
        batch_size=train_config.GSAMP_BATCH_SIZE,  # number of seed nodes
        num_workers=train_config.GSAMP_NUM_WORKER,
    )
    
    n_cell, n_genes = input_x.shape
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    log_str(f"Using device: {device}")

    model = trained_model.to(device)
    model.eval()

    dummy_x = torch.zeros(1, n_genes, device=device)
    dummy_edges = torch.tensor([range(1), range(1)], dtype=torch.long, device=device)
    dummy_out = model(dummy_x, dummy_edges)
    _, n_progs = dummy_out.shape
    output_scores = np.zeros((n_cell, n_progs))

    log_str(f"Running inference: {n_cell} cells - {n_genes} genes - {n_progs} programs")
    with torch.no_grad():
        for idx, subgraph in tqdm(enumerate(inference_loader)):
            n_nodes, _ = subgraph.x.shape
            original_indices = subgraph.input_id
            subgraph = subgraph.to(device)
            batch_identity_adj = torch.tensor([range(n_nodes), range(n_nodes)], dtype=torch.long, device=device)
            edges = torch.cat([batch_identity_adj, subgraph.edge_index], dim=1)
            batch_program_scores = model(subgraph.x, edges)
            seed_nodes_prog_scores = batch_program_scores[:len(original_indices), :].detach().cpu().numpy()
            output_scores[original_indices, :] = seed_nodes_prog_scores
            
    return output_scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint')
    parser.add_argument('--prepared_train_inputs', type=str, required=True, help='Path to the prepared training inputs')
    parser.add_argument('--output_save_path', type=str, required=True, help='Path to save the output scores and analysis')
    parser.add_argument('--output_csv_save_path', type=str, required=True, default="/home/ddz5/Desktop/c2s-RL/gene_programs_dev/gene_set_data", help='Path to save the csv file')

    args = parser.parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_save_path, exist_ok=True)

    conf = gnn_config()
    log_str(f"Reading prepared input at {os.path.abspath(args.prepared_train_inputs)}")

    try:
        with open(args.prepared_train_inputs, 'rb') as f:
            inputs = pickle.load(f)
    except Exception as e:
        log_str(f"Error loading input data: {e}")
        sys.exit(1)

    # Extract necessary data
    try:
        progs = inputs['progs_sparse'].toarray()
        num_prog, num_genes = progs.shape
        
        n_cell, _ = inputs['train_x_sparse'].shape
        input_train_x = inputs['train_x_sparse'].toarray() 
        
        edge_list = np.array(inputs['train_edges'])
    except KeyError as e:
        log_str(f"Missing key in input data: {e}")
        sys.exit(1)
    except Exception as e:
        log_str(f"Error processing input data: {e}")
        sys.exit(1)

    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    log_str(f"Using device for model: {device}")

    # Load model
    try:
        trained_model = models.get_gnn_model(num_genes, num_prog, conf).to(device)
        checkpoint = torch.load(args.model_path, map_location=device)
        trained_model.load_state_dict(checkpoint['model_state_dict'])
        log_str("Model loaded successfully")
    except Exception as e:
        log_str(f"Error loading model: {e}")
        sys.exit(1)

    # Run inference
    try:
        output_scores = infer_gnn(input_train_x, edge_list, trained_model, conf)
        
        # Save inference outputs
        output_scores_path = os.path.join(args.output_save_path, "output_scores.pickle")
        with open(output_scores_path, 'wb') as f:
            pickle.dump(output_scores, f)
        log_str(f"Inference scores saved to {output_scores_path}")
    except Exception as e:
        log_str(f"Error during inference: {e}")
        sys.exit(1)

    # Load AnnData file and perform analysis
    try:
        # avoid loading in adata
        # adata = ad.read_h5ad(inputs['input_h5ad_file_path'])
        
        # Check required columns
        # if 'cell_type' not in adata.obs.columns:
        #     log_str("Error: Required column 'cell_type' not found in data")
        #     sys.exit(1)
        
        # if 'disease' not in adata.obs.columns:
        #     log_str("Error: Required column 'disease' not found in data")
        #     sys.exit(1)

        if 'cell_types' not in inputs:
            log_str("Error: Required column 'cell_types' not found in input")
            sys.exit(1)

        if 'disease' not in inputs:
            log_str("Error: Required column 'disease' not found in input")
            sys.exit(1)

            
        # Create DataFrame with scores and metadata
        output_scores_df = pd.DataFrame(output_scores, columns=inputs['gene_set_names'])
        output_scores_df['cell_type'] = inputs['cell_types']
        output_scores_df['disease'] = inputs['disease']
        
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
                
                # Get top 100 programs
                top_programs = mean_scores.nlargest(100)
                
                # Get bottom 25 programs (probably just scores of 0), sparse
                bottom_programs = mean_scores.nsmallest(25)
                
                # Write to both console and file
                header = f"\nDisease: {disease}, Cell type: {cell_type}"
                log_str(header)
                txt_file.write(header + "\n")
                
                subheader = "Top 100 gene programs:"
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
                        'Mean Activity Score': score,
                        'Rank Type': 'Top'
                    })
                
                subheader = "\nBottom 25 gene programs:"
                log_str(subheader)
                txt_file.write(subheader + "\n")
                
                for program, score in bottom_programs.items():
                    result_line = f"  {program}: {score:.3f}"
                    log_str(result_line)
                    txt_file.write(result_line + "\n")
                    
                    results_list.append({
                        'Disease': disease,
                        'Cell Type': cell_type,
                        'Gene Program': program,
                        'Mean Activity Score': score,
                        'Rank Type': 'Bottom'
                    })
        
        # Save results
        results_df = pd.DataFrame(results_list)
        # Note .csv file is saved to a separate path
        csv_output_save_path = os.path.join(args.output_csv_save_path, os.path.basename(args.output_save_path))
        os.makedirs(csv_output_save_path, exist_ok=True)
        
        csv_path = os.path.join(csv_output_save_path, "gene_program_rankings.csv")
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
    
    except FileNotFoundError as e:
        log_str(f"File not found: {e}")
        sys.exit(1)
    except KeyError as e:
        log_str(f"Missing key in data: {e}")
        sys.exit(1)
    except Exception as e:
        log_str(f"Error during analysis: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()