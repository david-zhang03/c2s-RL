from datetime import datetime
import random
import os
import sys
import pickle
import json
# import yaml

import anndata as ad
import scanpy as sc
import wandb
import torch
import numpy as np
from tqdm import tqdm

from torch_geometric.data import Data
from torch_geometric.loader import NodeLoader
from torch_geometric.sampler.neighbor_sampler import NeighborSampler

from scipy import sparse
from sklearn.metrics import r2_score, roc_auc_score, log_loss, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import umap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt 

sys.path.append('..')
from cell2gsea.models import *
from cell2gsea.utils import *


class gnn_config:
    def __init__(self,
        # model hyperparameters
        MODEL_TYPE = "GCN_MLP",
        GCN_HIDDEN_DIM = 128,
        MLP_HIDDEN_DIM = 128,
        GCN_DROPOUT_LAYER_P = 0.5,
        NONNEG = True,
        # KNN graph hyperparameters
        KNN_GRAPH_K = 5,
        KNN_GRAPH_N_PCA = 50,
        # Training hyperparameters
        PROGRAM_NORMALIZE = True,
        PROGRAM_DROPOUT = 0.5, # used only when prog_groups is None
        PROGRAM_L1_PENALTY=0,
        LEARNING_RATE = 0.001,
        MIN_LRN_RATE = 1e-7,
        WEIGHT_DECAY = 1e-5,
        N_EPOCHS = 250,   
        VAL_SPLIT = 0.2,     
        # Graph sampling hyperparameters
        GSAMP_BATCH_SIZE = 10,  # number of seed nodes
        GSAMP_NUM_WORKER = 1,
        GSAMP_NUM_NBR = 3,
        CLUSTER_RESOL = 300,
        # General
        SEED = 42,
        UMAP_PLOTTING=50, # plotting every X epochs
        **kwargs
        ):
        # store config parameters in the config object (set the parameters as attributes)
        args_dict = locals().copy()
        args_dict.pop('self')
        for key, value in args_dict.items():
            setattr(self, key, value)

    def from_yaml(self,fpath):
        with open(fpath, 'r') as file:
            conf_dict = yaml.safe_load(file)
        for key, value in conf_dict.items():
            setattr(self, key, value)


def train_gnn(
        train_x=None,
        validation_x=None,
        train_edges = None,
        validation_edges = None,
        progs=None,
        prog_groups = None,
        prog_cluster_labels = None,
        train_tru_labels= None,
        train_cell_types = None,
        val_tru_labels= None,
        val_cell_types = None,
        training_config=None,
        wandb_run = None, 
        base_model = None,
        OUTPUT_PREFIX = "./cell2gsea_training_runs",
        ENABLE_SAVE_MODEL=True,
        ENABLE_OUTPUT_SCORES=False,
        REGRESSION=True,
        RUN_ID = None,
        RUN_NAME = "train_cell2gsea",
        ):

    print ("__________________________________", flush=True)
    script_path = os.path.abspath(__file__)
    log_str(f"Started running {script_path}")


    WANDB_LOGGING = False
    if wandb_run is not None:
        WANDB_LOGGING = True

    if RUN_ID is None:
        RUN_ID = f"{RUN_NAME}_{get_timestamp()}"

    conf_dict = training_config.__dict__
    conf = training_config
    SAVE_DIR = os.path.join(OUTPUT_PREFIX, RUN_ID)


    n_train , _ = train_x.shape
    # conf.NUM_TRAIN_EXAMPLES = n_train


    label_list = list(range(progs.shape[0]))
    if (train_tru_labels is not None and not REGRESSION):
        train_labels_exist = True
    else:
        log_str("Labels are not provided for training data. Ignore classification scores (AUC, Accuracy, etc.) for training data")
        train_labels_exist = False
        train_tru_labels = np.zeros(n_train)


    if validation_x is not None:
        n_validation , _ = validation_x.shape
        validation_exist= True
        if (val_tru_labels is not None and not REGRESSION):
            validation_labels_exist = True
        else:
            validation_labels_exist = False
            val_tru_labels = np.zeros(n_validation)
            log_str("Labels are not provided for validation data. Ignore classification scores (AUC, Accuracy, etc.) for validation data")
    else:
        log_str("Validation data is not provided. Continuing without validation")
        n_validation = 0
        validation_exist = False
        validation_labels_exist = False
        val_tru_labels = np.zeros(n_validation)

    # conf.NUM_VALIDATION_EXAMPLES = n_validation


    if train_edges is not None:
        log_str("Using provided graph for training data")
        train_edge_list = np.array(train_edges)
    else:
        log_str(f"Input graph was not provided for the training data. Building KNN graph with K={conf.KNN_GRAPH_K} using {conf.KNN_GRAPH_N_PCA} PCA dimensions.")
        train_edge_list = get_knn_edges(train_x,conf.KNN_GRAPH_K,conf.KNN_GRAPH_N_PCA)


    if validation_exist:
        if validation_edges is not None:
            validation_edge_list = np.array(validation_edges)
        else:
            log_str(f"Input graph was not provided for the validation data. Building KNN graph with K={conf.KNN_GRAPH_K} using {conf.KNN_GRAPH_N_PCA} PCA dimensions.")
            validation_edge_list = get_knn_edges(validation_x,conf.KNN_GRAPH_K,conf.KNN_GRAPH_N_PCA)

    
    if prog_groups is None:
        if prog_cluster_labels is not None:
            log_str("Grouping of programs were provided as cluster labels. Ignoring PROGRAM_DROPOUT.")
            unique_cluster_labels = np.unique(prog_cluster_labels)
            prog_groups = []
            for cluster_label in unique_cluster_labels:
                prog_groups.append(np.where(prog_cluster_labels == cluster_label)[0].tolist())

    else:
        log_str("Grouping of programs were provided as list of groups. Ignoring PROGRAM_DROPOUT.")
        
    if prog_groups is None:
        n_prog_groups = 0
        log_str(f"No grouping of programs. using program dropout (dropout ratio = {conf.PROGRAM_DROPOUT})")
    else:
        n_prog_groups = len(prog_groups)
        log_str(f"Using {n_prog_groups} program groups")
    # conf.N_PROG_GROUPS=n_prog_groups


    os.makedirs(SAVE_DIR, exist_ok=True)
    log_str(f"Run ID: {RUN_ID}\tOutput directory: {SAVE_DIR}")
    # conf.RUN_ID = RUN_ID
    print ("__________________________________",flush=True)

    # Check if CUDA is available, else check for MPS, otherwise default to CPU
    device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    log_str(f"Using device: {device_name}")
    device = torch.device(device_name)


    program_def = progs
    prog_def_tensor = torch.tensor(program_def, dtype=torch.float32).to(device)
    num_prog, num_genes = prog_def_tensor.shape
    # conf.NUM_PROG = num_prog
    # conf.NUM_VARS = num_genes


    n_train_edges = len(train_edge_list[0])
    log_str(f"Preparing inputs with:")
    log_str(f"{n_train:,} training examples. number of edges in the graph: {n_train_edges:,}")
    n_val_edges = 0

    if validation_exist:
        n_val_edges = len(validation_edge_list[0])
        log_str(f"{n_validation:,} validation examples. number of edges in the graph: {n_val_edges:,}")
    else:
        log_str("No validation.")
    
    # conf.N_TRAIN_EDGES = n_train_edges
    # conf.N_VAL_EDGES = n_val_edges


    train_X = torch.tensor(train_x, dtype=torch.float32)
    train_labels = torch.tensor(train_tru_labels, dtype=torch.long)
    train_edge_list = torch.tensor(train_edge_list, dtype=torch.long)
    train_node_pos = torch.arange(n_train).reshape(n_train,1)
    train_data = Data(x=train_X , edge_index=train_edge_list,y=train_labels,pos = train_node_pos)


    if validation_exist:
        validation_X= torch.tensor(validation_x, dtype=torch.float32)
        validation_labels = torch.tensor(val_tru_labels, dtype=torch.long)
        validation_edge_list = torch.tensor(validation_edge_list, dtype=torch.long)
        validation_node_pos = torch.arange(n_validation).reshape(n_validation,1)
        validation_data = Data(x=validation_X , edge_index=validation_edge_list,y=validation_labels,pos=validation_node_pos)


    if base_model is None:
        model = get_gnn_model(num_genes,num_prog,conf).to(device)
    else:
        model = base_model

    log_memory_usage("Load model", device)

    def count_parameters(model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return total_params, trainable_params

    total_params, trainable_params = count_parameters(model)
    log_str(f"Built model with {trainable_params:,} trainable parameters, and {total_params:,} total parameters:")
    print(model,flush = True)

    conf.model_desc = str(model)

    # Log config arguments
    config_dict = {attr: getattr(conf, attr) for attr in dir(conf) 
               if not attr.startswith('__') and not callable(getattr(conf, attr))}

    config_dict.update({
        'num_train_examples': n_train,
        'num_validation_examples': n_validation,
        'num_programs': progs.shape[0],
        'num_genes': progs.shape[1],
    })

    with open(os.path.join(SAVE_DIR, 'arguments.txt'), 'w') as f:
        for key, value in config_dict.items():
            f.write(f"{key}: {value}")
    
    wandb.config.update(config_dict)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LEARNING_RATE, weight_decay=conf.WEIGHT_DECAY)
    iterations_per_anneal_cycle = conf.N_EPOCHS 
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=iterations_per_anneal_cycle, eta_min=conf.MIN_LRN_RATE)

    train_neighbor_sampler = NeighborSampler(
        train_data,
        num_neighbors=[conf.GSAMP_NUM_NBR, conf.GSAMP_NUM_NBR]
    )

    train_loader = NodeLoader(
        train_data,
        node_sampler=train_neighbor_sampler,
        batch_size = conf.GSAMP_BATCH_SIZE,  # number of seed nodes
        num_workers = conf.GSAMP_NUM_WORKER,
    )


    if validation_exist:
        validation_neighbor_sampler = NeighborSampler(
            validation_data,
            num_neighbors=[conf.GSAMP_NUM_NBR, conf.GSAMP_NUM_NBR]
        )

        validation_loader = NodeLoader(
            validation_data,
            node_sampler = validation_neighbor_sampler,
            batch_size = conf.GSAMP_BATCH_SIZE,  # number of seed nodes
            num_workers = conf.GSAMP_NUM_WORKER,
        )


    if ENABLE_OUTPUT_SCORES:
        training_assigned_prog_scores = np.zeros((n_train,num_prog))
        output_size = sys.getsizeof(training_assigned_prog_scores)
        log_str(f"Allocated {format_size(output_size)} for output scores from training set")
    
    training_num_rep = np.zeros(n_train)
    training_source_epoch= np.array([-1]*n_train)
    training_cell_r2 = np.zeros(n_train)
    training_gene_r2 = np.zeros(num_genes)
    tr_batch_idx = 0
    val_batch_idx = 0
    
    if validation_exist:
        if ENABLE_OUTPUT_SCORES:
            validation_assigned_prog_scores = np.zeros((n_validation,num_prog))
            output_size = sys.getsizeof(validation_assigned_prog_scores)
            log_str(f"Allocated {format_size(output_size)} for output scores from validation set")
        validation_num_rep = np.zeros(n_validation)
        validation_source_epoch= np.array([-1]*n_validation)
        validation_cell_r2 = np.zeros(n_validation)
        validation_gene_r2 = np.zeros(num_genes)

    
    print ("__________________________________",flush=True)
    log_str(f"Started training...")

    if validation_exist:
        best_val_loss = 1e9
    for epoch in tqdm(range(conf.N_EPOCHS)):
        temp_train_loss = []
        temp_train_pearson_r = []
        temp_train_r2_score = []
        temp_train_cross_entropy = []
        temp_train_auc = []
        temp_train_acc = []
        model.train()
        default_auc = 0.5


        for idx, subgraph in enumerate(tqdm(train_loader)):

            #### Program dropout
            if (prog_groups is None) and (conf.PROGRAM_DROPOUT == 0):
                ## use all programs
                filtered_prog_def_tensor = prog_def_tensor
                keep_indices = list(range(num_prog))

            elif (prog_groups is None) and (conf.PROGRAM_DROPOUT > 0):
                ## randomly choose specified portion of the program matrix
                n_keep = int(num_prog * (1-conf.PROGRAM_DROPOUT))
                keep_indices = random.sample(range(num_prog), n_keep)
                filtered_prog_def_tensor = prog_def_tensor[keep_indices,:]

            else:
                ## randomly pick one representative from each group of programs 
                keep_indices = []
                for g in prog_groups:
                    representative_program_index = random.choice(g)
                    keep_indices.append(representative_program_index)
                filtered_prog_def_tensor = prog_def_tensor[keep_indices,:]


            n_nodes, _ = subgraph.x.shape
            x_array = subgraph.x.detach().numpy()
            flattened_x = x_array.flatten()
            original_indices = subgraph.n_id
            optimizer.zero_grad()
            subgraph = subgraph.to(device)
            
            # add self-loops to the edges
            self_loop_edges = torch.tensor([range(n_nodes),range(n_nodes)], dtype=torch.long,device=device)
            edges = torch.cat([self_loop_edges,subgraph.edge_index],dim = 1)
            
            # forward-pass through the model
            batch_program_scores = model(subgraph.x,edges)
            
            X_reconst = torch.matmul(batch_program_scores[:,keep_indices],filtered_prog_def_tensor)  # [cells, gene_programs] x [gene_programs, num_genes]
            if conf.PROGRAM_L1_PENALTY > 0:
                loss = criterion(subgraph.x, X_reconst) + conf.PROGRAM_L1_PENALTY * torch.norm(batch_program_scores[:,keep_indices], p = 1)
            else:
                loss = criterion(subgraph.x, X_reconst)
            loss.backward()


            prog_scores_cpu = batch_program_scores.detach().cpu().numpy()

            if ENABLE_OUTPUT_SCORES:
                ## write back scores
                training_assigned_prog_scores[original_indices,:] = prog_scores_cpu
            
            training_num_rep[original_indices] += 1
            training_source_epoch[original_indices] = epoch

            reconst_array = X_reconst.detach().cpu().numpy()
            flattened_reconst = reconst_array.flatten()
            r2_value = r2_score(flattened_x, flattened_reconst)
            batch_cell_r2 = r2_score(x_array.T, reconst_array.T,multioutput="raw_values")
            batch_gene_r2 = r2_score(x_array,reconst_array,multioutput="raw_values")

            ## write back cell and gene r2
            training_cell_r2[original_indices] = (training_num_rep[original_indices] * training_cell_r2[original_indices] + batch_cell_r2 ) / (1+training_num_rep[original_indices])
            training_gene_r2 = (tr_batch_idx * training_gene_r2 + batch_gene_r2) / (1 + tr_batch_idx)
            ##

            pearson_r_value = np.corrcoef(flattened_x, flattened_reconst)[0, 1]
            loss_val = loss.item()
            
            if train_labels_exist:
                proba = batch_program_scores.softmax(dim=1)
                proba_cpu = proba.detach().cpu().numpy()
                batch_labels_cpu = subgraph.y.detach().cpu().numpy()
                cross_entropy_val = log_loss(y_true=batch_labels_cpu, y_pred=proba_cpu,labels=label_list)
                top_prg = np.argmax(proba_cpu,axis = 1)
                acc = accuracy_score(batch_labels_cpu,top_prg)
           
                try:
                    auc = roc_auc_score(y_true=batch_labels_cpu, y_score=proba_cpu, multi_class='ovr',labels=label_list,average='macro')
                    default_auc  = auc
                    temp_train_auc.append(auc)
                    
                except:
                    auc = default_auc
                    log_str(f"No AUC: epoch {epoch}- batch {idx}")


            temp_train_pearson_r.append(pearson_r_value)            
            if not np.isnan(r2_value):
                temp_train_r2_score.append(r2_value)
            
            temp_train_loss.append(loss_val)

            if train_labels_exist:
                temp_train_cross_entropy.append(cross_entropy_val)
                temp_train_acc.append(acc)


            optimizer.step()
            scheduler.step(epoch + idx / len(train_loader)) # Adjust learning rate
            tr_batch_idx += 1

        log_memory_usage("Load train", device)

        avg_train_loss = np.mean(temp_train_loss)
        avg_train_r2_score = np.mean(temp_train_r2_score)
        avg_train_pearsonr_score = np.mean(temp_train_pearson_r)
        
        if train_labels_exist:
            avg_train_cross_entropy = np.mean(temp_train_cross_entropy)
            avg_train_auc = np.mean(temp_train_auc)
            avg_train_acc = np.mean(temp_train_acc)
        else:
            avg_train_cross_entropy = float('nan')
            avg_train_auc = float('nan')
            avg_train_acc = float('nan')


        if validation_exist:
            temp_val_loss = []
            temp_val_pearson_r = []
            temp_val_r2_score = []
            temp_val_cross_entropy = []
            temp_val_auc = []
            temp_val_acc = []

            model.eval()
            default_auc = 0.5

            log_str(f"Started validation for epoch {epoch}...")

            with torch.no_grad():
                for idx, subgraph in enumerate(tqdm(validation_loader)):

                    #### Program dropout
                    if (prog_groups is None) and (conf.PROGRAM_DROPOUT == 0):
                        ## use all programs
                        filtered_prog_def_tensor = prog_def_tensor
                        keep_indices = list(range(num_prog))

                    elif (prog_groups is None) and (conf.PROGRAM_DROPOUT > 0):
                        ## randomly choose specified portion of the program matrix
                        n_keep = int(num_prog * (1-conf.PROGRAM_DROPOUT))
                        keep_indices = random.sample(range(num_prog), n_keep)
                        filtered_prog_def_tensor = prog_def_tensor[keep_indices,:]

                    else:
                        ## randomly pick one representative from each group of programs 
                        keep_indices = []
                        for g in prog_groups:
                            representative_program_index = random.choice(g)
                            keep_indices.append(representative_program_index)
                        filtered_prog_def_tensor = prog_def_tensor[keep_indices,:]

                    n_nodes, _ = subgraph.x.shape

                    x_array = subgraph.x.detach().numpy()
                    flattened_x = x_array.flatten()

                    # original indices in the val set
                    original_indices = subgraph.pos.reshape(-1).numpy()      
                    subgraph = subgraph.to(device)

                    batch_identity_adj = torch.tensor([range(n_nodes),range(n_nodes)], dtype=torch.long,device=device)
                    edges = torch.cat([batch_identity_adj,subgraph.edge_index],dim = 1)
                    batch_program_scores = model(subgraph.x,edges)
                    prog_scores_cpu = batch_program_scores.detach().cpu().numpy()

                    if ENABLE_OUTPUT_SCORES:
                        # write back scores
                        validation_assigned_prog_scores[original_indices,:] = prog_scores_cpu
                    
                    validation_num_rep[original_indices] += 1
                    validation_source_epoch[original_indices] = epoch

                    
                    X_reconst = torch.matmul(batch_program_scores[:,keep_indices],filtered_prog_def_tensor)  # [cells, gene_programs] x [gene_programs, num_genes]

                    if conf.PROGRAM_L1_PENALTY > 0:
                        loss = criterion(subgraph.x, X_reconst) + conf.PROGRAM_L1_PENALTY * torch.norm(batch_program_scores[:,keep_indices], p = 1)
                    else:
                        loss = criterion(subgraph.x, X_reconst)

                    
                    reconst_array = X_reconst.detach().cpu().numpy()
                    flattened_reconst =  reconst_array.flatten()
                    
                    batch_cell_r2 = r2_score(x_array.T,reconst_array.T, multioutput="raw_values")
                    batch_gene_r2 = r2_score(x_array,reconst_array, multioutput="raw_values")

                    ## write back cell and gene r2
                    validation_cell_r2[original_indices] = (validation_num_rep[original_indices] * validation_cell_r2[original_indices] + batch_cell_r2 ) / (1+validation_num_rep[original_indices])
                    validation_gene_r2 = (val_batch_idx * validation_gene_r2 + batch_gene_r2) / (1 + val_batch_idx)
                    ##


                    #### plot example reconstructions
                    # num_batch_x_vals = len(flattened_x)
                    # samp_idx = np.random.randint(0, num_batch_x_vals , size=1000)
                    # plt.close('all')
                    # fig,ax = plt.subplots()
                    # ax.scatter(flattened_x[samp_idx],flattened_reconst[samp_idx])
                    # plt.ylabel('reconstructed')
                    # plt.xlabel('original')
                    ####


                    r2_value = r2_score(flattened_x, flattened_reconst)
                    pearson_r_value = np.corrcoef(flattened_x, flattened_reconst)[0, 1]
                    loss_val = loss.item()


                    if validation_labels_exist:
                        proba = batch_program_scores.softmax(dim=1)
                        proba_cpu = proba.detach().cpu().numpy()
                        batch_labels_cpu = subgraph.y.detach().cpu().numpy()
                        cross_entropy_val = log_loss(y_true=batch_labels_cpu, y_pred=proba_cpu,labels=label_list)
                        top_prg = np.argmax(proba_cpu,axis = 1)
                        acc = accuracy_score(batch_labels_cpu,top_prg)
                        
                        try:
                            auc = roc_auc_score(y_true=batch_labels_cpu, y_score=proba_cpu, multi_class='ovr',labels=label_list,average='macro')
                            default_auc  = auc                       
                        except:
                            auc = default_auc
                            log_str(f"No AUC: epoch {epoch}- batch {idx}")

                    temp_val_pearson_r.append(pearson_r_value)
                    if not np.isnan(r2_value):
                        temp_val_r2_score.append(r2_value)
                    temp_val_loss.append(loss_val)

                    if validation_labels_exist:
                        temp_val_cross_entropy.append(cross_entropy_val)
                        temp_val_acc.append(acc)
                        temp_val_auc.append(auc)

                    val_batch_idx +=1


            log_memory_usage("Load val", device)
            avg_validation_loss = np.mean(temp_val_loss)
            avg_validation_r2_score = np.mean(temp_val_r2_score)
            avg_validation_pearsonr_score = np.mean(temp_val_pearson_r)

            if avg_validation_loss < best_val_loss:
                best_val_loss = avg_validation_loss
                if ENABLE_SAVE_MODEL:
                    model_save_path = os.path.join(SAVE_DIR, "best_model_checkpoint.pt")
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict()}, model_save_path)
                    # log_str(f"At epoch {epoch}, saved model checkpoint at {os.path.abspath(model_save_path)}")


            # plot umap of 
            if ENABLE_OUTPUT_SCORES and (epoch) % conf.UMAP_PLOTTING == 0 and val_cell_types:
                # Apply UMAP
                adata = ad.AnnData(X=validation_assigned_prog_scores)
                adata.obs['cell_type'] = val_cell_types

                sc.pp.pca(adata)
                sc.pp.neighbors(adata)
                sc.tl.umap(adata)

                plt.figure(figsize=(10, 10))
                fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                sc.pl.umap(adata, 
                        ax=ax,
                        color="cell_type",
                        title=f"UMAP of Validation Output Scores at Epoch {epoch}",
                        size=5)

                umap_plot_path = os.path.join(SAVE_DIR, f"umap_validation_epoch_{epoch}.png")
                plt.savefig(umap_plot_path, bbox_inches='tight', dpi=400)
                plt.close('all')

                # Log to wandb
                if WANDB_LOGGING:
                    wandb.log({
                        f"UMAP_Validation_Epoch_{epoch}": wandb.Image(umap_plot_path)
                    }, step=epoch)

            if validation_labels_exist:
                avg_validation_auc = np.mean(temp_val_auc)
                avg_validation_acc = np.mean(temp_val_acc)
                avg_validation_cross_entropy = np.mean(temp_val_cross_entropy)
        
            else:
                avg_validation_cross_entropy = float('nan')
                avg_validation_auc = float('nan')
                avg_validation_acc = float('nan')

        if not validation_exist:                
                avg_validation_cross_entropy = float('nan')
                avg_validation_auc = float('nan')
                avg_validation_acc = float('nan')
                avg_validation_loss = float('nan')
                avg_validation_r2_score =float('nan')
                avg_validation_pearsonr_score = float('nan')

        if WANDB_LOGGING:
            wandb.log({
                "Learning Rate": scheduler.get_last_lr()[0],
                "Training Loss (MSE-reconstruction)": avg_train_loss,
                "Training R2 (reconstruction)": avg_train_r2_score,
                "Training Pearson (reconstruction)": avg_train_pearsonr_score,
                "Training Cross Entropy (program scores - labels)": avg_train_cross_entropy,
                "Training AUC (program scores vs labels)": avg_train_auc,
                "Training Accuracy (top program vs labels)": avg_train_acc,
                "Training - score from epoch": wandb.Histogram(training_source_epoch),
                "Training - times in minibatch": wandb.Histogram(training_num_rep),
                "Training - Cell R2": wandb.Histogram(training_cell_r2),
                "Training - Gene R2": wandb.Histogram(training_gene_r2), 
            },step=epoch)
            if validation_exist:
                wandb.log({
                    # "Validation orig vs reconstructed" : wandb.Image(fig),
                    "Validation Loss (MSE-reconstruction)": avg_validation_loss,
                    "Validation R2 (reconstruction)": avg_validation_r2_score,
                    "Validation Pearson (reconstruction)": avg_validation_pearsonr_score,
                    "Validation Cross Entropy (program scores - labels)": avg_validation_cross_entropy,
                    "Validation AUC (program scores vs labels)": avg_validation_auc,
                    "Validation Accuracy (top program vs labels)": avg_validation_acc,
                    "Validation - score from epoch": wandb.Histogram(validation_source_epoch),
                    "Validation - times in minibatch": wandb.Histogram(validation_num_rep),
                    "Validation - Cell R2": wandb.Histogram(validation_cell_r2),
                    "Validation - Gene R2": wandb.Histogram(validation_gene_r2), 
                }, step=epoch)
   
    log_str("Finished training.")    
    if ENABLE_SAVE_MODEL:
        model_save_path = os.path.join(SAVE_DIR, "final_model.pt")
        torch.save({'epoch': epoch,
        'model_state_dict': model.state_dict()}, model_save_path)
        log_str(f"Final model saved at {os.path.abspath(model_save_path)}")
    print ("__________________________________", flush=True)
    
    output = {
        "training_num_rep": training_num_rep,
        "training_cell_r2": training_cell_r2,
        "training_gene_r2": training_gene_r2,
    }
    
    if ENABLE_OUTPUT_SCORES:
        output["training_source_epoch"] = training_source_epoch
        output["training_assigned_prog_scores"] = training_assigned_prog_scores

    if validation_exist:
        output.update({
        "validation_num_rep": validation_num_rep,        
        "validation_cell_r2": validation_cell_r2,
        "validation_gene_r2": validation_gene_r2,
        })
        if ENABLE_OUTPUT_SCORES:
            output["validation_source_epoch"] = validation_source_epoch
            output['validation_assigned_prog_scores'] = validation_assigned_prog_scores


    return output
    


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
    

    device_name = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
    log_str(f"Using device: {device_name}")
    device = torch.device(device_name)

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


if __name__ == '__main__':
    # Random example
    n_train = 1000
    n_validation = 100
    n_genes = 100
    n_progs = 10

    train_config = gnn_config(
    )

    training_output = train_gnn(
        train_x = np.random.rand(n_train,n_genes),
        train_edges = np.array([random.sample(range(n_train), n_train), random.sample(range(n_train), n_train)]),
        validation_x = np.random.rand(n_validation,n_genes),
        validation_edges = np.array([random.sample(range(n_validation), n_validation), random.sample(range(n_validation), n_validation)]),
        progs = np.random.rand(n_progs,n_genes),
        prog_groups = [[0,1],[2,3],[4,5],[6,7],[8,9]],
        # train_tru_labels = np.random.randint(0,n_progs,n_train),
        # val_tru_labels = np.random.randint(0,n_progs,n_validation),
        training_config = train_config
    )