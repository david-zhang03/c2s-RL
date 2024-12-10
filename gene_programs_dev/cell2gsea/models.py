
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
import torch.nn as nn



class gene_program_model_gcn(torch.nn.Module):

    def __init__(self, in_dim: int,out_dim: int, conf):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = GCNConv(in_dim, conf.GCN_HIDDEN_DIM)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=conf.GCN_DROPOUT_LAYER_P)
        self.conv2 = GCNConv(conf.GCN_HIDDEN_DIM, conf.GCN_HIDDEN_DIM)

        # First fully connected layer
        self.fc1 = torch.nn.Linear(conf.GCN_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
        # Second fully connected layer
        self.fc2 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
        # Third fully connected layer that outputs our result
        self.fc3 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, out_dim)




    def forward(self, x, edge_index):         
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    






class gene_program_model_gcn_nonneg(torch.nn.Module):

    def __init__(self, in_dim: int,out_dim: int, conf):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv1 = GCNConv(in_dim, conf.GCN_HIDDEN_DIM)
        self.act1 = nn.ReLU()
        self.drop1 = nn.Dropout(p=conf.GCN_DROPOUT_LAYER_P)
        self.conv2 = GCNConv(conf.GCN_HIDDEN_DIM, conf.GCN_HIDDEN_DIM)

        # First fully connected layer
        self.fc1 = torch.nn.Linear(conf.GCN_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
        # Second fully connected layer
        self.fc2 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
        # Third fully connected layer that outputs our result
        self.fc3 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, out_dim)




    def forward(self, x, edge_index):         
        x = F.relu(self.conv1(x, edge_index))
        x = self.drop1(x)
        x = F.relu(self.conv2(x, edge_index))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x





def get_gnn_model(in_dim, out_dim, conf):
    if not conf.NONNEG:
        return gene_program_model_gcn(in_dim, out_dim, conf)
    else:
        return gene_program_model_gcn_nonneg(in_dim, out_dim, conf)
