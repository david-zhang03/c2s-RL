
import torch.nn.functional as F
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
import torch.nn as nn


class gene_program_model_gcn(torch.nn.Module):

    def __init__(self, in_dim: int, out_dim: int, conf):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        gcn_hidden_dim = conf.GCN_HIDDEN_DIM
        mlp_hidden_dim = conf.MLP_HIDDEN_DIM

        self.conv1 = SAGEConv(in_dim, gcn_hidden_dim)
        self.conv2 = SAGEConv(gcn_hidden_dim, gcn_hidden_dim)

        self.res_fc1 = nn.Linear(in_dim, gcn_hidden_dim)
        self.res_fc2 = nn.Linear(in_dim, gcn_hidden_dim)
        
        self.fc1 = nn.Linear(gcn_hidden_dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, mlp_hidden_dim)
        self.fc3 = nn.Linear(mlp_hidden_dim, out_dim)
        
        self.dropout = nn.Dropout(conf.GCN_DROPOUT_LAYER_P)

    def forward(self, x, edge_index):
        res1 = self.res_fc1(x)
        res2 = self.res_fc2(x)

        x = F.elu(self.conv1(x, edge_index))
        x = self.dropout(x)
        
        x = F.elu(self.conv2(x, edge_index)) + res1
        x = F.elu(self.fc1(x))
        
        x = F.elu(self.fc2(x)) + res2
        x = F.softplus(self.fc3(x))
        
        return x


    #     self.conv1 = GCNConv(in_dim, conf.GCN_HIDDEN_DIM)
    #     self.act1 = nn.ReLU()
    #     self.drop1 = nn.Dropout(p=conf.GCN_DROPOUT_LAYER_P)
    #     self.conv2 = GCNConv(conf.GCN_HIDDEN_DIM, conf.GCN_HIDDEN_DIM)

    #     # First fully connected layer
    #     self.fc1 = torch.nn.Linear(conf.GCN_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
    #     # Second fully connected layer
    #     self.fc2 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, conf.MLP_HIDDEN_DIM)
    #     # Third fully connected layer that outputs our result
    #     self.fc3 = torch.nn.Linear(conf.MLP_HIDDEN_DIM, out_dim)


    # def forward(self, x, edge_index):         
    #     x = F.relu(self.conv1(x, edge_index))
    #     x = self.drop1(x)
    #     x = F.relu(self.conv2(x, edge_index))
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x
    


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
