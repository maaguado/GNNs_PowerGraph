from torch_geometric_temporal.nn.recurrent import AGCRN
import torch
import torch.nn.functional as F
from utils.dygrae import DyGrEncoder
from utils.mpnn_lstm_dyn import MPNNLSTM
import torch.nn as nn

class AGCRNModel(torch.nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, name, n_target, k=2):
        self.name  =name
        self.n_nodes = n_nodes
        self.n_target = n_target
        self.n_features = n_features
        self.embedding_dim =embedding_dim
        self.k = k
        super(AGCRNModel, self).__init__()

        self.recurrent = AGCRN(number_of_nodes = n_nodes,
                              in_channels = n_features,
                              out_channels = n_nodes,
                              K = self.k,
                              embedding_dimensions = embedding_dim)
        self.linear = torch.nn.Linear(n_nodes,n_target)



    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        y = self.linear(y)
        return y, h_0


class DyGrEncoderModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, num_conv=1,  num_lstm=1, aggr="mean"):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        super(DyGrEncoderModel, self).__init__()
        self.recurrent = DyGrEncoder(conv_out_channels=100, conv_num_layers=num_conv, conv_aggr=aggr, lstm_out_channels=self.n_features, lstm_num_layers=num_lstm)
        self.linear = torch.nn.Linear(self.n_features, n_target)
        self.h =None

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h, h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h)
        h = self.linear(h)
        self.h = h
        return h, h_0, c_0


class MPNNLSTMModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, hidden_size, window=1, dropout=0.5, is_classification=False):
        super(MPNNLSTMModel, self).__init__()
        self.name = name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.hidden_size = hidden_size
        self.window = window
        self.is_classification = is_classification
        self.recurrent = MPNNLSTM(self.n_features, self.hidden_size, self.n_nodes, self.window, dropout)
        
        self.linear = torch.nn.Linear(2 * self.hidden_size + self.n_features + self.window - 1, self.n_target)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        
        if self.is_classification:
            h_avg = torch.mean(h, dim=0) 
            h_out = self.linear(h_avg)
            
            h_out = torch.softmax(h_out, dim=0) 
            return h_out
        else:
            # En caso de regresión, se procesa cada nodo por separado
            h_out = self.linear(h)
            return h_out


class LSTMModel(nn.Module):
    def __init__(self, name, node_features, node_count, n_target, hidden_size=50, num_layers=1, is_classification=False):
        super(LSTMModel, self).__init__()
        self.name = name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.is_classification = is_classification  
        
        self.lstm = nn.LSTM(self.n_features, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, n_target) 

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        
        if self.is_classification:
            out = out[:, -1, :]
            out = self.fc(out)
            out = torch.softmax(out, dim=1) 
        else:
            out = self.fc(out) 
        return out