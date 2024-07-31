from torch_geometric_temporal.nn.recurrent import AGCRN
import torch
import torch.nn.functional as F
from utils.dygrae import DyGrEncoder
from utils.mpnn_lstm_dyn import MPNNLSTM
from utils.mstgcn import MSTGCN
from torch_geometric_temporal.nn.recurrent import DCRNN
from utils.evolvegcn_dynamic import EvolveGCNO
from utils.astgcn import ASTGCN
from torch_geometric_temporal.nn.attention import MTGNN
from  utils.stgcn import STConv

import torch.nn as nn

class AGCRNModel(torch.nn.Module):
    def __init__(self, n_features, n_nodes, embedding_dim, hidden_size, name, n_target, k=2, is_classification=False):
        self.name  =name
        self.n_nodes = n_nodes
        self.n_target = n_target
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.embedding_dim =embedding_dim
        self.is_classification = is_classification
        self.k = k
        super(AGCRNModel, self).__init__()

        self.recurrent = AGCRN(number_of_nodes = n_nodes,
                              in_channels = n_features,
                              out_channels = hidden_size,
                              K = self.k,
                              embedding_dimensions = embedding_dim)
        self.linear = torch.nn.Linear(hidden_size,n_target)



    def forward(self, x, e, h):
        h_0 = self.recurrent(x, e, h)
        y = F.relu(h_0)
        if self.is_classification:
            y = torch.mean(y, dim=1) 
            y = self.linear(y)
            y = torch.softmax(y, dim=1) 
            return y, h_0
        else:
            # En caso de regresión, se procesa cada nodo por separado
            y = self.linear(y)
        return y, h_0

class EvolveGCNModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, is_classification=False):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.is_classification = is_classification
        super(EvolveGCNModel, self).__init__()
        self.recurrent = EvolveGCNO(self.n_features)
        self.linear = torch.nn.Linear(self.n_features, n_target)


    def forward(self, x, edge_index, edge_weight):
       
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        if self.is_classification:
            h_avg = torch.mean(h, dim=0) 
            h_out = self.linear(h_avg)
            h_out = torch.softmax(h_out, dim=0) 
        else:
            # En caso de regresión, se procesa cada nodo por separado
            h_out = self.linear(h)
        return h_out
    


class DyGrEncoderModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, conv_out =None, lstm_out =None ,num_conv=1,  num_lstm=1, aggr="mean", is_classification=False):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.conv_out = conv_out if conv_out != None else node_features
        self.lstm_out = lstm_out if lstm_out != None else node_features
        self.num_lstm = num_lstm
        self.num_conv = num_conv
        self.n_features = node_features
        self.is_classification = is_classification
        super(DyGrEncoderModel, self).__init__()
        self.recurrent = DyGrEncoder(conv_out_channels=self.conv_out, conv_num_layers=num_conv, conv_aggr=aggr, lstm_out_channels=self.lstm_out, lstm_num_layers=self.num_lstm)
        self.linear = torch.nn.Linear(self.lstm_out, n_target)
        self.h =None

    def forward(self, x, edge_index, edge_weight, h_0, c_0):
        h, h_0, c_0 = self.recurrent(x, edge_index, edge_weight, h_0, c_0)
        h = F.relu(h)
        if self.is_classification:
            h_avg = torch.mean(h, dim=0) 
            h_out = self.linear(h_avg)
            h_out = torch.softmax(h_out, dim=0) 
        else:
            # En caso de regresión, se procesa cada nodo por separado
            h_out = self.linear(h)
        self.h = h_out
        return h_out, h_0, c_0


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
            out = torch.mean(out, dim=1)
            out = self.fc(out)
            out = torch.softmax(out, dim=1) 
        else:
            out = self.fc(out) 
        return out
    


class MSTGCNModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, nb_block, hidden = 20,k=1, nb_chev_filter = 2, nb_time_filter =2, time_strides = 2, is_classification=False):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.nb_block = nb_block
        self.k = k
        self.is_classification = is_classification
        self.hidden = hidden
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides

    
        super(MSTGCNModel, self).__init__()
        self.recurrent = MSTGCN(in_channels=1, 
                                    num_for_predict=hidden if self.is_classification else n_target, 
                                    len_input=node_features,
                                    K=k, 
                                    nb_block=nb_block, 
                                    nb_chev_filter=nb_chev_filter,
                                    nb_time_filter=nb_time_filter, 
                                    time_strides=time_strides)
        if self.is_classification:
            self.linear = torch.nn.Linear(hidden,n_target)
    

    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        if self.is_classification:
            h = torch.mean(h, dim=1)
            h = self.linear(h)
            h = torch.softmax(h, dim=1) 
        return h



class DCRNNModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, hidden_dim, filter_size =1, is_classification=False):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.hidden_dim = hidden_dim
        self.is_classification = is_classification
        self.filter_size = filter_size
        super(DCRNNModel, self).__init__()
        self.recurrent = DCRNN(self.n_features,self.hidden_dim , self.filter_size)
        self.linear = torch.nn.Linear(self.hidden_dim, n_target)

    def forward(self, x, edge_index, edge_weight):
        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        if self.is_classification:
            h = torch.mean(h, dim=0)
            h = self.linear(h)
            h = torch.softmax(h, dim=0) 
        else:
            h = self.linear(h) 
        return h
    

class ASTGCNModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, nb_block, k=1, nb_chev_filter = 2, nb_time_filter =2, time_strides = 2, hidden_size = 20):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.nb_block = nb_block
        self.k = k
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides

    
        super(ASTGCNModel, self).__init__()
        self.recurrent = ASTGCN(in_channels=1, 
                                num_for_predict=n_target, 
                                len_input=node_features,
                                K=k, 
                                nb_block=nb_block, 
                                num_of_vertices=node_count,
                                nb_chev_filter=nb_chev_filter,
                                nb_time_filter=nb_time_filter, 
                                time_strides=time_strides)



    def forward(self, x, edge_index):
        h = self.recurrent(x, edge_index)
        h = F.relu(h)
        return h



class MTGNNModel(torch.nn.Module):
    def __init__(self, name, node_count, node_features, n_target, conv_channels, residual_channels, out_channels,skip_channels, end_channels,dilation_exponential=1, kernel_size=5, layers=2, propalpha=0.5, tanhalpha=0.3, dropout=0.3, layer_norm_affine=True, subgraph_size=3, gcn_depth=2, gcn_true=True, build_adj=True, is_classification=False):
        super(MTGNNModel, self).__init__()
        self.name = name
        self.n_nodes = node_count
        self.n_features = node_features
        self.n_target = n_target
        self.out_channels = out_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.end_channels = end_channels

        self.dilation_exponential = dilation_exponential
        self.layers = layers
        self.kernel_size = kernel_size
        self.dropout = dropout
        self.propalpha = propalpha
        self.tanhalpha = tanhalpha
        self.layer_norm_affine = layer_norm_affine
        self.kernel_set = [kernel_size] * conv_channels
        self.subgraph_size = subgraph_size

        self.gcn_true = gcn_true
        self.build_adj = build_adj
        self.gcn_depth = gcn_depth
        self.is_classification = is_classification
        self.recurrent = MTGNN(gcn_true=gcn_true, 
                   build_adj=build_adj, 
                   gcn_depth =gcn_depth, 
                   num_nodes=node_count, 
                   kernel_set = self.kernel_set, 
                   kernel_size =kernel_size, 
                   dropout=dropout, 
                   subgraph_size=subgraph_size, 
                   node_dim =1, 
                   dilation_exponential=dilation_exponential, 
                   conv_channels=conv_channels, 
                   residual_channels=residual_channels, 
                   skip_channels=skip_channels, 
                   end_channels=end_channels, 
                   seq_length=node_features, 
                   in_dim=1, out_dim=out_channels, 
                   layers=layers, propalpha=propalpha, 
                   tanhalpha=tanhalpha, 
                   layer_norm_affline=layer_norm_affine)
        self.linear = torch.nn.Linear(out_channels, n_target)

    def forward(self, x, matrix):
        h = self.recurrent(x, matrix)
        h = h.squeeze(0).squeeze(2).permute(1,0)
        if self.is_classification:
            h = torch.mean(h, dim=0)
            h = self.linear(h)
            h = torch.softmax(h, dim=0) 
        else:
            h = self.linear(h) 
        return h



class STConvModel(torch.nn.Module):
    def __init__(self, name, node_features, node_count, n_target, out_channels,k, kernel_size, hidden_channels, normalization="sym", is_classification=False):
        self.name  =name
        self.n_nodes = node_count
        self.n_target = n_target
        self.n_features = node_features
        self.out_channels = out_channels
        self.k = k
        self.kernel_size = kernel_size
        self.is_classification = is_classification
        self.normalization = normalization
        self.hidden_channels = hidden_channels
        super(STConvModel, self).__init__()
        self.recurrent = STConv(num_nodes=node_count,
                                in_channels=1,
                                hidden_channels=hidden_channels,
                                out_channels=out_channels,
                                kernel_size=kernel_size,
                                K=k,  # Grado del polinomio Chebyshev
                                normalization=normalization 
                            )

        self.linear = torch.nn.Linear(self.out_channels, self.n_target)


    def forward(self, x, edge_index, edge_weight):

        h = self.recurrent(x, edge_index, edge_weight)
        h = F.relu(h)
        h = h.squeeze(0).mean(dim=0)
        if self.is_classification:
            h = torch.mean(h, dim=0)
            h = self.linear(h)
            h = torch.softmax(h, dim=0) 
        else:
            h = self.linear(h) 
        return h
