import torch
from torch_geometric.nn import GCNConv


class TGCN(torch.nn.Module):
    r"""An implementation of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        node_dim : int=-2,
        add_self_loops: bool = True,
    ):
        super(TGCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.node_dim = node_dim
        self.add_self_loops = add_self_loops

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):

        self.conv_z = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
            node_dim = self.node_dim
        )

        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):

        self.conv_r = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
            node_dim = self.node_dim
        )

        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):

        self.conv_h = GCNConv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            add_self_loops=self.add_self_loops,
            node_dim = self.node_dim
        )

        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            H = torch.zeros(X.shape[0], self.out_channels).to(X.device)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        Z = torch.cat([self.conv_z(X, edge_index, edge_weight), H], axis=1)
        Z = self.linear_z(Z)
        Z = torch.sigmoid(Z)
        return Z

    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        R = torch.cat([self.conv_r(X, edge_index, edge_weight), H], axis=1)
        R = self.linear_r(R)
        R = torch.sigmoid(R)
        return R

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        H_tilde = torch.cat([self.conv_h(X, edge_index, edge_weight), H * R], axis=1)
        H_tilde = self.linear_h(H_tilde)
        H_tilde = torch.tanh(H_tilde)
        return H_tilde

    def _calculate_hidden_state(self, Z, H, H_tilde):
        H = Z * H + (1 - Z) * H_tilde
        return H

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
        H: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde)
        return H


class TGCN2(torch.nn.Module):
    r"""An implementation THAT SUPPORTS BATCHES of the Temporal Graph Convolutional Gated Recurrent Cell.
    For details see this paper: `"T-GCN: A Temporal Graph ConvolutionalNetwork for
    Traffic Prediction." <https://arxiv.org/abs/1811.05320>`_

    Args:
        in_channels (int): Number of input features.
        out_channels (int): Number of output features.
        batch_size (int): Size of the batch.
        improved (bool): Stronger self loops. Default is False.
        cached (bool): Caching the message weights. Default is False.
        add_self_loops (bool): Adding self-loops for smoothing. Default is True.
    """

    def __init__(self, in_channels: int, out_channels: int, 
                 batch_size: int,  # this entry is unnecessary, kept only for backward compatibility
                 improved: bool = False, cached: bool = False, 
                 add_self_loops: bool = True, node_dim: int=-2):
        super(TGCN2, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.batch_size = batch_size  # not needed
        self.node_dim = node_dim

        self._create_parameters_and_layers()

    def _create_update_gate_parameters_and_layers(self):
        self.conv_z = GCNConv(in_channels=self.in_channels,  out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops, node_dim = self.node_dim-1 )
        self.linear_z = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_reset_gate_parameters_and_layers(self):
        self.conv_r = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops, node_dim = self.node_dim-1 )
        self.linear_r = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_candidate_state_parameters_and_layers(self):
        self.conv_h = GCNConv(in_channels=self.in_channels, out_channels=self.out_channels, improved=self.improved,
                              cached=self.cached, add_self_loops=self.add_self_loops, node_dim = self.node_dim-1 )
        self.linear_h = torch.nn.Linear(2 * self.out_channels, self.out_channels)

    def _create_parameters_and_layers(self):
        self._create_update_gate_parameters_and_layers()
        self._create_reset_gate_parameters_and_layers()
        self._create_candidate_state_parameters_and_layers()

    def _set_hidden_state(self, X, H):
        if H is None:
            # can infer batch_size from X.shape, because X is [B, N, F]
            H = torch.zeros(X.shape[0], X.shape[1], self.out_channels).to(X.device) #(b, 207, 32)
        return H

    def _calculate_update_gate(self, X, edge_index, edge_weight, H):
        outputs = []
        for i in range(self.batch_size):  # Iterate over the batch dimension
            # Select data for the i-th graph in the batch
            X_i = X[i].view(X.size(self.node_dim), 1)
            edge_index_i = edge_index[i]
            edge_weight_i = edge_weight[i] if edge_weight is not None else None

            # Adjust edge_index_i to local node indices
            edge_index_i -= edge_index_i.min()

            # Apply GCNConv for the i-th graph
            Z_i = torch.cat([self.conv_z(X_i, edge_index_i, edge_weight_i), H[i]], dim=1)
            Z_i = self.linear_z(Z_i)
            Z_i = torch.sigmoid(Z_i)
            
            # Append the result for the i-th graph to the outputs list
            outputs.append(Z_i)
        
        return torch.stack(outputs, dim=0)


    def _calculate_reset_gate(self, X, edge_index, edge_weight, H):
        outputs = []
        for i in range(self.batch_size):  # Iterate over the batch dimension
            X_i = X[i].view(X.size(self.node_dim), 1)
            edge_index_i = edge_index[i]
            edge_weight_i = edge_weight[i] if edge_weight is not None else None

            edge_index_i -= edge_index_i.min()

            
            R_i = torch.cat([self.conv_r(X_i, edge_index_i, edge_weight_i), H[i]], dim=1)
            R_i = self.linear_r(R_i)
            R_i = torch.sigmoid(R_i)
            
            outputs.append(R_i)
        
        return torch.stack(outputs, dim=0)

    def _calculate_candidate_state(self, X, edge_index, edge_weight, H, R):
        outputs = []
        for i in range(self.batch_size):  # Iterate over the batch dimension
            X_i = X[i].view(X.size(self.node_dim), 1)
            edge_index_i = edge_index[i]
            edge_weight_i = edge_weight[i] if edge_weight is not None else None
            
            edge_index_i -= edge_index_i.min()

            H_tilde_i = torch.cat([self.conv_h(X_i, edge_index_i, edge_weight_i), H[i] * R[i]], dim=1)
            H_tilde_i = self.linear_h(H_tilde_i)
            H_tilde_i = torch.tanh(H_tilde_i)
            
            outputs.append(H_tilde_i)
        
        return torch.stack(outputs, dim=0)

    def _calculate_hidden_state(self, Z, H, H_tilde):
        outputs = []
        for i in range(self.batch_size):  # Iterate over the batch dimension
            H_i = Z[i] * H[i] + (1 - Z[i]) * H_tilde[i]
            outputs.append(H_i)
        
        return torch.stack(outputs, dim=0)



    def forward(self,X: torch.FloatTensor, edge_index: torch.LongTensor, edge_weight: torch.FloatTensor = None,
                H: torch.FloatTensor = None ) -> torch.FloatTensor:
        """
        Making a forward pass. If edge weights are not present the forward pass
        defaults to an unweighted graph. If the hidden state matrix is not present
        when the forward pass is called it is initialized with zeros.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node features.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Long Tensor, optional)* - Edge weight vector.
            * **H** *(PyTorch Float Tensor, optional)* - Hidden state matrix for all nodes.

        Return types:
            * **H** *(PyTorch Float Tensor)* - Hidden state matrix for all nodes.
        """
        H = self._set_hidden_state(X, H)
        Z = self._calculate_update_gate(X, edge_index, edge_weight, H)
        R = self._calculate_reset_gate(X, edge_index, edge_weight, H)
        H_tilde = self._calculate_candidate_state(X, edge_index, edge_weight, H, R)
        H = self._calculate_hidden_state(Z, H, H_tilde) # (b, 207, 32)
        return H
