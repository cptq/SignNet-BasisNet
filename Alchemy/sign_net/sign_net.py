import torch, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from sign_net.transform import to_dense_list_EVD

import sign_net.model_utils.masked_layers as masked_layers 
from sign_net.model import GNN
from sign_net.model_utils.transformer_module import TransformerEncoderLayer, PositionalEncoding 
from sign_net.model_utils.elements import DiscreteEncoder

class GNN3d(nn.Module): 
    """
    Apply GNN on a 3-dimensional data x: n x k x d. 
    Equivalent to apply GNN on k independent nxd 2-d feature.
    * Assume no edge feature for now.
    """
    def __init__(self, n_in, n_out, n_layer, gnn_type='MaskedGINConv'):
        super().__init__()
        self.convs = nn.ModuleList([getattr(masked_layers, gnn_type)(n_in if i==0 else n_out, n_out, bias=True, nhid=n_out) for i in range(n_layer)])
        self.norms = nn.ModuleList([masked_layers.MaskedBN(n_out) for _ in range(n_layer)])

    def reset_parameters(self):
        for conv, norm in zip(self.convs, self.norms): 
            conv.reset_parameters()
            norm.reset_parameters() 
    
    def forward(self, x, edge_index, edge_attr, mask=None):
        # x: n x k x d
        # mask: n x k
        x = x.transpose(0, 1) # k x n x d
        if mask is not None:
            mask = mask.transpose(0, 1) # k x n
        previous_x = 0
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x, edge_index, edge_attr, mask) # pass mask into
            # assert x[~mask].max() == 0 
            if mask is not None:
                x[~mask] = 0
            x = norm(x, mask)
            x = F.relu(x)
            x = x + previous_x
            previous_x = x
        return x.transpose(0, 1)

class SetTransformer(nn.Module):
    def __init__(self, nhid, nlayer):
        super().__init__()
        self.transformer_layers = nn.ModuleList(TransformerEncoderLayer(nhid, n_head=4) for _ in range(nlayer))
        self.out = nn.Sequential(nn.Linear(nhid, nhid, bias=False), nn.BatchNorm1d(nhid))

    def reset_parameters(self):
        for layer in self.transformer_layers:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for layer in out:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x, pos, mask):
        # x: n x k x d
        # pos: n x k 
        # mask: n x k
        # x[~mask] = 0
        x = x + pos
        #assert x[~mask].max() == 0
        for layer in self.transformer_layers:
            #assert x[~mask].max() == 0
            x,_ = layer(x, mask)
        x = torch.sum(x, dim=1) # n x d #### TODO: change later
        x = self.out(x)
        return x

class SignNet(nn.Module):
    """
        n x k node embeddings => n x n_hid 

        The output is sign invariant and permutation equivariant 
    """
    def __init__(self, n_hid, nl_phi, nl_rho=2, ignore_eigval=False):
        super().__init__()
        self.phi = GNN3d(1, n_hid, nl_phi, gnn_type='MaskedGINConv') 
        self.rho = SetTransformer(n_hid, nl_rho)

        self.ignore_eigval = ignore_eigval
        if not self.ignore_eigval:
            self.eigen_encoder = masked_layers.MaskedMLP(1, n_hid, nlayer=2)


    def reset_parameters(self):
        self.phi.reset_parameters()
        self.rho.reset_parameters()
        if not self.ignore_eigval:
            self.eigen_encoder.reset_parameters()

    def forward(self, data):
        eigS_dense, eigV_dense = to_dense_list_EVD(data.eigen_values, data.eigen_vectors, data.batch) 
        x = eigV_dense
        # get mask and prepare x
        size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add') # b x 1
        mask = torch.arange(x.size(1), device=x.device)[None, :] < size[:, None]                    # b x N_max
        mask_full = mask[data.batch]

        # transform eigens 
        x = x.unsqueeze(-1)
        # assert x[~mask_full].max() == 0
        if not self.ignore_eigval:
            pos = self.eigen_encoder(eigS_dense.unsqueeze(-1), mask_full)
        else:
            pos = 0 # ignore eigenvalues 

        # phi
        x = self.phi(x, data.edge_index, data.edge_attr, mask_full) + self.phi(-x, data.edge_index, data.edge_attr, mask_full)

        # rho = Transformer
        x = self.rho(x, pos=pos, mask=mask_full)

        return x # n x n_hid 

class SignNetGNN(nn.Module):
    def __init__(self, node_feat, edge_feat, n_hid, n_out, nl_signnet, nl_gnn, nl_rho=4, ignore_eigval=False, gnn_type='GINEConv'):
        super().__init__()
        self.sign_net = SignNet(n_hid, nl_signnet, nl_rho=4, ignore_eigval=ignore_eigval)
        self.gnn = GNN(node_feat, edge_feat, n_hid, n_out, nlayer=nl_gnn, gnn_type=gnn_type) 

    def reset_parameters(self):
        self.sign_net.reset_parameters()
        self.gnn.reset_parameters()

    def forward(self, data):
        pos = self.sign_net(data)
        return self.gnn(data, pos)
