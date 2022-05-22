import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import torch.nn.functional as F
from sign_net.model_utils.elements import Identity

class MaskedBN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)
    def reset_parameters(self):
        self.bn.reset_parameters() 
    def forward(self, x, mask=None):
        ### apply BN to the last dim
        #    x: n x k x d
        # mask: n x k  
        if mask is None:
            return self.bn(x.transpose(1,2)).transpose(1,2)
        x[mask] = self.bn(x[mask])
        return x

class MaskedLN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=1e-6)
    def reset_parameters(self):
        self.ln.reset_parameters() 
    def forward(self, x, mask=None):
        if mask is None:
            return self.ln(x)
        x[mask] = self.ln(x[mask])
        return x

class MaskedMLP(nn.Module):
    def __init__(self, nin, nout, nlayer=2, with_final_activation=True, with_norm=True, bias=True, nhid=None):
        super().__init__()
        n_hid = nin if nhid is None else nhid
        self.layers = nn.ModuleList([nn.Linear(nin if i==0 else n_hid, 
                                               n_hid if i<nlayer-1 else nout, 
                                               bias=True if (i==nlayer-1 and not with_final_activation and bias) # TODO: revise later
                                                or (not with_norm) else False) # set bias=False for BN
                                     for i in range(nlayer)])
        self.norms = nn.ModuleList([MaskedBN(n_hid if i<nlayer-1 else nout) if with_norm else Identity()
                                     for i in range(nlayer)])
        self.nlayer = nlayer
        self.with_final_activation = with_final_activation
        self.residual = (nin==nout) ## TODO: test whether need this

    def reset_parameters(self):
        for layer, norm in zip(self.layers, self.norms):
            layer.reset_parameters()
            norm.reset_parameters()

    def forward(self, x, mask=None):
        # x: n x k x d
        previous_x = x
        for i, (layer, norm) in enumerate(zip(self.layers, self.norms)):
            x = layer(x)
            if mask is not None:
                x[~mask] = 0
            if i < self.nlayer-1 or self.with_final_activation:
                x = norm(x, mask)
                x = F.relu(x)  
        return x    

class MaskedGINConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhid=None):
        super().__init__()
        self.nn = MaskedMLP(nin, nout, 2, False, bias=bias, nhid=nhid)
        self.layer = gnn.GINConv(Identity(), train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr, mask=None):
        x = self.layer(x, edge_index)
        if mask is not None:
            if x[~mask].numel() == 0:
                print('~mask numel = 0!!')
                print('x shape', x.shape)
                print('mask shape', mask.shape)
            #assert x[~mask].max() == 0 
        x = self.nn(x, mask)
        # assert x[~mask].max() == 0 
        return x


class MaskedGINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhid=None):
        super().__init__()
        self.nn = MaskedMLP(nin, nout, 2, False, bias=bias, nhid=nhid)
        self.layer = gnn.GINEConv(Identity(), train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr, mask=None):
        assert x[~mask].max() == 0
        x = self.layer(x, edge_index, edge_attr)
        if mask is not None:
            x[~mask] = 0 
        x = self.nn(x, mask)
        # assert x[~mask].max() == 0 
        return x
