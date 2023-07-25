import torch
import torch.nn as nn
import torch_geometric.nn as gnn
from core.model_utils.elements import MLP
import torch.nn.functional as F

class GINConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias, with_norm=False) ##### Do not use BN!!!!
        self.layer = gnn.GINConv(self.nn, train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)


class GINEConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        self.nn = MLP(nin, nout, 2, False, bias=bias)
        self.layer = gnn.GINEConv(self.nn, train_eps=True)
    def reset_parameters(self):
        self.nn.reset_parameters()
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index, edge_attr)

class GATConv(nn.Module):
    def __init__(self, nin, nout, bias=True, nhead=1):
        super().__init__()
        self.layer = gnn.GATConv(nin, nout//nhead, nhead, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)

class GCNConv(nn.Module):
    def __init__(self, nin, nout, bias=True):
        super().__init__()
        # self.nn = MLP(nin, nout, 2, False, bias=bias)
        # self.layer = gnn.GCNConv(nin, nin, bias=True)
        self.layer = gnn.GCNConv(nin, nout, bias=bias)
    def reset_parameters(self):
        self.layer.reset_parameters()
    def forward(self, x, edge_index, edge_attr):
        return self.layer(x, edge_index)
        # return self.nn(F.relu(self.layer(x, edge_index)))

from torch_scatter import scatter
from torch_geometric.utils import degree
class SimplifiedPNAConv(gnn.MessagePassing):
    def __init__(self, nin, nout, bias=True, aggregators=['mean'], **kwargs): # ['mean', 'min', 'max', 'std'],
        kwargs.setdefault('aggr', None)
        super().__init__(node_dim=0, **kwargs)
        self.aggregators = aggregators
        self.pre_nn = MLP(3*nin, nin, 2, False)
        self.post_nn = MLP((len(aggregators) + 1 +1) * nin, nout, 2, False, bias=bias)
        # self.post_nn = MLP((len(aggregators) + 1 ) * nin, nout, 2, False)
        self.deg_embedder = nn.Embedding(200, nin) 

    def reset_parameters(self):
        self.pre_nn.reset_parameters()
        self.post_nn.reset_parameters()
        self.deg_embedder.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        out = torch.cat([x, out], dim=-1)
        out = self.post_nn(out)
        # return x + out
        return out

    def message(self, x_i, x_j, edge_attr):
        if edge_attr is not None:
            h = torch.cat([x_i, x_j, edge_attr], dim=-1)
        else:
            h = torch.cat([x_i, x_j], dim=-1)
        return self.pre_nn(h)

    def aggregate(self, inputs, index, dim_size=None):
        outs = []
        for aggregator in self.aggregators:
            if aggregator == 'sum':
                out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
            elif aggregator == 'mean':
                out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
            elif aggregator == 'min':
                out = scatter(inputs, index, 0, None, dim_size, reduce='min')
            elif aggregator == 'max':
                out = scatter(inputs, index, 0, None, dim_size, reduce='max')
            elif aggregator == 'var' or aggregator == 'std':
                mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                mean_squares = scatter(inputs * inputs, index, 0, None, dim_size, reduce='mean')
                out = mean_squares - mean * mean
                if aggregator == 'std':
                    out = torch.sqrt(F.relu_(out) + 1e-5)
            else:
                raise ValueError(f'Unknown aggregator "{aggregator}".')  
            outs.append(out)

        outs.append(self.deg_embedder(degree(index, dim_size, dtype=index.dtype)))
        out = torch.cat(outs, dim=-1)

        return out
