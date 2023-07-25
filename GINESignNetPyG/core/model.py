import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
import core.model_utils.pyg_gnn_wrapper as gnn_wrapper 
from core.model_utils.elements import MLP, DiscreteEncoder, Identity, BN
from torch_geometric.nn.inits import reset

class GNN(nn.Module):
    # this version use nin as hidden instead of nout, resulting a larger model
    def __init__(self, nfeat_node, nfeat_edge, nhid, nout, nlayer, gnn_type, dropout=0, pooling='add', bn=BN, dos_bins=0, res=True):
        super().__init__()
        self.input_encoder = DiscreteEncoder(nhid-dos_bins) if nfeat_node is None else MLP(nfeat_node, nhid-dos_bins, 1)
        self.edge_encoders = nn.ModuleList([DiscreteEncoder(nhid) if nfeat_edge is None else MLP(nfeat_edge, nhid, 1) for _ in range(nlayer)])
        self.convs = nn.ModuleList([getattr(gnn_wrapper, gnn_type)(nhid, nhid, bias=not bn) for _ in range(nlayer)]) # set bias=False for BN
        self.norms = nn.ModuleList([nn.BatchNorm1d(nhid) if bn else Identity() for _ in range(nlayer)])
        self.output_encoder = MLP(nhid, nout, nlayer=2, with_final_activation=False, with_norm=False if pooling=='mean' else True)
        self.size_embedder = nn.Embedding(200, nhid) 
        self.linear = nn.Linear(2*nhid, nhid)

        if dos_bins > 0:
            self.ldos_encoder = MLP(dos_bins, nhid, nlayer=2, with_final_activation=True, with_norm=True)
            self.dos_encoder = MLP(dos_bins, nhid, nlayer=2, with_final_activation=False, with_norm=True)

        self.pooling = pooling
        self.dropout = dropout
        self.res = res
        # for additional feature from (L)DOS
        self.dos_bins = dos_bins

    def reset_parameters(self):
        self.input_encoder.reset_parameters()
        self.output_encoder.reset_parameters()
        self.size_embedder.reset_parameters()
        self.linear.reset_parameters()
        if self.dos_bins > 0:
            self.dos_encoder.reset_parameters()
            self.ldos_encoder.reset_parameters()
        for edge_encoder, conv, norm in zip(self.edge_encoders, self.convs, self.norms):
            edge_encoder.reset_parameters()
            conv.reset_parameters()
            norm.reset_parameters()
     
    def forward(self, data, additional_x=None):
        x = self.input_encoder(data.x.squeeze())

        # for PDOS 
        if self.dos_bins > 0:
            x = torch.cat([x, data.pdos], dim=-1)
            # x += self.ldos_encoder(data.pdos)

        if additional_x is not None:
            x = self.linear(torch.cat([x, additional_x], dim=-1))

        ori_edge_attr = data.edge_attr 
        if ori_edge_attr is None:
            ori_edge_attr = data.edge_index.new_zeros(data.edge_index.size(-1))

        previous_x = x
        for edge_encoder, layer, norm in zip(self.edge_encoders, self.convs, self.norms):
            edge_attr = edge_encoder(ori_edge_attr) 
            x = layer(x, data.edge_index, edge_attr)
            x = norm(x)
            x = F.relu(x)
            x = F.dropout(x, self.dropout, training=self.training)
            if self.res:
                x += previous_x 
                previous_x = x

        if self.pooling == 'mean':
            graph_size = scatter(torch.ones_like(x[:,0], dtype=torch.int64), data.batch, dim=0, reduce='add')
            x = scatter(x, data.batch, dim=0, reduce='mean') + self.size_embedder(graph_size)
        else:
            x = scatter(x, data.batch, dim=0, reduce='add')

        if self.dos_bins > 0:
            x = x + self.dos_encoder(data.dos)
        x = self.output_encoder(x)
        return x

