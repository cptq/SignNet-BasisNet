"""GCN using DGL nn package
References:
- Semi-Supervised Classification with Graph Convolutional Networks
- Paper: https://arxiv.org/abs/1609.02907
- Code: https://github.com/tkipf/gcn
"""
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv, GATConv, GINConv

from layers.mlp import MLP

activations = {'relu': nn.ReLU(), 'tanh': nn.Tanh()}

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5, activation='relu'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.activation = activations[activation]
        # input layer
        self.layers.append(GraphConv(in_channels, hidden_channels, activation=self.activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GraphConv(hidden_channels, hidden_channels, activation=self.activation))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        self.layers.append(GraphConv(hidden_channels, out_channels))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i-1](x)
                    elif x.ndim == 3:
                        x = self.bns[i-1](x.transpose(2,1)).transpose(2,1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(g, x)
        return x


class GAT(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, num_heads, use_bn=True, dropout=0.5, activation='relu'):
        super(GAT, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.activation = activations[activation]
        # input layer
        self.layers.append(GATConv(in_channels, hidden_channels, num_heads, activation=self.activation))
        # hidden layers
        for i in range(n_layers - 2):
            self.layers.append(GATConv(hidden_channels, hidden_channels, num_heads, activation=self.activation))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        self.layers.append(GATConv(hidden_channels, out_channels, num_heads))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i-1](x)
                    elif x.ndim == 3:
                        x = self.bns[i-1](x.transpose(2,1)).transpose(2,1)
                    else:
                        raise ValueError('invalid x dim')
            print('X GAT SHAPE', x.shape)
            x = layer(g, x)
        return x

class GIN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, n_layers, use_bn=True, dropout=0.5, activation='relu'):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        self.use_bn = use_bn
        self.activation = activations[activation]
        # input layer
        update_net = MLP(in_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net, 'sum'))
        # hidden layers
        for i in range(n_layers - 2):
            update_net = MLP(hidden_channels, hidden_channels, hidden_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
            self.layers.append(GINConv(update_net, 'sum'))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        # output layer
        update_net = MLP(hidden_channels, hidden_channels, out_channels, 2, use_bn=use_bn, dropout=dropout, activation=activation)
        self.layers.append(GINConv(update_net, 'sum'))
        if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, g, x):
        for i, layer in enumerate(self.layers):
            if i != 0:
                x = self.dropout(x)
                if self.use_bn:
                    if x.ndim == 2:
                        x = self.bns[i-1](x)
                    elif x.ndim == 3:
                        x = self.bns[i-1](x.transpose(2,1)).transpose(2,1)
                    else:
                        raise ValueError('invalid x dim')
            x = layer(g, x)
        return x
