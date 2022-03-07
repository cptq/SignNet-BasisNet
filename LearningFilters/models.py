import math
import numpy as np
from scipy.special import comb

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import ARMAConv, GATConv, ChebConv, GCNConv
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_self_loops, get_laplacian
from torch_geometric.nn.conv.gcn_conv import gcn_norm


activation_lst = {'relu': nn.ReLU()}

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, num_layers=3, use_bn=False, use_ln=False, dropout=0.0, activation='relu'):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels, track_running_stats=False))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels, track_running_stats=False))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.activation = activation_lst[activation]
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
            
    def forward(self, x, *args):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.activation(x)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid dimension of x')
            if self.use_ln: x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class EqDeepSetsEncoder(nn.Module):
    # equivariant deepsets encoder
    # *** x set size x feature size

    def __init__(self, in_channels, hidden_channels=32, out_channels=1, num_layers=3, use_bn=False, use_ln=False, dropout=0.0, activation='relu'):
        super(EqDeepSetsEncoder, self).__init__()
        self.lins1 = nn.ModuleList()
        self.lins2 = nn.ModuleList()
        if use_bn: self.bns = nn.ModuleList()
        if use_ln: self.lns = nn.ModuleList()
        
        if num_layers == 1:
            # linear mapping
            self.lins1.append(nn.Linear(in_channels, out_channels))
            self.lins2.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins1.append(nn.Linear(in_channels, hidden_channels))
            self.lins2.append(nn.Linear(in_channels, hidden_channels))
            if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels, track_running_stats=False))
            if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            for layer in range(num_layers-2):
                self.lins1.append(nn.Linear(hidden_channels, hidden_channels))
                self.lins2.append(nn.Linear(hidden_channels, hidden_channels))
                if use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels, track_running_stats=False))
                if use_ln: self.lns.append(nn.LayerNorm(hidden_channels))
            self.lins1.append(nn.Linear(hidden_channels, out_channels))
            self.lins2.append(nn.Linear(hidden_channels, out_channels))
        
        self.activation = activation_lst[activation]
        self.use_bn = use_bn
        self.use_ln = use_ln
        self.dropout = dropout
            
    def forward(self, x, *args):
        for i in range(len(self.lins1)-1):
            lin1 = self.lins1[i]
            lin2 = self.lins2[i]
            x1 = lin1(x)

            x2 = x.mean(dim=-2, keepdims=True)
            x2 = lin2(x2)
            x = self.activation(x1 + x2)
            if self.use_bn:
                if x.ndim == 2:
                    x = self.bns[i](x)
                elif x.ndim == 3:
                    x = self.bns[i](x.transpose(2,1)).transpose(2,1)
                else:
                    raise ValueError('invalid x dimension')
            if self.use_ln: x = self.lns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x1 = self.lins1[-1](x)
        x2 = x.mean(dim=-2, keepdims=True)
        x2 = self.lins2[-1](x2)
        x = x1 + x2
        return x

class Transformer(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, out_channels=1, num_layers=2, num_heads=4, dropout=0):
        super(Transformer, self).__init__()
        self.fc1 = Linear(in_channels, hidden_channels)
        self.encs=nn.ModuleList()
        for _ in range(num_layers):
            self.encs.append(nn.TransformerEncoderLayer(d_model=hidden_channels, dim_feedforward=hidden_channels, nhead=num_heads, dropout=dropout, norm_first=True, batch_first=True))
        self.fc2 = Linear(hidden_channels, out_channels)

    def forward(self, x, *args):
        #x = self.enc(x.unsqueeze(0)).squeeze(0)
        x = self.fc1(x)
        orig_dim = x.ndim
        if orig_dim == 2:
            x = x.unsqueeze(0)
        for enc in self.encs:
            x = enc(x)
        if orig_dim == 2:
            x = x.squeeze(0)
        x = self.fc2(x)
        return x


class GPR_prop(MessagePassing):
    def __init__(self, K, alpha=0.1, Init='Random', Gamma=None, bias=True, **kwargs):
        super(GPR_prop, self).__init__(aggr='add', **kwargs)
        self.K = K
        self.Init = Init
        self.alpha = alpha

        assert Init in ['SGC', 'PPR', 'NPPR', 'Random', 'WS']
        if Init == 'SGC':
            # SGC-like
            TEMP = 0.0*np.ones(K+1)
            TEMP[-1] = 1.0
        elif Init == 'PPR':
            # PPR-like
            TEMP = alpha*(1-alpha)**np.arange(K+1)
            TEMP[-1] = (1-alpha)**K
        elif Init == 'NPPR':
            # Negative PPR
            TEMP = (alpha)**np.arange(K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'Random':
            # Random
            bound = np.sqrt(3/(K+1))
            TEMP = np.random.uniform(-bound, bound, K+1)
            TEMP = TEMP/np.sum(np.abs(TEMP))
        elif Init == 'WS':
            # Specify Gamma
            TEMP = Gamma

        self.temp = nn.Parameter(torch.tensor(TEMP))

    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K+1):
            self.temp.data[k] = self.alpha*(1-self.alpha)**k
        self.temp.data[-1] = (1-self.alpha)**self.K

    def forward(self, x, edge_index, edge_weight=None):
        edge_index, norm = gcn_norm(
            edge_index, edge_weight, num_nodes=x.size(0), dtype=x.dtype)

        hidden = x*(self.temp[0])
        for k in range(self.K):
            x = self.propagate(edge_index, x=x, norm=norm)
            gamma = self.temp[k+1]
            hidden = hidden + gamma*x
        return hidden

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}(K={}, temp={})'.format(self.__class__.__name__, self.K,
                                          self.temp)
class GPRNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=2, K=10):
        super(GPRNet, self).__init__()
        self.lins = nn.ModuleList()
        self.lins.append(Linear(in_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.prop1 = GPR_prop(K)
        self.fc2 = torch.nn.Linear(hidden_channels, 1)
    
    def reset_parameters(self):
        self.prop1.reset_parameters()

    def forward(self, x, edge_index):

        for lin in self.lins:
            x = F.relu(lin(x))

        x = self.prop1(x, edge_index)
        return self.fc2(x)


class ARMANet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=2):
        super(ARMANet, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(ARMAConv(in_channels,hidden_channels,1,1,False,dropout=0))
        for _ in range(num_layers-1):
            self.convs.append(ARMAConv(hidden_channels,hidden_channels,1,1,False,dropout=0))
        self.fc2 = torch.nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  

        return self.fc2(x)


class GcnNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=2):
        super(GcnNet, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels, cached=False))
        for _ in range(num_layers-1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=False) )
        
        self.fc2 = torch.nn.Linear(hidden_channels, 1) 

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))  

        return self.fc2(x)

class GatNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_heads=4, num_layers=2):
        super(GatNet, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels//num_heads, heads=num_heads,concat=True, dropout=0.0))
        for _ in range(num_layers-1):
            self.convs.append(GATConv(hidden_channels, hidden_channels//num_heads, heads=num_heads,concat=True, dropout=0.0))
        
        self.fc2 = torch.nn.Linear(hidden_channels, 1) 

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = F.elu(conv(x, edge_index))

        return self.fc2(x) 

class ChebNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=2, K=3):
        super(ChebNet, self).__init__()
        
        self.convs = nn.ModuleList()
        self.convs.append(ChebConv(in_channels, hidden_channels, K))
        for _ in range(num_layers-1):
            self.convs.append(ChebConv(hidden_channels, hidden_channels, K))
        self.fc2 = torch.nn.Linear(hidden_channels, 1) 

    def forward(self, x, edge_index):

        for conv in self.convs:
            x = F.relu(conv(x, edge_index))   
           
        return self.fc2(x) 

class BernConv(MessagePassing):

    def __init__(self, in_channels, out_channels, K,bias=True, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super(BernConv, self).__init__(**kwargs)
        assert K > 0

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()
        self.K=K

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(-2) + self.weight.size(-1)))
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.fill_(0)

    def forward(self,x,edge_index,coe,edge_weight=None):

        TEMP=F.relu(coe)

        #L=I-D^(-0.5)AD^(-0.5)
        edge_index1, norm1 = get_laplacian(edge_index, edge_weight,normalization='sym', dtype=x.dtype, num_nodes=x.size(self.node_dim))

        #2I-L
        edge_index2, norm2 = add_self_loops(edge_index1,-norm1,fill_value=2.,num_nodes=x.size(self.node_dim))

        tmp=[]
        tmp.append(x)
        for i in range(self.K):
                x=self.propagate(edge_index2,x=x,norm=norm2,size=None)
                tmp.append(x)

        out=(comb(self.K,0)/(2**self.K))*TEMP[0]*tmp[self.K]

        for i in range(self.K):
                x=tmp[self.K-i-1]
                x=self.propagate(edge_index1,x=x,norm=norm1,size=None)
                for j in range(i):
                        x=self.propagate(edge_index1,x=x,norm=norm1,size=None)

                out=out+(comb(self.K,i+1)/(2**self.K))*TEMP[i+1]*x

        out=out@self.weight
        if self.bias is not None:
                out+=self.bias
        return out

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

    def __repr__(self):
        return '{}({}, {}, K={})'.format(
            self.__class__.__name__, self.in_channels, self.out_channels,
            self.weight.size(0))


class BernNet(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, num_layers=2, K=10):
        super(BernNet, self).__init__()

        self.convs = nn.ModuleList()
        self.convs.append(BernConv(in_channels, hidden_channels, K))
        for _ in range(num_layers-1):
            self.convs.append(BernConv(hidden_channels, hidden_channels, K))

        self.fc2 = torch.nn.Linear(hidden_channels, 1)
        self.coe = nn.Parameter(torch.Tensor(K+1))
        self.reset_parameters()

    def reset_parameters(self):
        self.coe.data.fill_(1)

    def forward(self, x, edge_index):
        
        for conv in self.convs:
            x = F.relu(conv(x, edge_index,self.coe))  

        return self.fc2(x) 

