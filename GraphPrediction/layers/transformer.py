# built off of GraphTransformer and SAN code
import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
import numpy as np

"""
batched transformers
"""

def src_dot_dst(src_field, dst_field, out_field):
    def func(edges):
        return {out_field: (edges.src[src_field] * edges.dst[dst_field])}
    return func


def scaling(field, scale_constant):
    def func(edges):
        return {field: ((edges.data[field]) / scale_constant)}
    return func

# Improving implicit attention scores with explicit edge features, if available
def imp_exp_attn(implicit_attn, explicit_edge):
    """
        implicit_attn: the output of K Q
        explicit_edge: the explicit edge features
    """
    def func(edges):
        return {implicit_attn: (edges.data[implicit_attn] * edges.data[explicit_edge])}
    return func

def exp_real(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func


def exp_fake(field, L):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': L*torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))/(L+1)}
    return func

def exp(field):
    def func(edges):
        # clamp for softmax numerical stability
        return {'score_soft': torch.exp((edges.data[field].sum(-1, keepdim=True)).clamp(-5, 5))}
    return func


"""
    Single Attention Head
"""

class BatchedAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, use_bias=True, use_edge=False):
        super().__init__()
       
        self.out_dim = out_dim
        self.num_heads = num_heads

        self.use_edge = use_edge

        self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.K = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.E = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        self.V = nn.Linear(in_dim, out_dim * num_heads, bias=use_bias)
        
    def propagate_attention(self, g):

        
        real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        #g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        
        g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
        
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)
        
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        return h_out
    

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, gamma, in_dim, out_dim, num_heads, full_graph, use_bias, attention_for):
        super().__init__()
        
       
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.full_graph=full_graph
        self.attention_for = attention_for
        self.gamma = gamma
        
        if self.attention_for == "h":
            if use_bias:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=True)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=True)

            else:
                self.Q = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.K = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                self.E = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                if self.full_graph:
                    self.Q_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.K_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)
                    self.E_2 = nn.Linear(in_dim, out_dim * num_heads, bias=False)

                self.V = nn.Linear(in_dim, out_dim * num_heads, bias=False)
        
        
    def propagate_attention(self, g):

        
        if self.full_graph:
            real_ids = torch.nonzero(g.edata['real']).squeeze()
            fake_ids = torch.nonzero(g.edata['real']==0).squeeze()

        else:
            real_ids = g.edges(form='eid')
            
        g.apply_edges(src_dot_dst('K_h', 'Q_h', 'score'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(src_dot_dst('K_2h', 'Q_2h', 'score'), edges=fake_ids)
        

        # scale scores by sqrt(d)
        g.apply_edges(scaling('score', np.sqrt(self.out_dim)))
        
        # Use available edge features to modify the scores for edges
        g.apply_edges(imp_exp_attn('score', 'E'), edges=real_ids)
        
        if self.full_graph:
            g.apply_edges(imp_exp_attn('score', 'E_2'), edges=fake_ids)
            
        
        if self.full_graph:
            # softmax and scaling by gamma
            L = torch.clamp(self.gamma, min=0.0, max=1.0)   # Gamma \in [0,1]
            g.apply_edges(exp_real('score', L), edges=real_ids)
            g.apply_edges(exp_fake('score', L), edges=fake_ids)
        
        else:
            g.apply_edges(exp('score'), edges=real_ids)

        # Send weighted values to target nodes
        eids = g.edges()
        g.send_and_recv(eids, fn.src_mul_edge('V_h', 'score_soft', 'V_h'), fn.sum('V_h', 'wV'))
        g.send_and_recv(eids, fn.copy_edge('score_soft', 'score_soft'), fn.sum('score_soft', 'z'))
        
    
    def forward(self, g, h, e):
        
        Q_h = self.Q(h)
        K_h = self.K(h)
        E = self.E(e)
        
        if self.full_graph:
            Q_2h = self.Q_2(h)
            K_2h = self.K_2(h)
            E_2 = self.E_2(e)
            
        V_h = self.V(h)

        
        # Reshaping into [num_nodes, num_heads, feat_dim] to 
        # get projections for multi-head attention
        g.ndata['Q_h'] = Q_h.view(-1, self.num_heads, self.out_dim)
        g.ndata['K_h'] = K_h.view(-1, self.num_heads, self.out_dim)
        g.edata['E'] = E.view(-1, self.num_heads, self.out_dim)
        
        
        if self.full_graph:
            g.ndata['Q_2h'] = Q_2h.view(-1, self.num_heads, self.out_dim)
            g.ndata['K_2h'] = K_2h.view(-1, self.num_heads, self.out_dim)
            g.edata['E_2'] = E_2.view(-1, self.num_heads, self.out_dim)
        
        g.ndata['V_h'] = V_h.view(-1, self.num_heads, self.out_dim)

        self.propagate_attention(g)
        
        h_out = g.ndata['wV'] / (g.ndata['z'] + torch.full_like(g.ndata['z'], 1e-6))
        
        return h_out
    

class BatchedTransformerLayer(nn.Module):
    """
        Param: 
    """
    def __init__(self, in_dim, out_dim, num_heads, full_graph, dropout=0.0,
                 layer_norm=False, batch_norm=True, residual=True, use_bias=False, use_edge=False):
        super().__init__()
        
        self.in_channels = in_dim
        self.out_channels = out_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.residual = residual
        self.layer_norm = layer_norm     
        self.batch_norm = batch_norm
        
        if use_edge is False:
            self.attention_h = BatchedAttentionLayer(in_dim, out_dim//num_heads, num_heads, use_bias=use_bias, use_edge=use_edge)
        else:
            self.gamma = nn.Parameter(torch.FloatTensor([0.1]))
            self.attention_h = MultiHeadAttentionLayer(self.gamma, in_dim, out_dim//num_heads, num_heads,
                                                   full_graph, use_bias, attention_for="h")
        self.O_h = nn.Linear(out_dim, out_dim)
        
        if self.layer_norm:
            self.layer_norm1_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm1_h = nn.BatchNorm1d(out_dim)
        
        # FFN for h
        self.FFN_h_layer1 = nn.Linear(out_dim, out_dim*2)
        self.FFN_h_layer2 = nn.Linear(out_dim*2, out_dim)

        if self.layer_norm:
            self.layer_norm2_h = nn.LayerNorm(out_dim)
            
        if self.batch_norm:
            self.batch_norm2_h = nn.BatchNorm1d(out_dim)
            
        
    def forward(self, g, h, p, e):
        h_in1 = h # for first residual connection
        
        # multi-head attention out
        h_attn_out = self.attention_h(g, h, e)
        
        #Concat multi-head outputs
        h = h_attn_out.view(-1, self.out_channels)
       
        h = F.dropout(h, self.dropout, training=self.training)

        h = self.O_h(h)

        if self.residual:
            h = h_in1 + h # residual connection

        if self.layer_norm:
            h = self.layer_norm1_h(h)

        if self.batch_norm:
            h = self.batch_norm1_h(h)

        h_in2 = h # for second residual connection

        # FFN for h
        h = self.FFN_h_layer1(h)
        h = F.relu(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.FFN_h_layer2(h)

        if self.residual:
            h = h_in2 + h # residual connection       

        if self.layer_norm:
            h = self.layer_norm2_h(h)

        if self.batch_norm:
            h = self.batch_norm2_h(h)         
        
        
        return h, None
        
    def __repr__(self):
        return '{}(in_channels={}, out_channels={}, heads={}, residual={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.num_heads, self.residual)
