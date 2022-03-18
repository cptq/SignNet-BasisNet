import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl

from scipy import sparse as sp
from scipy.sparse.linalg import norm 
from dgl.nn.pytorch import GINConv
"""
    Sparse Transformer 
    attention is acros neighbouring nodes only
    
"""
from layers.mlp_readout_layer import MLPReadout
from layers.mlp import MLP
from layers.transformer import BatchedTransformerLayer
from .sign_inv_net import get_sign_inv_net


class TransformerNet(nn.Module):
    def __init__(self, net_params):
        super().__init__()
        num_atom_type = net_params['num_atom_type']
        num_bond_type = net_params['num_bond_type']
        hidden_dim = net_params['hidden_dim']
        out_dim = net_params['out_dim']
        in_feat_dropout = net_params['in_feat_dropout']
        dropout = net_params['dropout']
        n_heads = net_params['n_heads']
        full_graph = net_params['full_graph']


        self.n_layers = net_params['L']
        self.readout = net_params['readout']
        self.batch_norm = net_params['batch_norm']
        self.layer_norm = net_params['layer_norm']
        print('BATCH NORM IN NET', self.batch_norm)
        self.residual = net_params['residual']
        print('RESIDUAL IN NET', self.residual)
        self.edge_feat = net_params['edge_feat']
        self.device = net_params['device']
        self.pe_init = net_params['pe_init']
        self.lap_method = net_params['lap_method']
        self.lap_lspe = net_params['lap_lspe']
        print('LAP LSPE', self.lap_lspe)
        
        self.use_lapeig_loss = net_params['use_lapeig_loss']
        self.lambda_loss = net_params['lambda_loss']
        self.alpha_loss = net_params['alpha_loss']
        
        self.pos_enc_dim = net_params['pos_enc_dim']

        self.pe_aggregate = net_params['pe_aggregate']
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            self.embedding_p = nn.Linear(self.pos_enc_dim, hidden_dim)

        self.embedding_h = nn.Embedding(num_atom_type, hidden_dim)

        if self.edge_feat:
            self.embedding_e = nn.Embedding(num_bond_type, hidden_dim)
        else:
            self.embedding_e = nn.Linear(1, hidden_dim)
        
        self.in_feat_dropout = nn.Dropout(in_feat_dropout)
        
        self.layers = nn.ModuleList([ BatchedTransformerLayer(hidden_dim, hidden_dim, n_heads, full_graph, use_edge=self.edge_feat) for _ in range(self.n_layers-1) ]) 
        self.layers.append(BatchedTransformerLayer(hidden_dim, out_dim, n_heads, full_graph, use_edge=self.edge_feat))
        
        self.MLP_layer = MLPReadout(out_dim, 1)   # 1 out dim since regression problem        

        if self.pe_init == 'rand_walk' or self.lap_lspe:
            self.p_out = nn.Linear(out_dim, self.pos_enc_dim)
            self.Whp = nn.Linear(out_dim+self.pos_enc_dim, out_dim)
        
        self.g = None              # For util; To be accessed in loss() function

        if self.lap_method == 'sign_inv':
            sign_inv_net = get_sign_inv_net(net_params)
            self.sign_inv_net = sign_inv_net
            
        if  self.pe_aggregate=='concat':
            self.pe_proj = nn.Linear(2*hidden_dim, hidden_dim)
        
    def forward(self, g, h, p, e, snorm_n):

        # input embedding
        h = self.embedding_h(h)
        h = self.in_feat_dropout(h)
        
        if self.pe_init in ['rand_walk', 'lap_pe']:
            p = self.embedding_p(p) 
            
        if self.pe_init == 'lap_pe' and not self.lap_lspe:

            if self.pe_aggregate=='concat':
                h = torch.cat([h, p], dim=1)
                h = self.pe_proj(h)
                p = None
            else:
                h = h + p
                p = None
        
        if not self.edge_feat: # edge feature set to 1
            e = torch.ones(e.size(0),1).to(self.device)
        e = self.embedding_e(e)   
        
        
        # convnets
        for conv in self.layers:
            h, p = conv(g, h, p, e)
            
        g.ndata['h'] = h
        
        if self.pe_init == 'rand_walk' or self.lap_lspe:
            # Implementing p_g = p_g - torch.mean(p_g, dim=0)
            p = self.p_out(p)
            g.ndata['p'] = p
            means = dgl.mean_nodes(g, 'p')
            batch_wise_p_means = means.repeat_interleave(g.batch_num_nodes(), 0)
            p = p - batch_wise_p_means

            # Implementing p_g = p_g / torch.norm(p_g, p=2, dim=0)
            g.ndata['p'] = p
            g.ndata['p2'] = g.ndata['p']**2
            norms = dgl.sum_nodes(g, 'p2')
            norms = torch.sqrt(norms)            
            batch_wise_p_l2_norms = norms.repeat_interleave(g.batch_num_nodes(), 0)
            p = p / batch_wise_p_l2_norms
            g.ndata['p'] = p
        
            # Concat h and p
            hp = self.Whp(torch.cat((g.ndata['h'],g.ndata['p']),dim=-1))
            g.ndata['h'] = hp
        
        # readout
        if self.readout == "sum":
            hg = dgl.sum_nodes(g, 'h')
        elif self.readout == "max":
            hg = dgl.max_nodes(g, 'h')
        elif self.readout == "mean":
            hg = dgl.mean_nodes(g, 'h')
        else:
            hg = dgl.mean_nodes(g, 'h')  # default readout is mean nodes
            
        self.g = g # For util; To be accessed in loss() function
        
        return self.MLP_layer(hg), g
        
    def loss(self, scores, targets):
        
        # Loss A: Task loss -------------------------------------------------------------
        loss_a = nn.L1Loss()(scores, targets)
        
        if self.use_lapeig_loss:
            # Loss B: Laplacian Eigenvector Loss --------------------------------------------
            g = self.g
            n = g.number_of_nodes()

            # Laplacian 
            A = g.adjacency_matrix(scipy_fmt="csr")
            N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
            L = sp.eye(n) - N * A * N

            p = g.ndata['p']
            pT = torch.transpose(p, 1, 0)
            loss_b_1 = torch.trace(torch.mm(torch.mm(pT, torch.Tensor(L.todense()).to(self.device)), p))

            # Correct batch-graph wise loss_b_2 implementation; using a block diagonal matrix
            bg = dgl.unbatch(g)
            batch_size = len(bg)
            P = sp.block_diag([bg[i].ndata['p'].detach().cpu() for i in range(batch_size)])
            PTP_In = P.T * P - sp.eye(P.shape[1])
            loss_b_2 = torch.tensor(norm(PTP_In, 'fro')**2).float().to(self.device)

            loss_b = ( loss_b_1 + self.lambda_loss * loss_b_2 ) / ( self.pos_enc_dim * batch_size * n) 

            del bg, P, PTP_In, loss_b_1, loss_b_2

            loss = loss_a + self.alpha_loss * loss_b
        else:
            loss = loss_a
        
        return loss

    
    
    
    
