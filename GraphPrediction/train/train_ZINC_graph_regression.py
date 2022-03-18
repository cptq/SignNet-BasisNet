"""
    Utility functions for training one epoch 
    and evaluating one epoch
"""
import torch
import torch.nn as nn
import math

import dgl

from train.metrics import MAE

def handle_lap(model, batch_pos_enc, batch_graphs, device):
    if model.lap_method == 'sign_flip':
        sign_flip = torch.rand(batch_pos_enc.size(1)).to(device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        batch_pos_enc = batch_pos_enc * sign_flip.unsqueeze(0)
    elif model.lap_method == 'abs_val':
        batch_pos_enc = batch_pos_enc.abs()
    elif model.lap_method == 'sign_inv':
        # n x k  -> k x n x 1
        batch_pos_enc = batch_pos_enc.unsqueeze(-1)
        batch_pos_enc = model.sign_inv_net(batch_graphs, batch_pos_enc)
        # k x n x 1 -> n x k
        batch_pos_enc = batch_pos_enc.squeeze(-1)
    elif model.lap_method == 'canonical':
        batch_graphs.ndata['pos_pos_enc_count'] = (batch_pos_enc >= 0).float()
        batch_graphs.ndata['neg_pos_enc_count'] = (batch_pos_enc < 0).float()
        less_nonneg = dgl.sum_nodes(batch_graphs, 'pos_pos_enc_count') <  dgl.sum_nodes(batch_graphs, 'neg_pos_enc_count')

        nonneg_pos_enc = torch.clone(batch_pos_enc)
        neg_pos_enc = torch.clone(batch_pos_enc)
        nonneg_pos_enc[batch_pos_enc < 0] = 0
        neg_pos_enc[batch_pos_enc >= 0] = 0
        batch_graphs.ndata['nonneg_pos_enc'] = nonneg_pos_enc
        batch_graphs.ndata['neg_pos_enc_abs'] = neg_pos_enc.abs()

        less_norm = dgl.sum_nodes(batch_graphs, 'nonneg_pos_enc') < dgl.sum_nodes(batch_graphs, 'neg_pos_enc_abs')
        sign_flip = less_nonneg + less_norm
        sign_flip = -sign_flip.float()
        sign_flip[sign_flip == 0] = 1
        sign_flip = dgl.broadcast_nodes(batch_graphs, sign_flip)
        batch_pos_enc = sign_flip * batch_pos_enc
 
    elif model.lap_method == 'none':
        # raw laplacian eigenvectors
        pass
    else:
        raise ValueError('invalid laplacian method')

    return batch_pos_enc


def train_epoch_sparse(model, optimizer, device, data_loader, epoch):
    model.train()
    epoch_loss = 0
    epoch_train_mae = 0
    nb_data = 0
    gpu_mem = 0
    for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
        batch_graphs = batch_graphs.to(device)
        batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
        batch_e = batch_graphs.edata['feat'].to(device)
        batch_targets = batch_targets.to(device)
        batch_snorm_n = batch_snorm_n.to(device)
        optimizer.zero_grad()

        try:
            batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
        except KeyError:
            batch_pos_enc = None
        
        if model.pe_init == 'lap_pe':
            batch_pos_enc = handle_lap(model, batch_pos_enc, batch_graphs, device)
            
        batch_scores, __ = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)
        del __

        loss = model.loss(batch_scores, batch_targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.detach().item()
        epoch_train_mae += MAE(batch_scores, batch_targets)
        nb_data += batch_targets.size(0)
    epoch_loss /= (iter + 1)
    epoch_train_mae /= (iter + 1)
    
    return epoch_loss, epoch_train_mae, optimizer

def evaluate_network_sparse(model, device, data_loader, epoch):
    model.eval()
    epoch_test_loss = 0
    epoch_test_mae = 0
    nb_data = 0
    out_graphs_for_lapeig_viz = []
    with torch.no_grad():
        for iter, (batch_graphs, batch_targets, batch_snorm_n) in enumerate(data_loader):
            batch_graphs = batch_graphs.to(device)
            batch_x = batch_graphs.ndata['feat'].to(device)
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_targets = batch_targets.to(device)
            batch_snorm_n = batch_snorm_n.to(device)

            try:
                batch_pos_enc = batch_graphs.ndata['pos_enc'].to(device)
            except KeyError:
                batch_pos_enc = None

            if model.pe_init == 'lap_pe':
                batch_pos_enc = handle_lap(model, batch_pos_enc, batch_graphs, device)
                               
            batch_scores, batch_g = model.forward(batch_graphs, batch_x, batch_pos_enc, batch_e, batch_snorm_n)

            loss = model.loss(batch_scores, batch_targets)
            epoch_test_loss += loss.detach().item()
            epoch_test_mae += MAE(batch_scores, batch_targets)
            nb_data += batch_targets.size(0)
            
            out_graphs_for_lapeig_viz += dgl.unbatch(batch_g)
        epoch_test_loss /= (iter + 1)
        epoch_test_mae /= (iter + 1)
        
    return epoch_test_loss, epoch_test_mae, out_graphs_for_lapeig_viz

