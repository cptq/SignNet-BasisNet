import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sign_net.model_utils.masked_layers import MaskedBN, MaskedLN
import math

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, freq=100):
        super().__init__()
        self.dim_model = dim_model
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(freq)) / dim_model).reshape(1,1,-1) # d/2 x 1
        self.register_buffer("division_term", division_term)

    def forward(self, pos, mask=None):
        # pos: n x k, k dim is pos. Pos can be continuous value between 0 and 2.
        # mask: n x k
        pos_encoding = torch.zeros(pos.size(0), pos.size(1), self.dim_model).to(pos.device) # n x k x d
        pos = pos.unsqueeze(-1) * self.division_term
        pos_encoding[:, :, 0::2] = torch.sin(pos) 
        pos_encoding[:, :, 1::2] = torch.cos(pos)
        if mask is not None: 
            pos_encoding[~mask] = 0
        # don't handle mask 
        return pos_encoding # n x k x d

class TransformerEncoderLayer(nn.Module):
    ''' Compose with two layers '''
    def __init__(self, d_model, n_head, dropout=0):
        super(TransformerEncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_model//n_head, d_model//n_head, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_model, dropout=dropout)

    def forward(self, x, mask=None):
        # x: n x k x d
        # mask: n x k 
        #assert x[~mask].max() == 0
        x, enc_slf_attn = self.slf_attn(x, x, x, mask=mask)
        x[~mask] = 0
        x = self.pos_ffn(x, mask)
        x[~mask] = 0
        return x, enc_slf_attn

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''
    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3)) # b x h x lq x lq  
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e10)
        attn = self.dropout(F.softmax(attn, dim=-1))
        attn = attn * mask
        output = torch.matmul(attn, v)
        return output, attn

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''
    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v
        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.norm = MaskedLN(d_model)

    def forward(self, q, k, v, mask=None):
        # mask: b x lq x lq
        attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x h x dv   
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x h x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1)   # For head axis broadcasting.
        q, attn = self.attention(q, k, v, mask=attn_mask)

        # Transpose to move the head dimension back: b x lq x h x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q = q + residual
        q = self.norm(q, mask)
        return q, attn

class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''
    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.norm = MaskedLN(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # x: n x k x d
        # mask: n x k 
        # assert x[~mask].max() == 0
        residual = x
        x = F.relu(self.w_1(x))
        if mask is not None:
            x[~mask] = 0
        x = self.w_2(x)
        if mask is not None:
            x[~mask] = 0
        x = self.dropout(x)
        x = x + residual
        x = self.norm(x, mask)
        return x
