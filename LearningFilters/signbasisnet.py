import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ign import IGN2to1

activation_lst = {'relu': nn.ReLU()}

    
class SignPlus(nn.Module):
    # negate v, do not negate x
    def __init__(self, model):
        super(SignPlus, self).__init__()
        self.model = model
    def forward(self, v, *args, x=None):
        if x == None:
            return self.model(v) + self.model(-v)
        else:
            return self.model(torch.cat((v, x), dim=-1)) + self.model(torch.cat((-v, x), dim=-1))


class IGNBasisInv(nn.Module):
    """ IGN based basis invariant neural network
    """
    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2):
        super(IGNBasisInv, self).__init__()
        self.encs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a phi for each choice of multiplicity
            self.encs.append(IGN2to1(1, hidden_channels, mult, num_layers=num_layers))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1


    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.encs[enc_idx](proj)
        return x

class IGNShared(nn.Module):
    """ IGN BasisNet with parameter sharing in phi
    """
    def __init__(self, mult_lst, in_channels, hidden_channels=16, num_layers=2):
        super(IGNShared, self).__init__()
        self.enc = IGN2to1(1, hidden_channels, 1, num_layers=num_layers)
        self.fcs = nn.ModuleList()
        self.mult_to_idx = {}
        curr_idx = 0
        for mult in mult_lst:
            # get a fc for each choice of multiplicity
            self.fcs.append(nn.Linear(1, mult))
            self.mult_to_idx[mult] = curr_idx
            curr_idx += 1

    def forward(self, proj, mult):
        enc_idx = self.mult_to_idx[mult]
        x = self.enc(proj)
        x = x.transpose(2,1) # b x n x d
        x = self.fcs[enc_idx](x)
        x = x.transpose(2,1) # b x d x n
        return x
