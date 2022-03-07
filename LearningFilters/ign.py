""" From https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch/blob/master/LICENSE
Truong Son Hy, Apache License
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class IGN2to1(nn.Module):
    ''' batch x d x n x n -> batch x d x n
    '''
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, device='cuda', use_bn=True):
        super(IGN2to1, self).__init__()

        self.use_bn = use_bn
        if self.use_bn: self.bns = nn.ModuleList()
        self.equi_layers = nn.ModuleList()
        self.equi_layers.append(layer_2_to_1(in_channels, hidden_channels, device=device))
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.equi_layers.append(layer_1_to_1(hidden_channels, hidden_channels, device=device))
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.equi_layers.append(layer_1_to_1(hidden_channels, hidden_channels, device=device))
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))

        self.fc1 = nn.Linear(hidden_channels, hidden_channels)
        if self.use_bn: self.bns.append(nn.BatchNorm1d(hidden_channels))
        self.fc2 = nn.Linear(hidden_channels, out_channels)

    def forward(self, x):
        for i, layer in enumerate(self.equi_layers):
            x = layer(x)
            x = F.relu(x)
            if self.use_bn: x = self.bns[i](x)
        # b x d x n
        x = x.transpose(2,1) # b x n x d
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = x.transpose(2,1) # b x d x n
        return x


# equi_2_to_2
class layer_2_to_2(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 15

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.diag_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = self.device)
        self.all_bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.diag_bias, self.all_bias])

    def forward(self, inputs):
        m = inputs.size(3)  # extract dimension

        ops_out = contractions_2_to_2(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)

        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
        mat_diag_bias = torch.eye(inputs.size(3)).unsqueeze(dim = 0).unsqueeze(dim = 0).to(device = self.device) * self.diag_bias
        output = output + self.all_bias + mat_diag_bias

        return output

# equi_2_to_1
class layer_2_to_1(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 5

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs):
        m = inputs.size(3)  # extract dimension

        ops_out = contractions_2_to_1(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)  # N x D x B x m

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m

        # bias
        output = output + self.bias

        return output

# equi_1_to_2
class layer_1_to_2(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 5

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs):
        m = inputs.size(2)  # extract dimension

        ops_out = contractions_1_to_2(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)  # N x D x B x m x m

        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
        output = output + self.bias

        return output

# equi_1_to_1
class layer_1_to_1(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m tensor
    :return: output: N x S x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 2

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs):
        m = inputs.size(2)  # extract dimension

        ops_out = contractions_1_to_1(inputs, m, normalization = self.normalization)
        ops_out = torch.stack(ops_out, dim = 2)  # N x D x B x m

        output = torch.einsum('dsb,ndbi->nsi', self.coeffs, ops_out)  # N x S x m

        # bias
        output = output + self.bias

        return output

# equi_basic
class layer_basic(nn.Module):
    '''
    :param name: name of layer
    :param input_depth: D
    :param output_depth: S
    :param inputs: N x D x m x m tensor
    :return: output: N x S x m x m tensor
    '''

    def __init__(self, input_depth, output_depth, normalization = 'inf', normalization_val = 1.0, device = 'cpu'):
        super().__init__()

        self.input_depth = input_depth
        self.output_depth = output_depth
        self.normalization = normalization
        self.normalization_val = normalization_val
        self.device = device

        self.basis_dimension = 4

        # initialization values for variables
        self.coeffs = torch.nn.Parameter(torch.randn(self.input_depth, self.output_depth, self.basis_dimension) * np.sqrt(2.0) / (self.input_depth + self.output_depth), requires_grad = True).to(device = self.device)

        # bias
        self.bias = torch.nn.Parameter(torch.zeros(1, self.output_depth, 1, 1)).to(device = self.device)

        # params
        # self.params = torch.nn.ParameterList([self.coeffs, self.bias])

    def forward(self, inputs):
        m = inputs.size(3)  # extract dimension
        float_dim = np.float32(m)

        # apply ops
        ops_out = []
        # w1 - identity
        ops_out.append(inputs)
        # w2 - sum cols
        sum_of_cols = torch.sum(inputs, dim = 2) / float_dim  # N x D x m
        ops_out.append(torch.cat([sum_of_cols.unsqueeze(dim = 2) for i in range(m)], dim = 2))  # N x D x m x m
        # w3 - sum rows
        sum_of_rows = torch.sum(inputs, dim = 3) / float_dim  # N x D x m
        ops_out.append(torch.cat([sum_of_rows.unsqueeze(dim = 3) for i in range(m)], dim = 3))  # N x D x m x m
        # w4 - sum all
        sum_all = torch.sum(sum_of_rows, dim = 2) / (float_dim ** 2)  # N x D
        out = torch.cat([sum_all.unsqueeze(dim = 2) for i in range(m)], dim = 2)  # N x D x m
        ops_out.append(torch.cat([out.unsqueeze(dim = 3) for i in range(m)], dim = 3))  # N x D x m x m

        ops_out = torch.stack(ops_out, dim = 2)
        output = torch.einsum('dsb,ndbij->nsij', self.coeffs, ops_out)  # N x S x m x m

        # bias
        output = output + self.bias

        return output

# op2_2_to_2
def contractions_2_to_2(inputs, dim, normalization = 'inf', normalization_val = 1.0):  # N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3)   # N x D x m
    sum_diag_part = torch.sum(diag_part, dim = 2).unsqueeze(dim = 2)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim = 3)  # N x D x m
    sum_of_cols = torch.sum(inputs, dim = 2)  # N x D x m
    sum_all = torch.sum(sum_of_rows, dim = 2)  # N x D

    # op1 - (1234) - extract diag
    op1 = torch.diag_embed(diag_part)  # N x D x m x m

    # op2 - (1234) + (12)(34) - place sum of diag on diag
    op2 = torch.diag_embed(torch.cat([sum_diag_part for d in range(dim)], dim = 2))  # N x D x m x m

    # op3 - (1234) + (123)(4) - place sum of row i on diag ii
    op3 = torch.diag_embed(sum_of_rows)  # N x D x m x m

    # op4 - (1234) + (124)(3) - place sum of col i on diag ii
    op4 = torch.diag_embed(sum_of_cols)  # N x D x m x m

    # op5 - (1234) + (124)(3) + (123)(4) + (12)(34) + (12)(3)(4) - place sum of all entries on diag
    op5 = torch.diag_embed(torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2))  # N x D x m x m

    # op6 - (14)(23) + (13)(24) + (24)(1)(3) + (124)(3) + (1234) - place sum of col i on row i
    op6 = torch.cat([sum_of_cols.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op7 - (14)(23) + (23)(1)(4) + (234)(1) + (123)(4) + (1234) - place sum of row i on row i
    op7 = torch.cat([sum_of_rows.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op8 - (14)(2)(3) + (134)(2) + (14)(23) + (124)(3) + (1234) - place sum of col i on col i
    op8 = torch.cat([sum_of_cols.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op9 - (13)(24) + (13)(2)(4) + (134)(2) + (123)(4) + (1234) - place sum of row i on col i
    op9 = torch.cat([sum_of_rows.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op10 - (1234) + (14)(23) - identity
    op10 = inputs  # N x D x m x m

    # op11 - (1234) + (13)(24) - transpose
    op11 = inputs.transpose(3, 2)  # N x D x m x m

    # op12 - (1234) + (234)(1) - place ii element in row i
    op12 = torch.cat([diag_part.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op13 - (1234) + (134)(2) - place ii element in col i
    op13 = torch.cat([diag_part.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op14 - (34)(1)(2) + (234)(1) + (134)(2) + (1234) + (12)(34) - place sum of diag in all entries
    op14 = torch.cat([sum_diag_part for d in range(dim)], dim = 2)
    op14 = torch.cat([op14.unsqueeze(dim = 3) for d in range(dim)], dim = 3) # N x D x m x m

    # op15 - sum of all ops - place sum of all entries in all entries
    op15 = torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2)
    op15 = torch.cat([op15.unsqueeze(dim = 3) for d in range(dim)], dim = 3) # N x D x m x m
    
    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / (dim ** 2)
            op6 = op6 / dim
            op7 = op7 / dim
            op8 = op8 / dim
            op9 = op9 / dim
            op14 = op14 / dim
            op15 = op15 / (dim ** 2)

    return [op1, op2, op3, op4, op5, op6, op7, op8, op9, op10, op11, op12, op13, op14, op15]

# ops_2_to_1
def contractions_2_to_1(inputs, dim, normalization='inf', normalization_val=1.0):  # N x D x m x m
    diag_part = torch.diagonal(inputs, dim1 = 2, dim2 = 3)  # N x D x m

    sum_diag_part = torch.sum(diag_part, dim = 2).unsqueeze(dim = 2)  # N x D x 1
    sum_of_rows = torch.sum(inputs, dim = 3)  # N x D x m
    sum_of_cols = torch.sum(inputs, dim = 2)  # N x D x m
    sum_all = torch.sum(inputs, dim = (2, 3))  # N x D

    # op1 - (123) - extract diag
    op1 = diag_part  # N x D x m

    # op2 - (123) + (12)(3) - tile sum of diag part
    op2 = torch.cat([sum_diag_part for d in range(dim)], dim = 2)  # N x D x m

    # op3 - (123) + (13)(2) - place sum of row i in element i
    op3 = sum_of_rows  # N x D x m

    # op4 - (123) + (23)(1) - place sum of col i in element i
    op4 = sum_of_cols  # N x D x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = torch.cat([sum_all.unsqueeze(dim = 2) for d in range(dim)], dim = 2)  # N x D x m

    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op3 = op3 / dim
            op4 = op4 / dim
            op5 = op5 / (dim ** 2)

    return [op1, op2, op3, op4, op5]

# ops_1_to_2
def contractions_1_to_2(inputs, dim, normalization = 'inf', normalization_val = 1.0):  # N x D x m x m
    sum_all = torch.sum(inputs, dim = 2).unsqueeze(dim = 2)  # N x D x 1

    # op1 - (123) - place on diag
    op1 = torch.diag_embed(inputs, dim1 = 2, dim2 = 3)  # N x D x m x m

    # op2 - (123) + (12)(3) - tile sum on diag
    op2 = torch.diag_embed(torch.cat([sum_all for d in range(dim)], dim = 2), dim1 = 2, dim2 = 3)  # N x D x m x m

    # op3 - (123) + (13)(2) - tile element i in row i
    op3 = torch.cat([torch.unsqueeze(inputs, dim = 2) for d in range(dim)], dim = 2)  # N x D x m x m

    # op4 - (123) + (23)(1) - tile element i in col i
    op4 = torch.cat([torch.unsqueeze(inputs, dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    # op5 - (1)(2)(3) + (123) + (12)(3) + (13)(2) + (23)(1) - tile sum of all entries
    op5 = torch.cat([sum_all for d in range(dim)], dim = 2)  # N x D x m
    op5 = torch.cat([op5.unsqueeze(dim = 3) for d in range(dim)], dim = 3)  # N x D x m x m

    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim
            op5 = op5 / dim

    return [op1, op2, op3, op4, op5]

# ops_1_to_1
def contractions_1_to_1(inputs, dim, normalization = 'inf', normalization_val = 1.0):
    sum_all = torch.sum(inputs, dim = 2).unsqueeze(dim = 2)  # N x D x 1

    # op1 - (12) - identity
    op1 = inputs  # N x D x m

    # op2 - (1)(2) - tile sum of all
    op2 = torch.cat([sum_all for d in range(dim)], dim = 2)  # N x D x m

    if normalization is not None:
        if normalization == 'inf':
            op2 = op2 / dim

    return [op1, op2]
