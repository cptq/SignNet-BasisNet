# Transform for extracting eigenvector and eigenvalues 
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import get_laplacian, to_undirected

# The needed pretransform to save result of EVD
class EVDTransform(object): 
    def __init__(self, norm=None):
        super().__init__()
        self.norm = norm
    def __call__(self, data):
        D, V = EVD_Laplacian(data, self.norm)
        data.eigen_values = D
        data.eigen_vectors = V.reshape(-1) # reshape to 1-d to save 
        return data

def EVD_Laplacian(data, norm=None):
    L_raw = get_laplacian(to_undirected(data.edge_index, num_nodes=data.num_nodes), 
                          normalization=norm, num_nodes=data.num_nodes)
    L = SparseTensor(row=L_raw[0][0], col=L_raw[0][1], value=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

    # L_raw = to_undirected(data.edge_index, num_nodes=data.num_nodes)
    # L = SparseTensor(row=L_raw[0], col=L_raw[1], sparse_sizes=(data.num_nodes, data.num_nodes)).to_dense()

    D, V  = torch.linalg.eigh(L)
    return D, V

from torch_scatter import scatter_add
def to_dense_EVD(eigS, eigV, batch):
    # eigS has the same dimension as batch
    batch_size = int(batch.max()) + 1
    num_nodes = scatter_add(batch.new_ones(eigS.size(0)), batch, dim=0, dim_size=batch_size)
    cum_nodes = torch.cat([batch.new_zeros(1), num_nodes.cumsum(dim=0)])
    max_num_nodes = int(num_nodes.max())

    idx = torch.arange(batch.size(0), dtype=torch.long, device=eigS.device)
    idx = (idx - cum_nodes[batch]) + (batch * max_num_nodes)
    eigS_dense = eigS.new_full([batch_size * max_num_nodes], 0) 
    eigS_dense[idx] = eigS
    eigS_dense = eigS_dense.view([batch_size, max_num_nodes])

    mask = torch.zeros(batch_size * max_num_nodes, dtype=torch.bool, device=eigS.device)
    mask[idx] = 1
    mask = mask.view(batch_size, max_num_nodes)
    mask_squared = mask.unsqueeze(2) * mask.unsqueeze(1)
    eigV_dense = eigV.new_full([batch_size * max_num_nodes * max_num_nodes], 0)
    eigV_dense[mask_squared.reshape(-1)] = eigV
    eigV_dense = eigV_dense.view([batch_size, max_num_nodes, max_num_nodes])

    # eigS_dense: B x N_max
    # eigV_dense: B x N_max x N_max
    return eigS_dense, eigV_dense, mask


def to_dense_list_EVD(eigS, eigV, batch):
    eigS_dense, eigV_dense, mask = to_dense_EVD(eigS, eigV, batch)
    # print(eigS_dense[0])
    # exit(0)

    nmax = eigV_dense.size(1)
    eigS_dense = eigS_dense.unsqueeze(1).repeat(1, nmax, 1)[mask]
    eigV_dense = eigV_dense[mask]
    # eigS_dense: (N1+N2+...+Nb) x N_max
    # eigV_dense: (N1+N2+...+Nb) x N_max 

    return eigS_dense, eigV_dense

