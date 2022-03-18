"""
    Utility file to select GraphNN model as
    selected by the user
"""

from nets.ZINC_graph_regression.gatedgcn_net import GatedGCNNet
from nets.ZINC_graph_regression.gin_net import GINNet
from nets.ZINC_graph_regression.gat_net import GATNet
from nets.ZINC_graph_regression.pna_net import PNANet
from nets.ZINC_graph_regression.transformer_net import TransformerNet

def GatedGCN(net_params):
    return GatedGCNNet(net_params)

def GIN(net_params):
    return GINNet(net_params)

def GAT(net_params):
    return GATNet(net_params)

def PNA(net_params):
    return PNANet(net_params)

def Transformer(net_params):
    return TransformerNet(net_params)

def gnn_model(MODEL_NAME, net_params):
    models = {
        'GatedGCN': GatedGCN,
        'GIN': GIN, 
        'GAT': GAT,     
        'PNA': PNA,
        'Transformer': Transformer
    }
        
    return models[MODEL_NAME](net_params)
