import torch
from core.config import cfg, update_cfg
from core.train import run 
from core.model import GNN
from core.sign_net import SignNetGNN
from core.transform import EVDTransform

from torch_geometric.datasets import ZINC

def check_distinct(data):
    return len(data.eigen_values) == len(torch.unique(data.eigen_values)) 

def create_dataset(cfg): 
    torch.set_num_threads(cfg.num_workers)
    transform = transform_eval = EVDTransform('sym')
    # transform = transform_eval = None
    root = 'data/ZINC'
    train_dataset = ZINC(root, subset=True, split='train', transform=transform)
    val_dataset = ZINC(root, subset=True, split='val', transform=transform_eval) 
    test_dataset = ZINC(root, subset=True, split='test', transform=transform_eval)   

    train_num_distincts = [check_distinct(data) for data in train_dataset]
    val_num_distincts = [check_distinct(data) for data in val_dataset]
    test_num_distincts = [check_distinct(data) for data in test_dataset]
    print(f"Percentage of graphs with distinct eigenvalues (train): {100*sum(train_num_distincts)/len(train_num_distincts)}%")
    print(f"Percentage of graphs with distinct eigenvalues (vali): {100*sum(val_num_distincts)/len(val_num_distincts)}%")
    print(f"Percentage of graphs with distinct eigenvalues (test): {100*sum(test_num_distincts)/len(test_num_distincts)}%")

    return train_dataset, val_dataset, test_dataset

def create_model(cfg):
    if cfg.model.gnn_type == 'SignNet':
        model = SignNetGNN(None, None,
                           n_hid=cfg.model.hidden_size, 
                           n_out=1, 
                           nl_signnet=cfg.model.num_layers_sign, 
                           nl_gnn=cfg.model.num_layers)
    else:
        model = GNN(None, None, 
                    nhid=cfg.model.hidden_size, 
                    nout=1, 
                    nlayer=cfg.model.num_layers, 
                    gnn_type=cfg.model.gnn_type, 
                    dropout=cfg.train.dropout, 
                    pooling=cfg.model.pool,
                    res=True)

    return model


def train(train_loader, model, optimizer, device):
    total_loss = 0
    N = 0 
    for data in train_loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        optimizer.zero_grad()
        loss = (model(data).squeeze() - y).abs().mean()
        loss.backward()
        total_loss += loss.item() * num_graphs
        optimizer.step()
        N += num_graphs
    return total_loss / N

@torch.no_grad()
def test(loader, model, evaluator, device):
    total_error = 0
    N = 0
    for data in loader:
        if isinstance(data, list):
            data, y, num_graphs = [d.to(device) for d in data], data[0].y, data[0].num_graphs 
        else:
            data, y, num_graphs = data.to(device), data.y, data.num_graphs
        total_error += (model(data).squeeze() - y).abs().sum().item()
        N += num_graphs
    test_perf = - total_error / N
    return test_perf


if __name__ == '__main__':
    # get config 
    cfg.merge_from_file('train/config/zinc.yaml')
    cfg = update_cfg(cfg)
    run(cfg, create_dataset, create_model, train, test)
