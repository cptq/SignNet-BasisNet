# built off of  https://github.com/chrsmrrs/SpeqNets/blob/master/neural_graph/main_1_alchemy_10K.py
# original author: Christopher Morris
import sys

sys.path.insert(0, '..')
sys.path.insert(0, '.')

import numpy as np
import os.path as osp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.data import DataLoader
from torch_geometric.datasets import TUDataset

from baseline_gin import NetGINE
from sign_net.transform import EVDTransform
from sign_net.sign_net import SignNetGNN

#model_name = 'gine'
model_name = 'signnet'
MIN_LR = 1e-6
PATIENCE = 20

def get_model(model_name):
    if model_name == 'gine':
        hidden_dim = 64
        model = NetGINE(hidden_dim)
    elif model_name == 'signnet':
        hidden_dim = 108
        model = SignNetGNN(6, 4, n_hid = hidden_dim, n_out=12, nl_signnet=8, nl_gnn=16, nl_rho=8, ignore_eigval=False, gnn_type='GINEConv')
        pass
    else:
        raise ValueError('invalid model name')
    return model.to(device)


plot_all = []
results = []
results = []
results_log = []
for _ in range(5):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    plot_it = []
    path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'datasets', "alchemy_full")

    infile = open("train_al_10.index", "r")
    for line in infile:
        indices_train = line.split(",")
        indices_train = [int(i) for i in indices_train]

    infile = open("val_al_10.index", "r")
    for line in infile:
        indices_val = line.split(",")
        indices_val = [int(i) for i in indices_val]

    infile = open("test_al_10.index", "r")
    for line in infile:
        indices_test = line.split(",")
        indices_test = [int(i) for i in indices_test]

    indices = indices_train
    indices.extend(indices_val)
    indices.extend(indices_test)

    transform = EVDTransform('sym')
    dataset = TUDataset(path, name="alchemy_full", transform=transform)[indices]
    print('Num points:', len(dataset))

    mean = dataset.data.y.mean(dim=0, keepdim=True)
    std = dataset.data.y.std(dim=0, keepdim=True)
    dataset.data.y = (dataset.data.y - mean) / std
    mean, std = mean.to(device), std.to(device)

    train_dataset = dataset[0:10000]
    val_dataset = dataset[10000:11000]
    test_dataset = dataset[11000:]

    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    model = get_model(model_name)
    print(model)
    print('Trainable params:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                           factor=0.5, patience=PATIENCE,
                                                           min_lr=MIN_LR/2)


    def train():
        model.train()
        loss_all = 0

        lf = nn.L1Loss()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            loss = lf(model(data), data.y)

            loss.backward()
            loss_all += loss.item() * data.num_graphs
            optimizer.step()
        return (loss_all / len(train_loader.dataset))


    @torch.no_grad()
    def test(loader):
        model.eval()
        error = torch.zeros([1, 12]).to(device)

        for data in loader:
            data = data.to(device)
            error += ((data.y * std - model(data) * std).abs() / std).sum(dim=0)

        error = error / len(loader.dataset)
        error_log = torch.log(error)

        return error.mean().item(), error_log.mean().item()


    best_val_error = None
    for epoch in range(1, 1001):
        start_time = time.time()
        lr = scheduler.optimizer.param_groups[0]['lr']
        loss = train()
        val_error, _ = test(val_loader)

        scheduler.step(val_error)
        if best_val_error is None or val_error <= best_val_error:
            test_error, test_error_log = test(test_loader)
            best_val_error = val_error
        elapsed = time.time() - start_time

        print('Epoch: {:03d}, LR: {:.7f}, Loss: {:.7f}, Validation MAE: {:.7f}, '
              'Test MAE: {:.7f}, Test log MAE: {:.7f}, Time (s): {:.2f}'.format(epoch, lr, loss, val_error, test_error, test_error_log, elapsed))

        if lr <= MIN_LR:
            print("Converged.")
            break


    results.append(test_error)
    results_log.append(test_error_log)


print('Trainable params:', sum([p.numel() for p in model.parameters() if p.requires_grad]))
print("########################")
print('\nTest MAE')
print(results)
results = np.array(results)
print(results.mean(), results.std())

print('\n Test Log MAE')
print(results_log)
results_log = np.array(results_log)
print(results_log.mean(), results_log.std())
