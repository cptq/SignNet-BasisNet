import numpy as np
import os 
import argparse
import torch
from sklearn.metrics import r2_score

from utils import filtering, TwoDGrid, data_to_eig
from models import ChebNet,BernNet,GcnNet,GatNet,ARMANet,GPRNet,MLP,EqDeepSetsEncoder, Transformer
from signbasisnet import SignPlus, IGNBasisInv, IGNShared

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=2000)
parser.add_argument('--lr',type=float,default=0.01)
parser.add_argument('--filter_type',type=str,choices=['low','high','band','rejection','comb','low_band'],default='band')
parser.add_argument('--net',type=str,choices=['ChebNet','BernNet','GcnNet','GatNet','ARMANet','GPRNet', 'MLP', 'DS', 'Linear', 'Transformer'],default='BernNet')
parser.add_argument('--img_num',type=int,default=3)
parser.add_argument('--use_eig', action='store_true')
parser.add_argument('--lap_method', type=str, default='none')
parser.add_argument('--sign_inv_net', type=str, default='DS')
parser.add_argument('--basis_inv_net', type=str, default='IGN')
parser.add_argument('--hidden_channels', type=int, default=32)
parser.add_argument('--num_layers', type=int, default=2)
args = parser.parse_args()
print(args)
print("---------------------------------------------")

if os.path.exists('labels/y_'+args.filter_type+'.npy'):
    y=np.load('labels/y_'+args.filter_type+'.npy')
else:
    y=filtering(args.filter_type)
y=torch.Tensor(y)

dataset = TwoDGrid(root='data/2Dgrid', pre_transform=None)
data=dataset[0]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
y=y.to(device)
data=data.to(device)

eigvals, eigvecs = data_to_eig(data)
eigvals = torch.as_tensor(eigvals).to(device).float()
eigvecs = torch.as_tensor(eigvecs).to(device).float()
N = eigvecs.shape[0]
print('NUMBER OF NODES', N)

def around(x, decimals=5):
    """ round to a number of decimal places """
    return torch.round(x * 10**decimals) / (10**decimals)

if args.lap_method == 'basis_inv':
    rounded_vals = around(eigvals, decimals=5)
    uniq_vals, inv_inds, counts = rounded_vals.unique(return_inverse=True, return_counts=True)
    uniq_mults = counts.unique()
    #print('Unique vals', uniq_vals.shape)
    #print('Unique multiplicities', uniq_mults)
    #print('Vals', rounded_vals)
    #print('Multiplicities', counts)
    #print('prop vecs in higher mult', (counts>1).sum()/counts.shape[0])
    print('prop vecs in higher mult', counts[counts>1].sum()/N)
    sections = torch.cumsum(counts, 0)
    eigenspaces = torch.tensor_split(eigvecs, sections.cpu(), dim=1)[:-1]
    projectors = [V @ V.T for V in eigenspaces]
    projectors = [P.reshape(1,1,N,N) for P in projectors]
    NUM_EIGENSPACES = len(projectors)
    print('Num eigenspaces:', NUM_EIGENSPACES)

    same_size_projs = {mult.item(): [] for mult in uniq_mults}
    for i in range(len(projectors)):
        mult = counts[i].item()
        same_size_projs[mult].append(projectors[i])
    for mult, projs in same_size_projs.items():
        same_size_projs[mult] = torch.cat(projs, dim=0)

if not args.use_eig:
    PE_DIM = 0
elif ('sign_inv' in args.lap_method) or ('basis_inv' in args.lap_method):
    PE_DIM = 32
else:
    PE_DIM = 2*eigvecs.shape[1]



if args.lap_method != 'none':
    assert args.use_eig, 'Specified lap method but not using eigs'

def get_lap_feat(use_eig, eigvals, eigvecs, feat,  lap_method, model):
    if not use_eig:
        return feat
    eigvals_mat = eigvals.unsqueeze(0).repeat(eigvecs.shape[0], 1) # n x k

    if lap_method == 'none':
        feat = torch.cat((feat, eigvecs, eigvals_mat), dim=-1).to(feat)
    elif lap_method == 'abs_val':
        feat = torch.cat((feat, eigvecs.abs(), eigvals_mat), dim=-1).to(feat)
    elif lap_method == 'sign_flip':
        sign_flip = torch.rand(eigvecs.shape[1]).to(device)
        sign_flip[sign_flip>=0.5] = 1.0; sign_flip[sign_flip<0.5] = -1.0
        flipped_eigvecs = eigvecs * sign_flip.unsqueeze(0)
        feat = torch.cat((feat, flipped_eigvecs, eigvals_mat), dim=-1).to(feat)
    elif 'sign_inv' in  lap_method:
        if 'eigval' not in lap_method:
            eigvecs = eigvecs.transpose(1,0).unsqueeze(-1) # n x k -> k x n x 1
            eig_feats = model.sign_inv_net(eigvecs)
            eig_feats = eig_feats.transpose(1,0).reshape(feat.shape[0], -1) # n x d
            eig_feats = torch.cat((eig_feats, eigvals_mat), dim=-1)
            if hasattr(model, 'rho'):
                eig_feats = model.rho(eig_feats)
            feat = torch.cat((feat, eig_feats), dim=-1).to(feat)
        elif 'eigval' in lap_method:
            raise NotImplementedError('Eigval in sign inv not done yet')
            eigvecs = eigvecs.transpose(1,0).unsqueeze(-1) # n x k -> k x n x 1
            eigvals_mat = eigvals_mat.transpose(1,0).unsqueeze(-1)
            eig_feats = model.sign_inv_net(eigvecs, x=eigvals_mat)
            eig_feats = eig_feats.transpose(1,0).reshape(feat.shape[0], -1)
            if hasattr(model, 'rho'):
                eig_feats = model.rho(eig_feats)
            feat = torch.cat((feat, eig_feats), dim=-1).to(feat)
    elif 'basis_inv' in lap_method:
        phi_outs = [model.basis_inv_net(projs, mult) for mult, projs in same_size_projs.items()]
        # b x d x n -> n x bd
        eig_feats = torch.cat([phi_out.reshape(N, -1) for phi_out in phi_outs], dim=-1)
        eig_feats = torch.cat((eig_feats, eigvals_mat), dim=-1)
        if hasattr(model, 'rho'):
            eig_feats = model.rho(eig_feats)
        feat = torch.cat((feat, eig_feats), dim=-1).to(feat)
    else:
        raise ValueError('Invalid eigvec operation')

    return feat

def train(img_idx,model,optimizer):
    model.train()
    optimizer.zero_grad()

    x_tmp=data.x[:,img_idx:img_idx+1]
    x_tmp = get_lap_feat(args.use_eig, eigvals, eigvecs, x_tmp, args.lap_method, model)
    data.x_tmp=x_tmp

    pre=model(data.x_tmp, data.edge_index)
    loss= torch.square(data.m*(pre- y[:,img_idx:img_idx+1])).sum()        
    loss.backward()
    optimizer.step()

    a=pre[data.m==1]    
    b=y[:,img_idx:img_idx+1] 
    b=b[data.m==1] 
    r2=r2_score(b.cpu().detach().numpy(),a.cpu().detach().numpy())

    return loss.item(),r2

def gen_model(args):

    if args.net=='ChebNet':
        model=ChebNet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='BernNet':
        model=BernNet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='GcnNet':
        model=GcnNet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='GatNet':
        model=GatNet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='ARMANet':
        model=ARMANet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='GPRNet':
        model=GPRNet(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='MLP':
        model=MLP(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='DS':
        model=EqDeepSetsEncoder(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    elif args.net=='Linear':
        model=MLP(1+PE_DIM, num_layers=1).to(device)
    elif args.net=='Transformer':
        model=Transformer(1+PE_DIM, hidden_channels=args.hidden_channels, num_layers=args.num_layers).to(device)
    else:
        raise ValueError('Invalid model')
    
    if 'sign_inv' in args.lap_method:
        sign_inv_net = gen_sign_inv(args)
        model.sign_inv_net = sign_inv_net.to(device)
        model.rho = gen_rho(args).to(device)
    elif 'basis_inv' in args.lap_method:
        basis_inv_net = gen_basis_inv(args)
        model.basis_inv_net = basis_inv_net.to(device)
        model.rho = gen_rho(args).to(device)
    return model

def gen_sign_inv(args):
    in_dim = 1
    if 'eigval' in args.lap_method:
        in_dim += 1
        raise NotImplementedError('Eigval in sign inv net not yet implemented')
    if args.sign_inv_net == 'DS':
        sign_inv_net = EqDeepSetsEncoder(in_dim, num_layers=3, use_bn=True)
        sign_inv_net = SignPlus(sign_inv_net)
    elif args.sign_inv_net == 'MLP':
        sign_inv_net = MLP(in_dim, num_layers=args.num_layers, use_bn=True)
        sign_inv_net = SignPlus(sign_inv_net)
    elif args.sign_inv_net == 'Transformer':
        sign_inv_net = Transformer(in_dim, num_layers=2)
        sign_inv_net = SignPlus(sign_inv_net)
    else:
        raise ValueError('Invalid sign inv net')
    return sign_inv_net

def gen_basis_inv(args):
    in_dim = 1
    if args.basis_inv_net == 'IGN':
        basis_inv_net = IGNBasisInv(uniq_mults.tolist(), in_dim, hidden_channels=32)
    elif args.basis_inv_net == 'IGNShared':
        basis_inv_net = IGNShared(uniq_mults.tolist(), in_dim, hidden_channels=16)
    else:
        raise ValueError('Invalid basis invariant network')
    return basis_inv_net


def gen_rho(args):
    in_channels=2*N
    rho = EqDeepSetsEncoder(in_channels, hidden_channels=10, num_layers=3, out_channels=PE_DIM, use_bn=True)
    if 'basis_inv' in args.lap_method and args.basis_inv_net == 'IGNv2':
        in_channels = N + NUM_EIGENSPACES
        rho = EqDeepSetsEncoder(in_channels, hidden_channels=12, num_layers=3, out_channels=PE_DIM, use_bn=True)
    return rho

model = gen_model(args)
print(model)
NUM_PARAMS = sum([p.numel() for p in model.parameters() if p.requires_grad])
print('PARAMETERS:', NUM_PARAMS)


results=[]
for img_idx in range(args.img_num):

    model = gen_model(args)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_r2=0
    min_loss=float('inf')
    re_epoch = 0

    for epoch in range(args.epochs):
        loss,r2=train(img_idx,model,optimizer)
        if(min_loss>loss):
            min_loss=loss
            best_r2=r2
            re_epoch = epoch

        if epoch % 100 == 0:
            print(f'Epoch: {epoch}, Min loss {min_loss:.6f}, Best r2 {best_r2:.4f}')


    results.append([min_loss,best_r2])
    print(f'img: {img_idx+1}, loss= {min_loss:.6f}, r2= {best_r2:.4f}, epoch: {re_epoch}, avg loss: {np.mean(results,axis=0)[0]:.6f}')

loss_mean, r2_mean = np.mean(results, axis=0)
loss_stdev, r2_stdev = np.std(results, axis=0)
print("---------------------------------------------")
print(f'mean loss= {loss_mean:.8f},  stdev loss= {loss_stdev:.8f},   mean r2 acc= {r2_mean:.6f}')
filename = f'results/{args.filter_type}_{args.img_num}.csv'
print(f'Saving results to {filename}')
with open(filename, 'a') as f:
    row = f'{args.net},{loss_mean:.8f},{loss_stdev:.8f},{args.use_eig},{args.lap_method},{args.hidden_channels},{args.num_layers},{NUM_PARAMS}\n'
    f.write(row)
print('---------------------------------------------')
