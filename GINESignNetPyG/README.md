## SignNet and GINE implentations in PyTorch Geometric

For reproduction of our results on ZINC in the paper.

This is approximately the same as our code used for the Alchemy experiments.

## Setup 

```
# params
# 10/6/2021, newest packages. 
ENV=pyg
CUDA=11.1
TORCH=1.9.1
PYG=2.0.1

# create env 
conda create --name $ENV python=3.9 -y
conda activate $ENV

# install pytorch 
conda install pytorch=$TORCH torchvision torchaudio cudatoolkit=$cuda -c pytorch -c nvidia -y

# install pyg2.0
conda install pyg=$PYG -c pyg -c conda-forge -y

# install ogb 
pip install ogb

# install rdkit
conda install -c conda-forge rdkit -y

# update yacs and tensorboard
pip install yacs==0.1.8 --force  # PyG currently use 0.1.6 which doesn't support None argument. 
pip install tensorboard
pip install matplotlib

```

## Run
```
python -m train.zinc model.gnn_type SignNet
```
