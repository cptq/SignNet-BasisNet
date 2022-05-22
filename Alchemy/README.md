## Graph Regression Experiments on Alchemy

### Usage

To run the experiments for SignNet, use `python main_alchemy.py`.

### Implementation

Our SignNet model is implemented in PyTorch Geometric in the `sign_net` folder.

### Setup

Requirements are in `setup.sh`. Simply running `bash setup.sh` will usually make a conda environment called `torch-1-9` that works for these experiments, which you can then activate with `conda activate torch-1-9`.

You may have to edit the `CUDA` variable in `setup.sh` depending on the CUDA version of your GPUs. We use PyTorch 1.9 and PyTorch Geometric 2.0.1.

### Attribution

We built off of the SpeqNets repo by Christopher Morris et al. (no license) [[link](https://github.com/chrsmrrs/SpeqNets/blob/master/neural_graph/main_1_alchemy_10K.py)].

The Alchemy dataset is from "Alchemy: A Quantum Chemistry Dataset for Benchmarking AI Models" Chen et al. 2019  [[arXiv link](https://arxiv.org/abs/1906.09427)].
