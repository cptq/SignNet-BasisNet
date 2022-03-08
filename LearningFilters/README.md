## Spectral Graph Convolution Experiments
For implementations of SignNet and BasisNet, see `signbasisnet.py` and `training.py`, specifically the functions `gen_sign_inv` and `gen_basis_inv`.

To run the experiments, use the scripts in `scripts/`.

To run our SignNet and BasisNet, use `bash scripts/sign_basis_inv.sh`. You can also pass in a filter type (one of low, high, band, rejection, comb) e.g. `bash scripts/sign_basis_inv.sh rejection`.

### Attribution
These codes are built off of the [[BernNet repo](https://github.com/ivam-he/BernNet)] by He et al. in 2021, which in turn builds off of experimental setups in [[this repo](https://github.com/balcilar/gnn-spectral-expressive-power)] from "Analyzing the Expressive Power of Graph Neural Networks in a Spectral Perspective" by Baliclar et al. 2021.

The Invariant Graph Networks (IGN) implementation in Pytorch is taken from [[this repo](https://github.com/HyTruongSon/InvariantGraphNetworks-PyTorch)] by Hy Truong Son.
