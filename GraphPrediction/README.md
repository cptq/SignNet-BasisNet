#### ZINC results

For implementations of SignNet see `layers/deepsigns.py` and to see its use as a positional encoding see `train/train_ZINC_graph_regression.py`.

To run the experiments, use the scripts in `scripts/`.

For example, to run our SignNet on ZINC with PNA base model, use 
```
bash scripts/ZINC/pna/script_ZINC_PNA_signinv_mask.sh
```

Sample results:

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| PNA | None | N/A | 0.128 | 
| PNA| SignNet | 8 | 0.105 | 
| PNA | SignNet | All | 0.084 | 

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| GatedGCN | None | N/A | 0.252 | 
| GatedGCN | SignNet | 8 | 0.121 | 
| GatedGCN | SignNet | All | 0.102 | 

| Base Model | Positional Encoding | #eigs | test MAE |
|----------|:---:|:---:|:---:|
| Sparse Transformer | None | N/A | 0.283 | 
| Sparse Transformer | SignNet | 16 | 0.115 | 
| Sparse Transformer | SignNet | All | 0.102 | 
