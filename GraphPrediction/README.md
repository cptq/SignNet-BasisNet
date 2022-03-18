#### ZINC results

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
