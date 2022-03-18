#!/bin/bash


############
# Usage
############


####################################
# ZINC - 4 SEED RUNS OF EACH EXPTS
####################################

seed0=41
seed1=95
seed2=12
seed3=35
code=main_ZINC_graph_regression.py 
dataset=ZINC

python $code --dataset $dataset --gpu_id 0 --seed $seed0 --config 'configs/gatedgcn/GatedGCN_ZINC_LapPE.json' 
python $code --dataset $dataset --gpu_id 0 --seed $seed1 --config 'configs/gatedgcn/GatedGCN_ZINC_LapPE.json' 
python $code --dataset $dataset --gpu_id 0 --seed $seed2 --config 'configs/gatedgcn/GatedGCN_ZINC_LapPE.json' 
python $code --dataset $dataset --gpu_id 0 --seed $seed3 --config 'configs/gatedgcn/GatedGCN_ZINC_LapPE.json' 



