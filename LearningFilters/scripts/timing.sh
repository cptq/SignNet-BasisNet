filter_type=${1:-"band"}

echo GCN
net="GcnNet"
python training.py --filter_type $filter_type --net $net --hidden_channels 128 --num_layers 4 --img_num 1

echo ChebNet
net="ChebNet"
python training.py --filter_type $filter_type --net $net --hidden_channels 74 --num_layers 4 --img_num 1

echo SignNet with DeepSets
net="DS"
python training.py --filter_type $filter_type --net $net --hidden_channels 32 --num_layers 3 --img_num 1 --use_eig --lap_method sign_inv --sign_inv_net DS

echo BasisNet with DeepSets
python training.py --filter_type $filter_type --net $net --hidden_channels 16 --img_num 1 --use_eig --lap_method basis_inv
