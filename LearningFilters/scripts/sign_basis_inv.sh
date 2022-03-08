filter_type=${1:-"band"}

net="DS"
python training.py --filter_type $filter_type --net $net --hidden_channels 32 --num_layers 3 --img_num 50 --use_eig --lap_method sign_inv --sign_inv_net DS

python training.py --filter_type $filter_type --net $net --hidden_channels 16 --img_num 50 --use_eig --lap_method basis_inv

net="Transformer"
python training.py --filter_type $filter_type --net $net --hidden_channels 16  --img_num 50 --use_eig --lap_method sign_inv --sign_inv_net DS

python training.py --filter_type $filter_type --net $net --hidden_channels 12  --img_num 50 --use_eig --lap_method basis_inv
