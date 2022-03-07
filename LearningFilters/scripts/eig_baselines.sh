filter_type=${1:-"band"}

net="Transformer"
# no eig
python training.py --filter_type $filter_type --net $net --hidden_channels 48 --num_layers 3 --img_num 50  --lap_method none

# sign flip eig
python training.py --filter_type $filter_type --net $net --hidden_channels 20 --num_layers 3 --img_num 50 --use_eig  --lap_method sign_flip

# absolute value eig
python training.py --filter_type $filter_type --net $net --hidden_channels 20 --num_layers 3 --img_num 50 --use_eig  --lap_method abs_val

