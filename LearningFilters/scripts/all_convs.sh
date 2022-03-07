filter_type=${1:-"band"}

# only one parameter matrix per layer
for net in GcnNet GPRNet BernNet; do
    for hidden_channels in 32 128; do
        for num_layers in 2 4; do
            python training.py --filter_type $filter_type --net $net --hidden_channels $hidden_channels --num_layers $num_layers --img_num 50
done
done
done

# GAT
net="GatNet"
for hidden_channels in 32 124; do
        for num_layers in 2 4; do
            python training.py --filter_type $filter_type --net $net --hidden_channels $hidden_channels --num_layers $num_layers --img_num 50
done
done

# ARMA
net="ARMANet"
for hidden_channels in 32 70; do
        for num_layers in 2 4; do
            python training.py --filter_type $filter_type --net $net --hidden_channels $hidden_channels --num_layers $num_layers --img_num 50
done
done

# Cheby
net="ChebNet"
for hidden_channels in 32 74; do
        for num_layers in 2 4; do
            python training.py --filter_type $filter_type --net $net --hidden_channels $hidden_channels --num_layers $num_layers --img_num 50
done
done
