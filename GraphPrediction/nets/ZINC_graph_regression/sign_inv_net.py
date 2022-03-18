from layers.deepsigns import GCNDeepSigns, GINDeepSigns, MaskedGINDeepSigns, TransformerDeepSigns

def get_sign_inv_net(net_params):
    assert net_params['sign_inv_net'] is not None, 'did not specify sign inv net'
    print('SIGN INV NET', net_params['sign_inv_net'])
    if net_params['sign_inv_net'] == 'gcn':
        sign_inv_net = GCNDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'], net_params['sign_inv_layers'], net_params['pos_enc_dim'], use_bn=True, dropout=net_params['dropout'], activation=net_params['sign_inv_activation'])
    elif net_params['sign_inv_net'] == 'gin':
        sign_inv_net = GINDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'], net_params['sign_inv_layers'], net_params['pos_enc_dim'], use_bn=True, dropout=net_params['dropout'], activation=net_params['sign_inv_activation'])
    elif net_params['sign_inv_net'] == 'masked_gin':
        sign_inv_net = MaskedGINDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'], net_params['sign_inv_layers'], net_params['pos_enc_dim'], net_params['device'], use_bn=True, dropout=net_params['dropout'], activation=net_params['sign_inv_activation'])
    elif net_params['sign_inv_net'] == 'transformer':
        sign_inv_net = TransformerDeepSigns(1, net_params['hidden_dim'], net_params['phi_out_dim'], net_params['sign_inv_layers'], net_params['pos_enc_dim'], use_bn=True, dropout=net_params['dropout'], activation=net_params['sign_inv_activation'])
    else:
        raise ValueError('Invalid sign inv net')

    return sign_inv_net

