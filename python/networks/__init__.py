from networks.networks_vqvae import VQVAE
from networks.networks_geovae import GeoVAE
from networks.networks_spvae import SPVAE
from networks.networks_pixelsnail import PixelSNAILTop, PixelSNAIL
from dataset import get_part_names

def get_network(config):
    model_name = config['model']['name']
    part_num = len(get_part_names(config['data']['category']))
    if model_name  == 'vqvae':
        net = VQVAE(
                    config['model']['in_channel'],
                    config['model']['channel'],
                    config['model']['n_res_block'],
                    config['model']['n_res_channel'],
                    config['model']['embed_dim'],
                    config['model']['n_embed'],
                    config['model']['decay'], 
                    config['model']['eps'], 
                    config['model']['beta'],
                    config['model']['stride'],
                    )
    elif model_name  == 'geovae':
        net = GeoVAE(
                    config['model']['geo_hidden_dim'],
                    config['model']['ref_mesh_mat'], 
                    config['train']['device']
                    )
    elif model_name  == 'spvae':
        net = SPVAE(
                    config['model']['geo_hidden_dim'],
                    part_num=part_num
                    )
    elif model_name == 'pixelsnail_top_center' or \
        model_name == 'pixelsnail_top_others':
        net = PixelSNAILTop(shape=[config['model']['shape']*6, config['model']['shape']],
                n_class=config['model']['n_class'],
                channel=config['model']['channel'],
                kernel_size=config['model']['kernel_size'],
                n_block=config['model']['n_block'],
                n_res_block=config['model']['n_res_block'],
                res_channel=config['model']['res_channel'],
                attention=config['model']['attention'],
                dropout=config['model']['dropout'],
                n_cond_res_block=config['model']['n_cond_res_block'],
                cond_res_channel=config['model']['cond_res_channel'],
                cond_res_kernel=config['model']['cond_res_kernel'],
                n_out_res_block=config['model']['n_out_res_block'],
                n_condition_dim=config['model']['n_condition_dim'],
                n_condition_class=config['model']['n_condition_class'],
                n_condition_sub_dim=config['model']['n_condition_sub_dim']
                )
    elif model_name == 'pixelsnail_bottom_center' or \
        model_name == 'pixelsnail_bottom_others':
        net = PixelSNAIL(
                shape=[config['model']['shape']*6, config['model']['shape']],
                n_class=config['model']['n_class'],
                channel=config['model']['channel'],
                kernel_size=config['model']['kernel_size'],
                n_block=config['model']['n_block'],
                n_res_block=config['model']['n_res_block'],
                res_channel=config['model']['res_channel'],
                attention=config['model']['attention'],
                dropout=config['model']['dropout'],
                n_cond_res_block=config['model']['n_cond_res_block'],
                cond_res_channel=config['model']['cond_res_channel'],
                cond_res_kernel=config['model']['cond_res_kernel'],
                n_out_res_block=config['model']['n_out_res_block'],
                n_condition_class=config['model']['n_condition_class']
                )
    else:
        raise ValueError
    return net

def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad
