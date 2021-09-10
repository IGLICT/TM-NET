from torch.utils.data import DataLoader
from dataset.dataset_vqvae import VQVAEDataset
from dataset.dataset_spvae import SPVAEDataset
from dataset.dataset_geovae import GeoVAEDataset
from dataset.dataset_latent_geo_2levels import LatentGeo2LevelsDataset
from dataset.dataset_latent_geo_VGG_2levels import LatentGeoVGG2LevelsDataset

def get_dataloader(config, mode):
    if not (mode == 'train' or mode == 'test' or mode == 'val'):
        print('mode should be train test or val, but got {} instead'.format(mode))
        raise NotImplementedError

    module = config['model']['name']
    data_root = config['data']['data_root']
    batch_size = config[mode]['batch_size']
    is_shuffle = config[mode]['is_shuffle']
    num_workers = config[mode]['num_workers']

    if module == 'vqvae':
        category = config['data']['category']
        part_name = config['data']['part_name']
        height = config['data']['height']
        width = config['data']['width']
        dataset = VQVAEDataset(
                            data_root, 
                            mode, 
                            category=category, 
                            part_name=part_name, 
                            height=height, 
                            width=width)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers, drop_last=True)
    elif module == 'spvae':
        dataset = SPVAEDataset(data_root, 
                            mode, )
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
    elif module == 'geovae':
        part_name = config['data']['part_name']
        dataset = GeoVAEDataset(data_root, 
                            mode, 
                            part_name=part_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
    elif module == 'pixelsnail_top_center' or module == 'pixelsnail_bottom_center':
        part_name = config['data']['part_name']
        dataset = LatentGeo2LevelsDataset(data_root, 
                                        mode,
                                        part_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
    elif module == 'pixelsnail_top_others' or module == 'pixelsnail_bottom_others':
        part_name = config['data']['part_name']
        dataset = LatentGeoVGG2LevelsDataset(data_root, 
                                        mode,
                                        part_name)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=is_shuffle, num_workers=num_workers)
    else:
        raise NotImplementedError
    return dataloader

def get_part_names(category):
    if category == 'chair':
        part_names = ['back', 'seat', 'leg_ver_1', 'leg_ver_2', 'leg_ver_3', 'leg_ver_4', 'hand_1', 'hand_2']
    elif category == 'knife':
        part_names = ['part1', 'part2']
    elif category == 'guitar':
        part_names = ['part1', 'part2', 'part3']
    elif category == 'cup':
        part_names = ['part1', 'part2']
    elif category == 'car':
        part_names = ['body', 'left_front_wheel', 'right_front_wheel', 'left_back_wheel', 'right_back_wheel','left_mirror','right_mirror']
    elif category == 'table':
        # part_names = ['surface', 'leg1_1', 'leg1_2', 'leg2_1', 'leg2_2', 'leg3_1', 'leg3_2', 'leg4_1', 'leg4_2']
        part_names = ['surface', 'left_leg1', 'left_leg2', 'left_leg3', 'left_leg4', 'left_leg5', 'left_leg6', 'left_leg7', 'left_leg8', 'right_leg1', 'right_leg2', 'right_leg3', 'right_leg4', 'right_leg5', 'right_leg6', 'right_leg7', 'right_leg8']
    elif category == 'plane':
        part_names = ['body', 'left_wing', 'right_wing', 'left_tail', 'right_tail', 'up_tail', 'down_tail', 'front_gear', 'left_gear', 'right_gear', 'left_engine1', 'right_engine1', 'left_engine2', 'right_engine2']
    else:
        raise Exception("Error")
    return part_names

def get_central_part_name(category):
    if category == 'chair':
        central_part_name = 'back'
    elif category == 'knife':
        central_part_name = 'part2'
    elif category == 'guitar':
        central_part_name = 'part3'
    elif category == 'cup':
        central_part_name = 'part1'
    elif category == 'car':
        central_part_name = 'body'
    elif category == 'table':
        central_part_name = 'surface'
    elif category == 'plane':
        central_part_name = 'body'
    else:
        raise Exception("Error")
    return central_part_name