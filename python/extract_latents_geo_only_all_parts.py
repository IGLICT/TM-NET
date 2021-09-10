import argparse
import os
import pickle
from collections import namedtuple

import numpy as np
import scipy.io as sio
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataset import get_part_names
from dataset.dataset_geoall import GeometryAllPartsDataset
from config import load_config
from networks import get_network


def extract_latents_geo(loader, geo_models, args):
    index = 0
    pbar = tqdm(loader)

    for i, (geo_inputs, origin_geo_inputs, logrmaxs, logrmins, smaxs, smins, filenames) in enumerate(pbar):
        geo_inputs = geo_inputs.to(device).float().contiguous()
        for j in range(geo_inputs.shape[0]):
            filename = filenames[j]
            

            code_mat_dir = os.path.join(args.mat_dir, filename, 'code.mat')
            code_mat = sio.loadmat(code_mat_dir, verify_compressed_data_integrity=False)
            code_mat = code_mat['code']

            geo_zs_all_parts = []
            for k in range(len(args.part_names)):
                geo_zs, geo_outputs, _, _ = geo_models[k](geo_inputs[j:j+1, k, :, :])
                geo_zs = geo_zs.detach().cpu().numpy()
                geo_zs = np.concatenate([geo_zs, np.expand_dims(code_mat[k, :].flatten(), axis=0)], axis=1)
                geo_zs_all_parts.append(geo_zs)
            filename = filenames[j]
            geo_zs_all_parts = np.array(geo_zs_all_parts)
            geo_zs_all_parts = geo_zs_all_parts.transpose(1, 0, 2)
            
            
            data_dict = {
                'geo_zs': geo_zs_all_parts[j:j+1, :]
            }
            sub_dir = os.path.join(args.save_path, '{}'.format(filename))
            if not os.path.exists(sub_dir):
                os.makedirs(sub_dir)
            sio.savemat(os.path.join(sub_dir, 'geo_zs.mat'), data_dict)
            index += 1
            pbar.set_description('inserted {}: {} {}'.format(index, filename, geo_zs_all_parts[j:j+1, :].shape))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=True)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--vertex_num', type=int, required=True)
    parser.add_argument('--mode', type=str, required=True)

    parser.add_argument('--geovae_yaml', type=str, required=True)
    parser.add_argument('--geovae_ckpt_dir', type=str, required=True)
    
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    part_names = get_part_names(args.category)
    args.part_names = part_names
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    part_names = get_part_names(args.category)

    geovae_config = load_config(args.geovae_yaml)
    geovae_config['train']['device'] = args.device

    # geometry dataset
    geo_dataset = GeometryAllPartsDataset(args.mat_dir, 
                                    part_names=part_names, 
                                    vertex_num=args.vertex_num, 
                                    mode=args.mode
                                    )
    geo_loader = DataLoader(geo_dataset, batch_size=1, shuffle=False, num_workers=1)

    geo_models = []
    for i, part_name in enumerate(part_names):
        # geo
        geo_model = get_network(geovae_config).to(args.device)
        geo_model = geo_model.float()

        if i > 0:
            part_name = 'leg'

        # load ckpt
        geo_ckpt = os.path.join(args.geovae_ckpt_dir, part_name, 'latest1.pth')
        print('loading {}'.format(geo_ckpt))
        ckpt_dict = torch.load(geo_ckpt, map_location=torch.device(device))
        geo_model.load_state_dict(ckpt_dict['model_state_dict'])
        geo_model.eval()
        geo_models.append(geo_model)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    args.device = device
    extract_latents_geo(geo_loader, geo_models, args)
