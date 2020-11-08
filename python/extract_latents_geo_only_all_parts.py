import argparse
import pickle
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
import numpy as np
import lmdb
from tqdm import tqdm
from PIL import Image
from dataset import DownsampleImageDataset, GeometryDataset, GeometryDatasetALL, get_part_names
import GeoVAE
from collections import namedtuple
import scipy.io as sio

CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs'])

def extract_latents_geo(lmdb_env, loader, geo_models, args):
    index = 0
    with lmdb_env.begin(write=True) as txn:
        pbar = tqdm(loader)

        for i, (geo_inputs, origin_geo_inputs, logrmaxs, logrmins, smaxs, smins, filenames) in enumerate(pbar):
            geo_inputs = geo_inputs.to(device).float().contiguous()
            for j in range(geo_inputs.shape[0]):
                geo_zs_all_parts = []

                code_mat_dir = os.path.join(args.mat_dir, filenames[j], 'code.mat')
                code_mat = sio.loadmat(code_mat_dir, verify_compressed_data_integrity=False)
                code_mat = code_mat['code']

                for k in range(len(args.part_names)):
                    geo_zs, geo_outputs, _, _ = geo_models[k](geo_inputs[j:j+1, k, :, :])
                    geo_zs = geo_zs.detach().cpu().numpy()
                    geo_zs = np.concatenate([geo_zs, np.expand_dims(code_mat.flatten(), axis=0)], axis=1)
                    geo_zs_all_parts.append(geo_zs)
                filename = filenames[j]
                geo_zs_all_parts = np.array(geo_zs_all_parts)
                geo_zs_all_parts = geo_zs_all_parts.transpose(1, 0, 2)
                row = CodeRow(ID=filename, geo_zs=geo_zs_all_parts)
                txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                index += 1
                pbar.set_description(f'inserted: {index}')

        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mat_dir', type=str, required=False)
    parser.add_argument('--ref_mesh_mat', type=str, required=False)
    parser.add_argument('--geo_hidden_dim', type=int, default=64)
    parser.add_argument('--geo_ckpt_dir', type=str, required=False)
    
    parser.add_argument('--category', type=str, required=False)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    part_names = get_part_names(args.category)
    args.part_names = part_names
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    
    # geometry dataset
    geo_dataset = GeometryDatasetALL(args.mat_dir, category=args.category)
    geo_loader = DataLoader(geo_dataset, batch_size=1, shuffle=False, num_workers=1)

    geo_models = []
    for part_name in part_names:
        # geo
        geo_model = GeoVAE.GeoVAE(geo_hidden_dim=args.geo_hidden_dim, ref_mesh_mat=args.ref_mesh_mat, device=device).to(device)
        geo_model = geo_model.float()

        # load ckpt
        geo_ckpt = os.path.join(args.geo_ckpt_dir, part_name, 'geovae_newest.pt')
        print('loading {}'.format(geo_ckpt))
        geo_model.load_state_dict(torch.load(geo_ckpt, map_location=torch.device(device)))
        geo_model.eval()
        geo_models.append(geo_model)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)
    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(args.save_path, map_size=map_size)
    args.device = device
    extract_latents_geo(env, geo_loader, geo_models, args)