import argparse
import os
from locale import NOEXPR

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader

from dataset.dataset_latent_geo_2levels import LatentGeo2LevelsDataset
from config import load_config
from networks import get_network

@torch.no_grad()
def sample_model(model, device, batch, size, temperature, condition=None, initial_row=None):
    row = torch.zeros(batch, *size, dtype=torch.int64).to(device)
    if initial_row is not None:
        row = initial_row
    cache = {}

    for i in tqdm(range(size[0])):
        for j in range(size[1]):
            out, cache, _ = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            # print(prob.max())
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample
    return row

def load_model(checkpoint, config, device):
    # ckpt = torch.load(os.path.join('checkpoint', checkpoint))
    ckpt = torch.load(os.path.join(checkpoint), map_location=device)
    model = get_network(config)
    model.load_state_dict(ckpt['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model

if __name__ == '__main__':
    # device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--part_name', type=str, required=True)
    parser.add_argument('--batch', type=int, default=1)

    parser.add_argument('--vqvae', type=str, required=True)
    parser.add_argument('--vqvae_yaml', type=str, required=True)
    parser.add_argument('--top', type=str, required=True)
    parser.add_argument('--top_yaml', type=str, required=True)
    parser.add_argument('--bottom', type=str, required=True)
    parser.add_argument('--bottom_yaml', type=str, required=True)

    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    vqvae_config = load_config(args.vqvae_yaml)
    top_config = load_config(args.top_yaml)
    bottom_config = load_config(args.bottom_yaml)

    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device

    # load ckpt
    model_vqvae = load_model(args.vqvae, vqvae_config, args.device)
    model_top = load_model(args.top, top_config, args.device)
    model_bottom = load_model(args.bottom, bottom_config, args.device)

    # prediction directory
    head_tail = os.path.split(args.top)
    head = head_tail[0]
    auto_texture_dir = os.path.join(head, 'auto_texture')
    # auto_texture_dir = os.path.join(head, 'val_decode')
    if not os.path.exists(auto_texture_dir):
        os.mkdir(auto_texture_dir)

    # geometry latents dataset
    # test
    dataset = LatentGeo2LevelsDataset(args.path, 
                                    mode='train', 
                                    part_name=args.part_name)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False
    )
    ploader = tqdm(loader)
    for i, (geo_zs, top, bottom, filenames) in enumerate(ploader):
        filename = filenames[0]
            
        print(filename)
        head_tail = os.path.split(filename)
        head = head_tail[0]
        basename = os.path.basename(filename)
        basename_without_ext = basename.split('.')[0]
        this_id = basename_without_ext.split('_')[0]

        top = top.to(device)
        bottom = bottom.to(device)
        top = top.squeeze(0)
        bottom = bottom.squeeze(0)

        geo_zs = geo_zs.to(device)
        geo_zs = geo_zs.unsqueeze(1)
        for k in range(1):
            decoded_sample = model_vqvae.decode_code(top, bottom)
            merged_image = torch.zeros(args.batch, decoded_sample.shape[1], 768, 1024).to(args.device)
            H_begin = [0, 256, 256, 256, 256, 512]
            W_begin = [256, 0, 256, 512, 768, 256]
            print(decoded_sample.shape)
            for b in range(args.batch):
                for i in range(6):
                    merged_image[b, :, H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256] = decoded_sample[b*6+i, :, :, :]
                save_image(
                    merged_image[b, :, :, :], 
                    os.path.join(auto_texture_dir, basename_without_ext+'_'+str(k)+'_'+str(b)+'_sample.png'), 
                    normalize=True, 
                    range=(-1, 1),
                    nrow=1
                    )

            # top_sample = sample_model(
            #     model_top, device, args.batch, [16, 96], args.temp, condition=geo_zs
            # )
            # bottom_sample = sample_model(
            #     model_bottom, device, args.batch, [32, 192], args.temp, condition=top_sample
            # )

            top_sample = sample_model(
                model_top, device, args.batch, [96, 16], args.temp, condition=geo_zs
            )
            bottom_sample = sample_model(
                model_bottom, device, args.batch, [192, 32], args.temp, condition=top_sample
            )
            top_sample = top_sample.reshape(-1, 16, 16)
            print(torch.abs(top_sample-top).sum())
            bottom_sample = bottom_sample.reshape(-1, 32, 32)
            print(torch.abs(bottom_sample-bottom).sum())

            decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
            decoded_sample = decoded_sample.clamp(-1, 1)
            
            merged_image = torch.zeros(args.batch, decoded_sample.shape[1], 768, 1024).to(args.device)
            H_begin = [0, 256, 256, 256, 256, 512]
            W_begin = [256, 0, 256, 512, 768, 256]
            for b in range(args.batch):
                for i in range(6):
                    merged_image[b, :, H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256] = decoded_sample[b*6+i, :, :, :]
                save_image(
                    merged_image[b, :, :, :], 
                    os.path.join(auto_texture_dir, basename_without_ext+'_'+str(k)+'_'+str(b)+'_sample.png'), 
                    normalize=True, 
                    range=(-1, 1),
                    nrow=1
                    )