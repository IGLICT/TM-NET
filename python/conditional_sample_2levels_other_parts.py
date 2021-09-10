import argparse
import os
from locale import NOEXPR

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import torchvision.models as models

from dataset import get_central_part_name, get_part_names
from dataset.dataset_latent_geo_VGG_2levels import LatentGeoVGG2LevelsDataset
from config import load_config
from networks import get_network


def max_min_normalize(AA, min_v, max_v):
    (B, H, W) = AA.shape
    AA = AA.view(B, -1)
    AA -= AA.min(1, keepdim=True)[0]
    AA /= AA.max(1, keepdim=True)[0]
    AA = AA.view(B, H, W)
    AA = AA * (max_v - min_v) + min_v
    return AA

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

def get_central_vgg(vgg_model, filenames, args):
    args.height = 256
    args.width = 256
    args.vgg_height = 224
    args.vgg_width = 224
    vgg_transform = transforms.Compose([
        transforms.Resize((args.height, args.width)),
        transforms.CenterCrop((args.vgg_height, args.vgg_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    H_begin = [0, 256, 256, 256, 256, 512]
    W_begin = [256, 0, 256, 512, 768, 256]
    all_central_vggs = []

    if type(filenames) is str:
        temp = []
        temp.append(filenames)
        filenames = temp
    
    for filename in filenames:
        head_tail = os.path.split(filename)
        head = head_tail[0]
        basename = os.path.basename(filename)
        basename_without_ext = basename.split('.')[0]
        fid = basename_without_ext.split('_')[0]

        central_img_filename = os.path.join(args.central_part_sample_dir, fid+'_'+args.central_part_name+'_0_0_sample.png')
        flag_exist = True
        if os.path.exists(central_img_filename):
            central_image = Image.open(central_img_filename)
            np_central_image = np.array(central_image)
            np_central_image.setflags(write=1)
            np_central_image = np_central_image[:, :, 0:3]
            flag_exist = True
        else:
            print('warning: {} not exists'.format(central_img_filename))
            flag_exist = False
        # central part
        central_patches = torch.zeros(6, 3, args.vgg_height, args.vgg_width)

        for i in range(6):
            central_patches[i, :, :, :] = vgg_transform(
                Image.fromarray(np.uint8(
                np_central_image[H_begin[i]:H_begin[i]+256, W_begin[i]:W_begin[i]+256, :]))
                )
        central_patches = central_patches.to(args.device)
        central_vggs = vgg_model(central_patches)
        all_central_vggs.append(central_vggs)
    all_central_vggs = torch.stack(all_central_vggs)
    return all_central_vggs
        
if __name__ == '__main__':
    # device = 'cuda'
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--central_part_name', type=str, required=True)
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
    
    parser.add_argument('--central_part_sample_dir', type=str, required=True)

    args = parser.parse_args()
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device

    vqvae_config = load_config(args.vqvae_yaml)
    top_config = load_config(args.top_yaml)
    bottom_config = load_config(args.bottom_yaml)

    # torch pretrained vgg model
    vgg16 = models.vgg16(pretrained=True).to(device)
    vgg16.eval()

    # load ckpt
    model_vqvae = load_model(args.vqvae, vqvae_config, args.device)
    model_top = load_model(args.top, top_config, args.device)
    model_bottom = load_model(args.bottom, bottom_config, args.device)

    # prediction directory
    head_tail = os.path.split(args.top)
    head = head_tail[0]
    auto_texture_dir = os.path.join(head, 'auto_texture')
    if not os.path.exists(auto_texture_dir):
        os.mkdir(auto_texture_dir)

    # geometry latents dataset
    # test
    dataset = LatentGeoVGG2LevelsDataset(args.path, mode='test', part_name=args.part_name)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False
    )
    ploader = tqdm(loader)
    for i, (geo_zs, top, bottom, central_vggs, ID) in enumerate(ploader):
        filename = ID[0]
        print(filename)
        head_tail = os.path.split(filename)
        head = head_tail[0]
        basename = os.path.basename(filename)
        basename_without_ext = basename.split('.')[0]
        this_id = basename_without_ext.split('_')[0]
        central_img_filename = os.path.join(args.central_part_sample_dir, this_id+'_'+args.central_part_name+'_0_0_sample.png')
        if not os.path.exists(central_img_filename):
            print('warning: {} not exists'.format(central_img_filename))
            continue
            
        geo_zs = geo_zs.unsqueeze(1)
        geo_zs = geo_zs.to(args.device)
        all_central_vggs = get_central_vgg(vgg16, filename, args)
        all_central_vggs = all_central_vggs.reshape((-1, 1, 1, 6000))
        all_central_vggs = all_central_vggs.to(args.device)
        top_condition = torch.cat([geo_zs, all_central_vggs], 3)
        print(top_condition.shape)
        for k in range(5):
            top_sample = sample_model(
                model_top, device, args.batch, [96, 16], args.temp, condition=top_condition
            )
            bottom_sample = sample_model(
                model_bottom, device, args.batch, [192, 32], args.temp, condition=top_sample
            )
            top_sample = top_sample.reshape(-1, 16, 16)
            bottom_sample = bottom_sample.reshape(-1, 32, 32)

            decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
            decoded_sample = decoded_sample.clamp(-1, 1)
            
            merged_image = torch.zeros(args.batch, 4, 768, 1024).to(args.device)
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