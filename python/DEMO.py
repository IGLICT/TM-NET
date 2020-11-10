import argparse
from locale import NOEXPR
import os
import numpy as np
import torch
from torchvision.utils import save_image
import torch.nn.functional as F

from tqdm import tqdm

from VQVAE2Levels import VQVAE, Quantize
from pixelsnail import PixelSNAIL, PixelSNAILTop
from torch.utils.data import DataLoader
from LatentsDatasetWithGeo2levels import LatentsDatasetWithGeo2levels

from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs', 'id_t', 'id_b'])

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
            out, cache = model(row[:, : i + 1, :], condition=condition, cache=cache)
            prob = torch.softmax(out[:, :, i, j] / temperature, 1)
            sample = torch.multinomial(prob, 1).squeeze(-1)
            row[:, i, j] = sample

    return row


def load_model(model, checkpoint, device):
    # ckpt = torch.load(os.path.join('checkpoint', checkpoint))
    ckpt = torch.load(os.path.join(checkpoint), map_location=device)

    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE(in_channel=args.in_channel,
            channel=args.channel,
            n_res_block=args.n_res_block,
            n_res_channel=args.n_res_channel,
            embed_dim=args.embed_dim,
            n_embed=args.n_embed,
            decay=args.decay,
            stride=args.stride)
    elif args.hier == 'top':
        model = PixelSNAILTop(
            [96, 16],
            256,
            # args.channel,
            128,
            5,
            4,
            args.n_res_block,
            # args.n_res_channel,
            128,
            dropout=args.dropout,
            n_out_res_block=args.n_out_res_block,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            n_condition_dim=64,
            n_condition_class=2000
        )
    elif args.hier == 'bottom':
        model = PixelSNAIL(
            [192, 32],
            256,
            args.channel,
            # 128,
            5,
            4,
            args.n_res_block,
            # args.n_res_channel,
            128,
            attention=False,
            dropout=args.dropout,
            n_cond_res_block=args.n_cond_res_block,
            cond_res_channel=args.n_res_channel,
            n_condition_class=256
        )
    if 'model' in ckpt:
        ckpt = ckpt['model']

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model

if __name__ == '__main__':
    # device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--top', type=str)
    parser.add_argument('--bottom', type=str)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--path', type=str, default='./latents_geo_only_test')

    args = parser.parse_args()
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device

    # load ckpt
    model_vqvae = load_model('vqvae', args.vqvae, device)
    model_top = load_model('pixelsnail_top', args.top, device)
    model_bottom = load_model('pixelsnail_bottom', args.bottom, device)

    # prediction directory
    head_tail = os.path.split(args.top)
    head = head_tail[0]
    auto_texture_dir = os.path.join(head, 'auto_texture')
    if not os.path.exists(auto_texture_dir):
        os.mkdir(auto_texture_dir)

    # geometry latents dataset
    # test
    dataset = LatentsDatasetWithGeo2levels(args.path)
    loader = DataLoader(
        dataset, batch_size=1, shuffle=True, num_workers=1, drop_last=False
    )
    ploader = tqdm(loader)
    for k in range(2000):
        for i, (ID, geo_zs, top, bottom) in enumerate(ploader):
            filename = ID[0]
            print(filename)
            if '5b423f02b76df0ec1472a7f3e4685aa' not in filename:
                continue
            head_tail = os.path.split(filename)
            head = head_tail[0]
            basename = os.path.basename(filename)
            basename_without_ext = basename.split('.')[0]
            this_id = basename_without_ext.split('_')[0]

            geo_zs = geo_zs.to(device)
            geo_zs = geo_zs.unsqueeze(1)
            

            top_sample = sample_model(
                model_top, device, args.batch, [96, 16], args.temp, condition=geo_zs
            )
            bottom_sample = sample_model(
                model_bottom, device, args.batch, [192, 32], args.temp, condition=top_sample
            )
            top_sample = top_sample.reshape(-1, 16, 16)
            bottom_sample = bottom_sample.reshape(-1, 32, 32)

            decoded_sample = model_vqvae.decode_code(top_sample, bottom_sample)
            decoded_sample = decoded_sample.clamp(-1, 1)
            
            merged_image = torch.zeros(args.batch, 3, 768, 1024).to(args.device)
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