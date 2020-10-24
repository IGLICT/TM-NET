import argparse
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

from tqdm import tqdm

from scheduler import CycleScheduler

from PIL import Image
import torchvision.transforms.functional as F
import os
from dataset import get_part_names
import LatentsDatasetGeoOnly
import scipy.io as sio
import numpy as np
import SPVAE
from collections import namedtuple

CodeRow = namedtuple('CodeRow', ['ID', 'geo_zs'])

def train(epoch, loader, model, optimizer, scheduler, device):
    loader = tqdm(loader)

    for i, (fullname, geo_zs) in enumerate(loader):
        model.zero_grad()
        geo_zs = geo_zs.reshape(geo_zs.shape[0], -1)
        geo_zs = geo_zs.to(device).float().contiguous()
        out = model(geo_zs)
        loss = model.loss_function(*out)
        total_loss = loss['loss']
        recon_loss = loss['Reconstruction_Loss']
        kl_loss = loss['KLD']
        total_loss.backward()
            
        optimizer.step()
        lr = optimizer.param_groups[0]['lr']

        loader.set_description(
            (
                f'epoch: {epoch + 1} '
                f'recon: {recon_loss.item():.4f} '
                f'kl: {kl_loss.item():.4f} '
                f'lr: {lr:.1f}'
            )
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.1)
    parser.add_argument('--sched', type=str)
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--geo_hidden_dim', type=int, default=64)
    parser.add_argument('--ckpt_dir', type=str, required=False)
    parser.add_argument('--category', type=str, required=True)
    parser.add_argument('--device', type=int, default=-1)
    parser.add_argument('--is_test', type=int, default=0)
    parser.add_argument('--load_ckpt', action='store_true')
    args = parser.parse_args()
    print(args)
    num2device_dict = {-1: 'cpu', 0: 'cuda:0', 1: 'cuda:1', 2: 'cuda:2', 3: 'cuda:3'}
    device = num2device_dict[args.device]
    args.device = device
    print(device)
    part_names = get_part_names(args.category)
    args.part_names = part_names


    dataset = LatentsDatasetGeoOnly.LatentsDatasetGeoOnly(args.path)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = SPVAE.SPVAE(geo_hidden_dim=args.geo_hidden_dim, part_num=len(args.part_names)).to(args.device)
    model = model.float()

    if not os.path.exists(args.ckpt_dir) and args.is_test == False:
        os.makedirs(args.ckpt_dir)
    if args.is_test == 0:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        scheduler = None
        if args.sched == 'cycle':
            scheduler = CycleScheduler(
                optimizer, args.lr, n_iter=len(loader) * args.epoch, momentum=None
            )

        if args.ckpt_dir is not None and args.load_ckpt == True:
            model.load_state_dict(torch.load(os.path.join(args.ckpt_dir, 'spvae_newest.pt'), map_location=device))
        
        for i in range(args.epoch):
            train(i, loader, model, optimizer, scheduler, device)
            if (i+1) % 1 == 0:
                torch.save(model.state_dict(), os.path.join(args.ckpt_dir, 'spvae_newest.pt'))
    
